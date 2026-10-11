"""Timestamped news/availability archive.

Layout under data/news/:
  raw/{source}/{YYYY-MM-DD}/{fetched}_{sha8}.{xml,html,json}   exact bytes as fetched
  snapshots.jsonl   one line per stored raw file (source, url, fetched_at, sha256, path)
  items.jsonl       one line per new RSS item, with first_seen_at
  state.json        per-source HTTP validators, last content hash, last status

``first_seen_at`` / ``fetched_at`` are the point-in-time keys: a backtest may
only use an item for a game that kicks off after it. Everything is append-only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import tomllib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx

from cfbr.config import REPO_ROOT
from cfbr.news.policy import Policy

log = logging.getLogger(__name__)
NEWS_CONFIG = REPO_ROOT / "config" / "news.toml"
NEWS_DIR = REPO_ROOT / "data" / "news"


@dataclass
class Source:
    name: str
    kind: str
    url: str
    tier: str
    terms: str = ""  # recorded terms-of-use review; required before collecting


def load_sources(path: Path = NEWS_CONFIG) -> tuple[dict, list[Source]]:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    return cfg, [Source(**s) for s in cfg.get("source", [])]


def _now() -> datetime:
    return datetime.now(UTC)


class Archive:
    def __init__(self, root: Path = NEWS_DIR):
        self.root = root
        self.raw = root / "raw"
        self.raw.mkdir(parents=True, exist_ok=True)
        self.state_path = root / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self._seen: set[tuple[str, str]] | None = None

    def save_state(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, sort_keys=True))
        tmp.replace(self.state_path)

    def _append(self, name: str, row: dict) -> None:
        with open(self.root / name, "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    def store(self, source: str, url: str, fetched: datetime, body: bytes, ext: str, extra: dict | None = None) -> str:
        sha = hashlib.sha256(body).hexdigest()
        d = self.raw / source / f"{fetched:%Y-%m-%d}"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{fetched:%Y%m%dT%H%M%SZ}_{sha[:8]}.{ext}"
        path.write_bytes(body)
        rel = str(path.relative_to(self.root))
        self._append("snapshots.jsonl", {"source": source, "url": url, "fetched_at": fetched.isoformat(),
                                         "sha256": sha, "path": rel, **(extra or {})})
        return rel

    def seen(self) -> set[tuple[str, str]]:
        if self._seen is None:
            self._seen = set()
            p = self.root / "items.jsonl"
            if p.exists():
                with open(p) as f:
                    for line in f:
                        r = json.loads(line)
                        self._seen.add((r["source"], r["guid"]))
        return self._seen

    def add_item(self, row: dict) -> bool:
        key = (row["source"], row["guid"])
        if key in self.seen():
            return False
        self._append("items.jsonl", row)
        self._seen.add(key)
        return True


# -- RSS / Atom ------------------------------------------------------------------

def _text(el, *names: str) -> str | None:
    for n in names:
        found = el.find(n)
        if found is not None and (found.text or found.get("href")):
            return (found.text or found.get("href") or "").strip()
    return None


def _parse_date(s: str | None) -> str | None:
    if not s:
        return None
    try:
        return parsedate_to_datetime(s).astimezone(UTC).isoformat()
    except (TypeError, ValueError):
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(UTC).isoformat()
        except ValueError:
            return None


def parse_feed(body: bytes) -> list[dict]:
    """Items from RSS 2.0 or Atom: guid, url, title, summary, published_at."""
    root = ET.fromstring(body)
    atom = "{http://www.w3.org/2005/Atom}"
    out = []
    for it in root.iter("item"):
        link = _text(it, "link")
        out.append({"guid": _text(it, "guid") or link, "url": link, "title": _text(it, "title"),
                    "summary": _text(it, "description"), "published_at": _parse_date(_text(it, "pubDate"))})
    for it in root.iter(f"{atom}entry"):
        link_el = it.find(f"{atom}link")
        link = link_el.get("href") if link_el is not None else None
        out.append({"guid": _text(it, f"{atom}id") or link, "url": link, "title": _text(it, f"{atom}title"),
                    "summary": _text(it, f"{atom}summary", f"{atom}content"),
                    "published_at": _parse_date(_text(it, f"{atom}published", f"{atom}updated"))})
    return [o for o in out if o["guid"]]


def collect_rss(src: Source, client: httpx.Client, policy: Policy, archive: Archive) -> str:
    st = archive.state.setdefault(src.name, {})
    headers = {}
    if st.get("etag"):
        headers["If-None-Match"] = st["etag"]
    if st.get("last_modified"):
        headers["If-Modified-Since"] = st["last_modified"]
    policy.pace(src.url)
    fetched = _now()
    r = client.get(src.url, headers=headers)
    if r.status_code == 304:
        return "not modified"
    r.raise_for_status()
    st["etag"], st["last_modified"] = r.headers.get("ETag"), r.headers.get("Last-Modified")
    sha = hashlib.sha256(r.content).hexdigest()
    if sha == st.get("sha256"):
        return "unchanged"
    path = archive.store(src.name, src.url, fetched, r.content, "xml", {"kind": "rss", "status": r.status_code})
    st["sha256"] = sha
    new = 0
    for item in parse_feed(r.content):
        new += archive.add_item({"source": src.name, "tier": src.tier, **item,
                                 "first_seen_at": fetched.isoformat(), "snapshot": path})
    return f"{new} new items"


# -- Official availability pages (headless browser) --------------------------------

CHALLENGE_MARKERS = ("_Incapsula_Resource", "Request unsuccessful", "cf-challenge", "Just a moment...")


def collect_availability(src: Source, policy: Policy, archive: Archive, user_agent: str) -> str:
    """Render the conference page as a normal visitor would and store the embedded
    report's rendered HTML. Bot challenges are never bypassed."""
    try:
        from playwright.sync_api import Error as PWError
        from playwright.sync_api import sync_playwright
    except ImportError:
        return "skipped: playwright not installed"
    policy.pace(src.url)
    fetched = _now()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                page = browser.new_page(user_agent=user_agent)
                page.goto(src.url, wait_until="networkidle", timeout=60_000)
                page.wait_for_timeout(4_000)
                top = page.content()
                if any(m in top for m in CHALLENGE_MARKERS) and len(top) < 20_000:
                    return "skipped: bot challenge (not bypassed)"
                frames = []
                for fr in page.frames:
                    if fr is page.main_frame:
                        continue
                    reason = policy.check(fr.url) if fr.url.startswith("http") else None
                    if reason:
                        continue
                    try:
                        frames.append({"url": fr.url, "html": fr.content(),
                                       "text": fr.evaluate("() => document.body ? document.body.innerText : ''")})
                    except PWError:
                        continue
            finally:
                browser.close()
    except PWError as e:
        msg = str(e).splitlines()[0]
        return f"skipped: browser unavailable ({msg[:120]})"
    report = [f for f in frames if "hdintelligence" in f["url"]]
    if not report:
        return "no report frame found"
    st = archive.state.setdefault(src.name, {})
    text = "\n\n".join(f["text"] for f in report)
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha == st.get("sha256"):
        return "unchanged"
    body = json.dumps({"page": src.url, "fetched_at": fetched.isoformat(), "frames": report}).encode()
    archive.store(src.name, src.url, fetched, body, "json", {"kind": "availability", "text_sha256": sha})
    st["sha256"] = sha
    return f"stored ({len(text)} chars of report text)"


def collect(only: list[str] | None = None, config: Path | None = None, root: Path | None = None) -> dict[str, str]:
    cfg, sources = load_sources(config or NEWS_CONFIG)
    ua = cfg.get("user_agent", "cfb-ratings-news-collector")
    archive = Archive(root or NEWS_DIR)
    results: dict[str, str] = {}
    with httpx.Client(headers={"User-Agent": ua}, timeout=30, follow_redirects=True) as client:
        policy = Policy(client, ua, float(cfg.get("min_seconds_between_requests_per_host", 5)))
        for src in sources:
            if only and src.name not in only:
                continue
            reason = None if src.terms.strip() else "terms of use not reviewed (add `terms` in config/news.toml)"
            reason = reason or policy.check(src.url)
            if reason:
                results[src.name] = f"skipped: {reason}"
            else:
                try:
                    if src.kind == "rss":
                        results[src.name] = collect_rss(src, client, policy, archive)
                    elif src.kind == "availability_page":
                        results[src.name] = collect_availability(src, policy, archive, ua)
                    else:
                        results[src.name] = f"skipped: unknown kind {src.kind!r}"
                except (httpx.HTTPError, ET.ParseError) as e:
                    results[src.name] = f"error: {type(e).__name__}: {str(e)[:120]}"
            st = archive.state.setdefault(src.name, {})
            st["last_run"], st["last_result"] = _now().isoformat(), results[src.name]
            log.info("news %-26s %s", src.name, results[src.name])
    archive.save_state()
    return results


def purge_source(source: str, root: Path | None = None, apply: bool = False) -> dict[str, int]:
    """Remove a source's raw files, index lines and state. Used when a source is
    dropped because it can't be relied on going forward (e.g. its terms forbid
    automated collection). Takes the collector's lock so it can't race a run."""
    import fcntl
    import shutil

    root = root or NEWS_DIR
    if not root.exists():
        return {}
    with open(root / ".collect.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        raw = root / "raw" / source
        report = {"raw_files": sum(1 for p in raw.rglob("*") if p.is_file()) if raw.exists() else 0}
        for name in ("items.jsonl", "snapshots.jsonl"):
            p = root / name
            lines = p.read_text().splitlines() if p.exists() else []
            keep = [line for line in lines if json.loads(line).get("source") != source]
            report[name] = len(lines) - len(keep)
            if apply and len(keep) != len(lines):
                tmp = p.with_suffix(".tmp")
                tmp.write_text("".join(line + "\n" for line in keep))
                tmp.replace(p)
        st_path = root / "state.json"
        state = json.loads(st_path.read_text()) if st_path.exists() else {}
        report["state"] = int(source in state)
        if apply:
            if raw.exists():
                shutil.rmtree(raw)
            if source in state:
                del state[source]
                tmp = st_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(state, indent=1, sort_keys=True))
                tmp.replace(st_path)
    return report
