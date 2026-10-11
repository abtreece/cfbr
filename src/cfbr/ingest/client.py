"""CFBD HTTP client with an on-disk response cache and a call ledger.

Every network call is appended to ``{raw_dir}/_calls.jsonl`` so the monthly
free-tier quota can be tracked across commands and days.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from cfbr.config import Config, api_key

log = logging.getLogger(__name__)


class QuotaExceeded(RuntimeError):
    pass


class CFBDClient:
    def __init__(self, cfg: Config, refresh: bool = False):
        self.cfg = cfg
        self.raw_dir: Path = cfg.paths.raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = self.raw_dir / "_calls.jsonl"
        self.refresh = refresh
        self.calls = 0
        self.cache_hits = 0
        self.remaining_header: int | None = None
        self._http: httpx.Client | None = None

    # -- quota -------------------------------------------------------------
    def calls_this_month(self) -> int:
        if not self.ledger.exists():
            return 0
        month = datetime.now(UTC).strftime("%Y-%m")
        n = 0
        with open(self.ledger) as f:
            for line in f:
                if json.loads(line)["ts"].startswith(month):
                    n += 1
        return n

    def budget_left(self) -> int:
        local = self.cfg.ingest.monthly_quota - self.calls_this_month()
        if self.remaining_header is not None:
            return min(local, self.remaining_header)
        return local

    # -- http --------------------------------------------------------------
    @property
    def http(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(
                base_url=self.cfg.ingest.base_url,
                headers={"Authorization": f"Bearer {api_key()}", "Accept": "application/json"},
                timeout=120.0,
            )
        return self._http

    def close(self) -> None:
        if self._http is not None:
            self._http.close()

    @staticmethod
    def cache_path(raw_dir: Path, endpoint: str, year: int, key: str) -> Path:
        slug = endpoint.strip("/").replace("/", "_")
        return raw_dir / slug / str(year) / f"{key}.json"

    def is_cached(self, endpoint: str, year: int, key: str) -> bool:
        return self.cache_path(self.raw_dir, endpoint, year, key).exists()

    def get(self, endpoint: str, params: dict[str, Any], year: int, key: str) -> Any:
        """Return JSON for endpoint+params, from cache unless refresh is set.

        ``year``/``key`` place the response at data/raw/{endpoint}/{year}/{key}.json.
        """
        path = self.cache_path(self.raw_dir, endpoint, year, key)
        if path.exists() and not self.refresh:
            self.cache_hits += 1
            return json.loads(path.read_text())

        if self.budget_left() <= self.cfg.ingest.quota_reserve:
            raise QuotaExceeded(
                f"monthly call budget nearly exhausted ({self.budget_left()} left, "
                f"reserve {self.cfg.ingest.quota_reserve}); resume next month"
            )

        for attempt in range(4):
            resp = self.http.get(endpoint, params=params)
            self._record(endpoint, params, resp.status_code)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(2 ** attempt * 2)
                continue
            break
        resp.raise_for_status()
        hdr = resp.headers.get("X-CallLimit-Remaining")
        if hdr is not None and hdr.isdigit():
            self.remaining_header = int(hdr)
        data = resp.json()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(path)
        return data

    def _record(self, endpoint: str, params: dict[str, Any], status: int) -> None:
        self.calls += 1
        entry = {"ts": datetime.now(UTC).isoformat(), "endpoint": endpoint, "params": params, "status": status}
        with open(self.ledger, "a") as f:
            f.write(json.dumps(entry) + "\n")
        log.debug("API call %s %s -> %s", endpoint, params, status)
