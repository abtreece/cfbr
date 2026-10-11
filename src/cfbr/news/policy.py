"""Collection policy: robots.txt for our user agent, per-host pacing.

Terms of use are the binding constraint and are reviewed per source by a person
before it is added (recorded as ``terms`` in config/news.toml); the collector
refuses sources without a recorded review. AI agents named in a site's
robots.txt are reported for information but are not, by themselves, a reason
to skip (they target other parties' crawlers, not this collector).
"""

from __future__ import annotations

import time
import urllib.robotparser
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

# AI agents a robots.txt may name; reported by ``ai_opt_out`` for information.
AI_AGENTS = (
    "anthropic-ai", "claudebot", "claude-web", "gptbot", "chatgpt-user", "google-extended", "perplexitybot",
    "perplexity-user", "cohere-ai", "bytespider", "applebot-extended", "meta-externalagent", "ai2bot", "diffbot",
    "omgili",
)


def ai_opt_out(robots_txt: str) -> list[str]:
    """AI agents a robots.txt names (any mention counts as an opt-out)."""
    named = []
    for line in robots_txt.splitlines():
        key, _, val = line.partition(":")
        if key.strip().lower() == "user-agent":
            agent = val.split("#", 1)[0].strip().lower()
            if any(a in agent for a in AI_AGENTS):
                named.append(agent)
    return sorted(set(named))


@dataclass
class Policy:
    client: httpx.Client
    user_agent: str
    min_interval: float = 5.0
    _robots: dict[str, tuple[urllib.robotparser.RobotFileParser, list[str]]] = field(default_factory=dict)
    _last: dict[str, float] = field(default_factory=dict)

    def _host(self, url: str) -> str:
        p = urlsplit(url)
        return f"{p.scheme}://{p.netloc}"

    def check(self, url: str) -> str | None:
        """None if robots.txt allows our user agent to fetch ``url``, else the reason."""
        host = self._host(url)
        if host not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            ai: list[str] = []
            try:
                r = self.client.get(host + "/robots.txt")
                text = r.text if r.status_code == 200 else ""
            except httpx.HTTPError:
                text = ""
            rp.parse(text.splitlines())
            ai = ai_opt_out(text)
            self._robots[host] = (rp, ai)
        rp, _ai = self._robots[host]
        if not rp.can_fetch(self.user_agent, url):
            return "disallowed by robots.txt"
        return None

    def pace(self, url: str) -> None:
        host = self._host(url)
        wait = self._last.get(host, 0.0) + self.min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last[host] = time.monotonic()
