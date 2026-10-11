"""Push alerts for unattended runs, through ntfy (https://ntfy.sh, or a self-hosted server).

The topic name comes from ``NTFY_TOPIC`` (env or the gitignored .keys file) and acts
as the only secret: anyone who knows it can read the alerts, so it's long and
random. With no topic set, alerts are logged and nothing is sent. Sending never
raises: an alert must not turn a working run into a failed one.
"""

from __future__ import annotations

import logging
import os

import httpx

from cfbr.config import REPO_ROOT, Config

log = logging.getLogger(__name__)
TOPIC_KEY = "NTFY_TOPIC"


def ntfy_topic() -> str | None:
    if os.environ.get(TOPIC_KEY):
        return os.environ[TOPIC_KEY]
    keys_file = REPO_ROOT / ".keys"
    if keys_file.exists():
        for line in keys_file.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == TOPIC_KEY and value.strip():
                return value.strip().strip('"').strip("'")
    return None


def send(cfg: Config, title: str, message: str, priority: str = "high") -> bool:
    """Push ``message`` (kept short: phones truncate). Returns whether it was delivered."""
    topic = ntfy_topic()
    if not cfg.alerts.enabled or topic is None:
        log.warning("alert not sent (%s): %s: %s", "disabled" if not cfg.alerts.enabled else "no NTFY_TOPIC",
                    title, message)
        return False
    try:
        r = httpx.post(f"{cfg.alerts.server.rstrip('/')}/{topic}", content=message.encode()[:3500],
                       headers={"Title": title, "Priority": priority, "Tags": "football"}, timeout=15.0)
        r.raise_for_status()
        return True
    except Exception as e:  # noqa: BLE001 - alerting is best effort
        log.warning("alert delivery failed (%s): %s: %s", e, title, message)
        return False
