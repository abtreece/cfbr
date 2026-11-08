"""Historical shakedown set: sample up to N fired items per check and shape from
`cfbr anomalies --history` output, so the investigation agent can be judged on 2022–2025
before it runs on Sundays. Only checks the agent investigates (A2, A3, A4) are sampled;
B1/B3 need weekly rank output that history seasons don't have. Sampling is deterministic
(rows ordered by their hash id)."""

from __future__ import annotations

from datetime import datetime


def pattern_group(row: dict, a3_min: int) -> str | None:
    check, value = row["check"], row.get("value")
    if check == "A2":
        if value is None:
            return "A2 no scores"
        return "A2 gap 9-16" if value <= 16 else "A2 gap 17-30" if value <= 30 else "A2 gap >30"
    if check == "A3":
        return "A3 low" if value < a3_min else "A3 high"
    if check == "A4":
        return "A4 4-10" if value <= 10 else "A4 11-40" if value <= 40 else "A4 >40"
    return None


def sample_patterns(rows: list[dict], per_group: int, a3_min: int) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in sorted(rows, key=lambda r: r["id"]):
        group = pattern_group(row, a3_min)
        if group is not None and len(groups.setdefault(group, [])) < per_group:
            groups[group].append({**row, "group": group})
    return [row for group in sorted(groups) for row in groups[group]]


def shakedown_doc(rows: list[dict], seasons: list[int], now: datetime) -> dict:
    stamp = now.isoformat()
    items = [{**row, "status": "pending", "verdict": None, "first_seen_at": stamp, "updated_at": stamp}
             for row in rows]
    return {"title": f"shakedown {seasons[0]}–{seasons[-1]}", "season": seasons[0], "week": 0,
            "generated_at": stamp, "items": items}
