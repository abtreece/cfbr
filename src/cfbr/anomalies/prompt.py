"""Frozen system prompt, tool schemas and the per-item message for the investigation agent.
The system prompt is identical for every item in a run (schema built once from the snapshot),
so it caches across a run's investigations. Domain notes come from DECISIONS.md and CLAUDE.md."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from cfbr.anomalies.tools import OUTPUT_FILES

SCHEMA_TABLES = ("games", "plays_raw", "model_plays", "team_status", "lines", "teams_season", "calendar")
ITEM_FIELDS = ("check", "season", "week", "key", "message", "value", "threshold", "game_id", "team")

VERDICT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["verdict", "confidence", "summary", "evidence", "affected", "model_impact", "suggested_action"],
    "properties": {
        "verdict": {"type": "string", "enum": ["data_bug", "real", "inconclusive"]},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "summary": {"type": "string", "description": "At most 280 characters, readable on a phone."},
        "evidence": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["claim", "query_id", "excerpt"],
            "properties": {
                "claim": {"type": "string"},
                "query_id": {"type": "integer", "description": "The result_id of a query or read_output result."},
                "excerpt": {"type": "string", "description": "Copied verbatim from that result, at most 500 characters."}}}},
        "affected": {"type": "object", "additionalProperties": False, "required": ["game_ids", "teams"],
                     "properties": {"game_ids": {"type": "array", "items": {"type": "integer"}},
                                    "teams": {"type": "array", "items": {"type": "string"}}}},
        "model_impact": {"type": "string", "enum": ["none", "small", "material", "unknown"]},
        "suggested_action": {"type": "string", "description": "At most 200 characters; advisory, never executed."},
    },
}

TOOLS = [
    {"name": "query", "strict": True,
     "description": "Run read-only DuckDB SQL on the cfbr store (a snapshot taken at the start of this run). "
                    "Returns a tab-separated table headed by its result_id. At most 200 rows and 20,000 characters "
                    "are shown: aggregate or use LIMIT. "
                    "Only a single SELECT is allowed (DESCRIBE, SHOW and SUMMARIZE work; PRAGMA, writes and most "
                    "table functions are refused); each query has a time and memory limit.",
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["sql"],
                      "properties": {"sql": {"type": "string"}}}},
    {"name": "read_output", "strict": True,
     "description": "Read one of this anomaly week's `cfbr rank` output files. "
                    + "; ".join(f"{k}: {v}" for k, v in OUTPUT_FILES.items()),
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["name"],
                      "properties": {"name": {"type": "string", "enum": list(OUTPUT_FILES)}}}},
    {"name": "submit_verdict", "strict": True,
     "description": "Finish the investigation. Call it exactly once, as your last action.",
     "input_schema": VERDICT_SCHEMA},
]

SYSTEM_TEMPLATE = """You investigate data anomalies flagged by cfbr, a college football rating engine. A deterministic \
check flagged the anomaly in the user message. Decide whether it is a data bug (the source data misrepresents what \
happened) or real (the data is internally consistent and reflects what happened on the field, or a legitimate model \
reaction). Your verdict is advisory: it goes to the operator's phone and to a report. It never changes ratings, bets \
or data.

## Store schema
{schema}

## Data facts
- plays_raw has one row per CFBD play. play_id is VARCHAR: order plays within a game numerically with \
play_id::DECIMAL(38,0), never as text. Late-game order is unreliable: the last play by id misses the final score in \
5-10% of games, so compare maximum scores, not the last row.
- offense_score and defense_score are the score after the play, from the offense's view. Home score = offense_score \
when offense = home, else defense_score.
- games has one row per game: home_points/away_points are finals, home_class/away_class are 'fbs', 'fcs', 'ii', \
'iii' or NULL, plus completed and neutral_site.
- model_plays holds the plays the model fits on, after filtering. Kickoffs, punts, field goals, penalty-only rows, \
timeouts, end-of-period rows, two-point tries and uncategorized rows are dropped. Plays where both offense and \
defense are non-FBS are dropped. Non-FBS teams are pooled into FCS entities (offense_fcs/defense_fcs), and plays with \
a pooled FCS side get reduced weight in the fit.
- Penalty-only rows have yards_gained signed from the offense's view through 2025 week 8 and unsigned after.
- CFBD has listed the same play twice: hundreds of games a season in 2022-2025, rare in 2026.
- team_status is derived from games plus CFBD's FBS team list, per season.
- lines holds the latest spread per game and provider, home perspective (negative = home favored); the consensus is \
the median across providers. Lines are never model inputs.
- A team's rating is its expected point margin against an average FBS team on a neutral field: an opponent-adjusted \
ridge fit on play efficiency (PPA and success rate) with recency weighting, shrunk toward a preseason prior.

## Method
- Start with targeted queries: the game's row, play counts by period or drive, duplicate rows, the plays behind a \
score. Use LIMIT and aggregates. Stop as soon as the evidence settles the question.
- If a query is refused or times out, simplify it (fewer nested subqueries, aggregate earlier) rather than retrying \
the same shape.
- Evidence must come from your tool results. Each evidence item cites the result_id as query_id and an excerpt \
copied verbatim from that result (at most 500 characters). Football knowledge from memory (injuries, weather, real \
results) is not evidence; if it matters, say so in the summary as unverified.
- verdict: data_bug, real, or inconclusive when the evidence does not settle it. data_bug and real need at least one \
evidence item.
- confidence: high only when a result shows it directly.
- model_impact: none if the affected rows never reach model_plays or no FBS team is involved; small if they change \
under about 5% of a team's model plays this season; material if more; unknown otherwise. Check with a query.
- Finish by calling submit_verdict exactly once.
"""


def schema_text(db_path: Path) -> str:
    lines = []
    with duckdb.connect(str(db_path), read_only=True) as con:
        tables = {t for (t,) in con.execute("SHOW TABLES").fetchall()}
        for table in SCHEMA_TABLES:
            if table in tables:
                cols = con.execute(f"DESCRIBE {table}").fetchall()
                lines.append(f"{table}({', '.join(f'{c[0]} {c[1]}' for c in cols)})")
    return "\n".join(lines)


def build_system(schema: str) -> str:
    return SYSTEM_TEMPLATE.replace("{schema}", schema)


def item_message(row: dict, context: list[dict]) -> str:
    payload = {"anomaly": {k: row.get(k) for k in ITEM_FIELDS},
               "context": [{"check": c["check"], "message": c["message"]} for c in context]}
    return ("Investigate this anomaly. Context rows are related single-game efficiency outliers, for information.\n"
            + json.dumps(payload, indent=2) + "\n\nFinish with submit_verdict.")
