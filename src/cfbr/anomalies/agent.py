"""One investigation: a manual tool loop over the Messages API (beta), capped by turns and
wall clock, ending in a submit_verdict call that is checked against the session's own tool
results before it is trusted. API exceptions propagate; run.investigate_doc classifies them."""

from __future__ import annotations

import time
from pathlib import Path

from cfbr.anomalies import ledger
from cfbr.anomalies.prompt import TOOLS, VERDICT_SCHEMA, item_message
from cfbr.anomalies.tools import ToolSession
from cfbr.config import Investigate

BETAS = ["task-budgets-2026-03-13", "server-side-fallback-2026-07-01"]
NUDGE = "Finish by calling submit_verdict."
CUT_OFF = "Cut off at max_tokens before this call finished; call it again, shorter."
REASON_TEXT = {"budget": "Stopped at the turn or time limit before a verdict.",
               "refused": "The model declined this request.",
               "no_verdict": "No verdict was submitted."}


def inconclusive(summary: str) -> dict:
    return {"verdict": "inconclusive", "confidence": "low", "summary": summary[:280], "evidence": [],
            "affected": {"game_ids": [], "teams": []}, "model_impact": "unknown", "suggested_action": ""}


def _norm(text: str) -> str:
    return " ".join(str(text).split())


def grounded(verdict: dict, results: dict[int, str]) -> bool:
    """Every excerpt must appear in the body of a successful result (the `result_id N (...)`
    header line is not evidence) and carry at least 8 non-whitespace characters."""
    evidence = verdict.get("evidence") or []
    if verdict.get("verdict") in ("data_bug", "real") and not evidence:
        return False
    for item in evidence:
        source = results.get(item.get("query_id"))
        excerpt = _norm(item.get("excerpt", ""))
        if source is None or len(excerpt.replace(" ", "")) < 8:
            return False
        body = source.split("\n", 1)[1] if "\n" in source else ""
        if excerpt not in _norm(body):
            return False
    return True


def _clip(verdict: dict) -> dict:
    v = dict(verdict)
    v["summary"] = str(v.get("summary", ""))[:280]
    v["suggested_action"] = str(v.get("suggested_action", ""))[:200]
    v["evidence"] = [{**e, "excerpt": str(e.get("excerpt", ""))[:500]} for e in v.get("evidence") or []]
    return v


def _split_at_fallback(content: list) -> tuple[list, list]:
    """(blocks to echo back, blocks the model acted on this turn). After a mid-output fallback,
    only text survives from before the last `fallback` block; that block and everything after
    it echo normally and are the only blocks acted on."""
    boundary = max((i for i, b in enumerate(content) if b.type == "fallback"), default=None)
    if boundary is None:
        return list(content), list(content)
    echo = [b for b in content[:boundary] if b.type == "text"] + list(content[boundary:])
    return echo, list(content[boundary + 1:])


def _record_usage(ledger_path: Path, msg, purpose: str, request_id: str | None) -> float:
    """One ledger row per attempt: when a fallback ran, top-level usage covers only the final
    attempt and usage.iterations lists every attempt."""
    iterations = getattr(msg.usage, "iterations", None)
    if not iterations:
        return ledger.record(ledger_path, model=msg.model, usage=msg.usage, purpose=purpose, request_id=request_id)
    return sum(ledger.record(ledger_path, model=getattr(it, "model", None) or msg.model, usage=it, purpose=purpose,
                             request_id=request_id) for it in iterations)


def _run_tool(session: ToolSession, block) -> tuple[str, bool]:
    if block.name == "query":
        return session.query(str(block.input.get("sql", "")))
    if block.name == "read_output":
        return session.read_output(str(block.input.get("name", "")))
    return f"unknown tool {block.name!r}", True


def investigate(client, row: dict, context: list[dict], session: ToolSession, system: str, cfg: Investigate,
                ledger_path: Path, clock=time.monotonic) -> dict:
    start = clock()
    messages: list[dict] = [{"role": "user", "content": item_message(row, context)}]
    output_config: dict = {"effort": cfg.effort}
    if cfg.task_budget_tokens:
        output_config["task_budget"] = {"type": "tokens", "total": cfg.task_budget_tokens}
    meta = {"turns": 0, "cost_usd": 0.0, "request_ids": [], "served_by": [], "refusal_category": None}
    verdict, reason = None, None
    for _ in range(cfg.max_turns):
        # Checked between turns only, so one long turn can overrun the wall clock.
        if clock() - start > cfg.wall_clock_minutes * 60:
            reason = "budget"
            break
        with client.beta.messages.stream(model=cfg.model, max_tokens=cfg.max_tokens, system=system,
                                         cache_control={"type": "ephemeral"}, tools=TOOLS, messages=messages,
                                         output_config=output_config, betas=BETAS, fallbacks="default") as stream:
            msg = stream.get_final_message()
            request_id = getattr(stream, "request_id", None) or getattr(msg, "_request_id", None)
        meta["turns"] += 1
        meta["cost_usd"] += _record_usage(ledger_path, msg, row["id"], request_id)
        meta["request_ids"].append(request_id)
        if msg.model not in meta["served_by"]:
            meta["served_by"].append(msg.model)
        echo, acted = _split_at_fallback(msg.content)
        messages.append({"role": "assistant", "content": echo})
        if msg.stop_reason == "refusal":
            reason = "refused"
            meta["refusal_category"] = getattr(getattr(msg, "stop_details", None), "category", None)
            break
        uses = [b for b in acted if b.type == "tool_use"]
        if msg.stop_reason == "max_tokens":
            # Tool input from a cut-off turn is partially parsed (trailing fields silently
            # missing), so nothing in it is run or trusted.
            if uses:
                messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": b.id, "content": CUT_OFF, "is_error": True} for b in uses]})
            else:
                messages.append({"role": "user", "content": NUDGE})
            continue
        submit = next((b for b in uses if b.name == "submit_verdict"), None)
        if submit is not None:
            verdict = dict(submit.input)
            break
        if not uses:
            messages.append({"role": "user", "content": NUDGE})
            continue
        results = []
        for block in uses:
            text, is_error = _run_tool(session, block)
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": text, "is_error": is_error})
        messages.append({"role": "user", "content": results})
    else:
        reason = "budget"

    if verdict is not None and not all(k in verdict for k in VERDICT_SCHEMA["required"]):
        verdict = None
    if verdict is None:
        reason = reason or "no_verdict"
        out = inconclusive(REASON_TEXT[reason])
    else:
        clipped = _clip(verdict)
        if grounded(clipped, session.results):
            out = clipped
        else:
            reason = "ungrounded"
            out = inconclusive(clipped["summary"])
            out["original_verdict"] = clipped
    out["reason"] = reason
    meta["queries"] = len(session.log)
    meta["cost_usd"] = round(meta["cost_usd"], 4)
    meta["duration_s"] = round(clock() - start, 1)
    out["meta"] = meta
    return out
