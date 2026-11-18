import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.paths import EVALS_DIR, GOLDEN_DATASET, ROOT
from helpers.cosmos_db_client import query_items

router = APIRouter(tags=["dashboards"])

QUALITY_METRICS = ("groundedness", "relevance", "completeness")
LOW_SCORE_THRESHOLD = 4
REPORT_GLOB = "planner-results*.json"


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _date(timestamp) -> str | None:
    if not isinstance(timestamp, str):
        return None
    try:
        return datetime.fromisoformat(timestamp).date().isoformat()
    except ValueError:
        return None


async def _query_cosmos(query: str) -> list[dict]:
    try:
        return await query_items(query)
    except Exception as error:
        raise HTTPException(
            status_code=502, detail=f"Could not read usage data from Cosmos DB: {error}"
        ) from error


# ---------------------------------------------------------------- Usage


def build_usage_summary(items: list[dict]) -> dict:
    tokens_by_agent: dict[str, float] = defaultdict(float)
    calls_by_agent: dict[str, int] = defaultdict(int)
    tokens_by_model: dict[str, float] = defaultdict(float)
    tokens_by_date: dict[str, float] = defaultdict(float)
    tokens_by_thread: dict[str, float] = defaultdict(float)
    total_tokens = 0.0

    for item in items:
        tokens = _number(item.get("total_tokens")) or 0
        total_tokens += tokens
        if (agent := item.get("agent")) is not None:
            tokens_by_agent[str(agent)] += tokens
            calls_by_agent[str(agent)] += 1
        if (model := item.get("model")) is not None:
            tokens_by_model[str(model)] += tokens
        if (date := _date(item.get("timestamp"))) is not None:
            tokens_by_date[date] += tokens
        if (thread_id := item.get("thread_id")) is not None:
            tokens_by_thread[str(thread_id)] += tokens

    total_calls = len(items)
    agent_token_total = sum(tokens_by_agent.values())
    top_threads = sorted(tokens_by_thread.items(), key=lambda entry: -entry[1])[:10]

    return {
        "total_calls": total_calls,
        "total_tokens": total_tokens,
        "avg_tokens_per_call": (
            round(total_tokens / total_calls, 2) if total_calls else 0
        ),
        "tokens_by_agent": [
            {
                "agent": agent,
                "total_tokens": tokens,
                "percent": tokens / agent_token_total * 100 if agent_token_total else 0,
            }
            for agent, tokens in sorted(tokens_by_agent.items())
        ],
        "calls_by_agent": [
            {"agent": agent, "calls": calls}
            for agent, calls in sorted(calls_by_agent.items())
        ],
        "tokens_by_model": [
            {"model": model, "total_tokens": tokens}
            for model, tokens in sorted(tokens_by_model.items())
        ],
        "daily_tokens": [
            {"date": date, "total_tokens": tokens}
            for date, tokens in sorted(tokens_by_date.items())
        ],
        "top_threads": [
            {"thread_id": thread_id, "total_tokens": tokens}
            for thread_id, tokens in top_threads
        ],
    }


@router.get("/usage")
async def get_usage():
    return build_usage_summary(await _query_cosmos("SELECT * FROM c"))


# ---------------------------------------------------------------- Performance


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def build_performance_summary(items: list[dict]) -> dict:
    scores: dict[str, list[float]] = {metric: [] for metric in QUALITY_METRICS}
    daily_scores: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {metric: [] for metric in QUALITY_METRICS}
    )
    low_scores = []

    for item in items:
        item_scores = {metric: _number(item.get(metric)) for metric in QUALITY_METRICS}
        date = _date(item.get("timestamp"))
        for metric, score in item_scores.items():
            if score is None:
                continue
            scores[metric].append(score)
            if date is not None:
                daily_scores[date][metric].append(score)
        if any(
            score is not None and score < LOW_SCORE_THRESHOLD
            for score in item_scores.values()
        ):
            low_scores.append(
                {
                    "timestamp": item.get("timestamp"),
                    "thread_id": item.get("thread_id"),
                    **item_scores,
                }
            )

    return {
        "rated_responses": len(items),
        "averages": {metric: _mean(scores[metric]) for metric in QUALITY_METRICS},
        "daily": [
            {
                "date": date,
                **{metric: _mean(day_scores[metric]) for metric in QUALITY_METRICS},
            }
            for date, day_scores in sorted(daily_scores.items())
        ],
        "low_scores": low_scores,
        "low_score_threshold": LOW_SCORE_THRESHOLD,
    }


@router.get("/performance")
async def get_performance():
    items = await _query_cosmos(
        "SELECT * FROM c WHERE IS_DEFINED(c.groundedness) "
        "AND IS_DEFINED(c.relevance) AND IS_DEFINED(c.completeness)"
    )
    return build_performance_summary(items)


# ---------------------------------------------------------------- Evaluation


def _is_ratio(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def validate_report(report: object) -> dict:
    if not isinstance(report, dict):
        raise ValueError("The report must be a JSON object.")

    required_fields = {
        "sample_count",
        "exact_matches",
        "exact_match_accuracy",
        "intent_micro_precision",
        "intent_micro_recall",
        "intent_micro_f1",
        "invalid_plan_count",
        "per_intent",
        "mismatches",
    }
    missing_fields = required_fields - set(report)
    if missing_fields:
        raise ValueError(
            "The report is missing required fields: "
            + ", ".join(sorted(missing_fields))
        )

    for field in ("sample_count", "exact_matches", "invalid_plan_count"):
        if (
            not isinstance(report[field], int)
            or isinstance(report[field], bool)
            or report[field] < 0
        ):
            raise ValueError(f"Report field {field!r} must be a non-negative integer.")

    for field in (
        "exact_match_accuracy",
        "intent_micro_precision",
        "intent_micro_recall",
        "intent_micro_f1",
    ):
        if not _is_ratio(report[field]):
            raise ValueError(f"Report field {field!r} must be between 0 and 1.")

    if not isinstance(report["per_intent"], dict):
        raise ValueError("Report field 'per_intent' must be an object.")
    for intent, scores in report["per_intent"].items():
        if not isinstance(scores, dict):
            raise ValueError(f"Invalid per-intent metrics for {intent!r}.")
        for metric in ("precision", "recall", "f1"):
            if not _is_ratio(scores.get(metric)):
                raise ValueError(f"Invalid {metric} score for intent {intent!r}.")

    if not isinstance(report["mismatches"], list):
        raise ValueError("Report field 'mismatches' must be a list.")
    for mismatch in report["mismatches"]:
        if not isinstance(mismatch, dict):
            raise ValueError("Each mismatch must be an object.")
        if not isinstance(mismatch.get("id"), str):
            raise ValueError("Each mismatch must have a string case ID.")
        for field in ("expected_intents", "predicted_intents"):
            intents = mismatch.get(field)
            if not isinstance(intents, list) or not all(
                isinstance(intent, str) for intent in intents
            ):
                raise ValueError(f"Each mismatch must have a string list for {field!r}.")
    return report


def _load_golden_prompts() -> dict[str, str]:
    prompts = {}
    with GOLDEN_DATASET.open(encoding="utf-8") as dataset_file:
        for line in dataset_file:
            if line.strip():
                case = json.loads(line)
                prompts[case["id"]] = case["prompt"]
    return prompts


def build_evaluation_view(report: dict, source: str) -> dict:
    report = validate_report(report)
    warning = None
    golden_prompts: dict[str, str] = {}
    if report["mismatches"]:
        try:
            golden_prompts = _load_golden_prompts()
        except (OSError, json.JSONDecodeError, KeyError) as error:
            warning = f"Could not load golden prompts for mismatch details: {error}"

    return {
        "source": source,
        "summary": {
            field: report[field]
            for field in (
                "sample_count",
                "exact_matches",
                "exact_match_accuracy",
                "intent_micro_precision",
                "intent_micro_recall",
                "intent_micro_f1",
                "invalid_plan_count",
            )
        },
        "per_intent": [
            {
                "intent": intent,
                **{metric: scores[metric] * 100 for metric in ("precision", "recall", "f1")},
            }
            for intent, scores in sorted(report["per_intent"].items())
        ],
        "mismatches": [
            {
                "id": mismatch["id"],
                "prompt": golden_prompts.get(mismatch["id"], ""),
                "expected_intents": mismatch["expected_intents"],
                "predicted_intents": mismatch["predicted_intents"],
                "error": str(mismatch.get("error") or ""),
            }
            for mismatch in report["mismatches"]
        ],
        "warning": warning,
    }


def _saved_reports() -> list[Path]:
    return sorted(
        EVALS_DIR.glob(REPORT_GLOB),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


@router.get("/evaluation/reports")
async def list_reports():
    return {
        "reports": [
            {"name": path.name, "path": path.relative_to(ROOT).as_posix()}
            for path in _saved_reports()
        ]
    }


@router.get("/evaluation/reports/{name}")
async def get_report(name: str):
    # Match against the listing so arbitrary paths can never be read.
    report_path = next((path for path in _saved_reports() if path.name == name), None)
    if report_path is None:
        raise HTTPException(status_code=404, detail="Evaluation report not found.")
    source = report_path.relative_to(ROOT).as_posix()
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return build_evaluation_view(report, source)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise HTTPException(
            status_code=422, detail=f"Could not read {source}: {error}"
        ) from error


@router.post("/evaluation/reports")
async def view_uploaded_report(file: UploadFile = File(...)):
    """Validate and summarize an uploaded report without saving it."""
    try:
        report = json.loads((await file.read()).decode("utf-8"))
        return build_evaluation_view(report, file.filename or "uploaded report")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise HTTPException(
            status_code=422,
            detail=f"Could not read the uploaded evaluation report: {error}",
        ) from error
