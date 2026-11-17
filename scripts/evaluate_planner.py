import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.azure_config import llm
from nodes.planner import (
    SUPPORTED_INTENTS,
    build_planner_prompt,
    normalize_execution_plan,
)

DEFAULT_DATASET = ROOT / "evals" / "planner_routing.jsonl"


def load_dataset(path: Path = DEFAULT_DATASET) -> list[dict]:
    cases = []
    seen_ids = set()

    with path.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on dataset line {line_number}: {error}"
                ) from error

            if not isinstance(case, dict) or set(case) != {
                "id",
                "prompt",
                "uploaded_file",
                "expected_intents",
            }:
                raise ValueError(
                    f"Dataset line {line_number} must contain exactly id, prompt, "
                    "uploaded_file, and expected_intents."
                )
            if (
                not isinstance(case["id"], str)
                or not case["id"]
                or case["id"] in seen_ids
            ):
                raise ValueError(
                    f"Dataset line {line_number} has a missing or duplicate id."
                )
            if not isinstance(case["prompt"], str) or not case["prompt"].strip():
                raise ValueError(
                    f"Dataset case {case['id']!r} must have a non-empty prompt."
                )
            if case["uploaded_file"] is not None and not isinstance(
                case["uploaded_file"], str
            ):
                raise ValueError(
                    f"Dataset case {case['id']!r} uploaded_file must be a string or null."
                )
            expected_intents = case["expected_intents"]
            if (
                not isinstance(expected_intents, list)
                or not expected_intents
                or any(
                    not isinstance(intent, str) or intent not in SUPPORTED_INTENTS
                    for intent in expected_intents
                )
                or len(set(expected_intents)) != len(expected_intents)
            ):
                raise ValueError(
                    f"Dataset case {case['id']!r} has invalid expected_intents."
                )

            seen_ids.add(case["id"])
            cases.append(case)

    if not 50 <= len(cases) <= 100:
        raise ValueError(
            f"Planner routing dataset must contain 50-100 cases; found {len(cases)}."
        )
    return cases


def score_routing(cases: list[dict], predictions: dict[str, list[str] | None]) -> dict:
    exact_matches = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    per_intent_counts = {
        intent: {"true_positive": 0, "false_positive": 0, "false_negative": 0}
        for intent in SUPPORTED_INTENTS
    }
    mismatches = []
    invalid_plan_count = 0

    for case in cases:
        expected = set(case["expected_intents"])
        predicted_intents = predictions.get(case["id"])
        if predicted_intents is None:
            invalid_plan_count += 1
            predicted = set()
        else:
            predicted = set(predicted_intents)

        if predicted == expected:
            exact_matches += 1
        else:
            mismatch = {
                "id": case["id"],
                "expected_intents": case["expected_intents"],
                "predicted_intents": (
                    predicted_intents if predicted_intents is not None else []
                ),
            }
            if predicted_intents is None:
                mismatch["error"] = "Planner returned an invalid plan."
            mismatches.append(mismatch)

        for intent in SUPPORTED_INTENTS | predicted:
            true_positive = int(intent in expected and intent in predicted)
            false_positive = int(intent in predicted and intent not in expected)
            false_negative = int(intent in expected and intent not in predicted)
            counts = per_intent_counts.setdefault(
                intent,
                {
                    "true_positive": 0,
                    "false_positive": 0,
                    "false_negative": 0,
                },
            )
            counts["true_positive"] += true_positive
            counts["false_positive"] += false_positive
            counts["false_negative"] += false_negative
            true_positives += true_positive
            false_positives += false_positive
            false_negatives += false_negative

    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    per_intent = {}
    for intent, counts in sorted(per_intent_counts.items()):
        intent_precision = _ratio(
            counts["true_positive"],
            counts["true_positive"] + counts["false_positive"],
        )
        intent_recall = _ratio(
            counts["true_positive"],
            counts["true_positive"] + counts["false_negative"],
        )
        per_intent[intent] = {
            **counts,
            "precision": intent_precision,
            "recall": intent_recall,
            "f1": _f1(intent_precision, intent_recall),
        }

    sample_count = len(cases)
    return {
        "sample_count": sample_count,
        "exact_matches": exact_matches,
        "exact_match_accuracy": _ratio(exact_matches, sample_count),
        "intent_micro_precision": precision,
        "intent_micro_recall": recall,
        "intent_micro_f1": _f1(precision, recall),
        "invalid_plan_count": invalid_plan_count,
        "per_intent": per_intent,
        "mismatches": mismatches,
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


async def evaluate(cases: list[dict]) -> dict:
    predictions = {}
    errors = {}
    for index, case in enumerate(cases, start=1):
        print(
            f"[{index}/{len(cases)}] Evaluating {case['id']}...",
            file=sys.stderr,
            flush=True,
        )
        prompt = build_planner_prompt(case["prompt"], case["uploaded_file"])
        response = await llm.ainvoke(prompt)
        try:
            execution_plan = normalize_execution_plan(json.loads(response.content))
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            predictions[case["id"]] = None
            errors[case["id"]] = f"{type(error).__name__}: {error}"
            print(f"  Invalid plan: {errors[case['id']]}", file=sys.stderr, flush=True)
            continue

        predictions[case["id"]] = [
            task["intent"] for task in execution_plan["tasks"]
        ]
        print(
            f"  Predicted intents: {', '.join(predictions[case['id']])}",
            file=sys.stderr,
            flush=True,
        )

    report = score_routing(cases, predictions)
    for mismatch in report["mismatches"]:
        case_id = mismatch["id"]
        if case_id in errors:
            mismatch["error"] = errors[case_id]
    return report


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate planner intent routing on the golden prompt dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="JSONL golden dataset path (default: evals/planner_routing.jsonl).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N cases (useful for a quick smoke test).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Also write the final JSON report to this path.",
    )
    args = parser.parse_args()
    cases = load_dataset(args.dataset)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be greater than zero.")
        cases = cases[: args.limit]

    report_text = json.dumps(await evaluate(cases), indent=2)
    print(report_text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_text + "\n", encoding="utf-8")
        print(f"Saved report to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
