import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = ROOT / "evals"
GOLDEN_DATASET = EVALS_DIR / "planner_routing.jsonl"


def _read_report(report_path: Path) -> dict:
    with report_path.open(encoding="utf-8") as report_file:
        return json.load(report_file)


def _read_uploaded_report(uploaded_report) -> dict:
    return json.loads(uploaded_report.getvalue().decode("utf-8"))


def _load_golden_prompts() -> dict[str, str]:
    prompts = {}
    with GOLDEN_DATASET.open(encoding="utf-8") as dataset_file:
        for line in dataset_file:
            if line.strip():
                case = json.loads(line)
                prompts[case["id"]] = case["prompt"]
    return prompts


def _validate_report(report: object) -> dict:
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
        value = report[field]
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"Report field {field!r} must be between 0 and 1.")

    if not isinstance(report["per_intent"], dict):
        raise ValueError("Report field 'per_intent' must be an object.")
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


st.title("Planner Evaluation Dashboard")
st.caption("Routing quality from saved planner evaluation JSON reports.")

report_files = sorted(
    EVALS_DIR.glob("planner-results*.json"),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
uploaded_report = st.file_uploader(
    "Upload an evaluation report",
    type=["json"],
)

selected_report = None
if report_files:
    selected_report = st.selectbox(
        "Saved evaluation report",
        report_files,
        format_func=lambda path: str(path.relative_to(ROOT)),
        disabled=uploaded_report is not None,
    )

if uploaded_report is not None:
    report_source = uploaded_report.name
    try:
        report = _validate_report(_read_uploaded_report(uploaded_report))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        st.error(f"Could not read the uploaded evaluation report: {error}")
        st.stop()
elif selected_report is not None:
    report_source = str(selected_report.relative_to(ROOT))
    try:
        report = _validate_report(_read_report(selected_report))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        st.error(f"Could not read {report_source}: {error}")
        st.stop()
else:
    st.info(
        "No evaluation report found. Run "
        "`python scripts\\evaluate_planner.py --output "
        "evals\\planner-results.json` to create one."
    )
    st.stop()

st.caption(f"Report: {report_source}")

metric_columns = st.columns(6)
metric_columns[0].metric("Cases", report["sample_count"])
metric_columns[1].metric(
    "Exact match",
    f"{report['exact_match_accuracy']:.1%}",
    help=f"{report['exact_matches']} plans matched all expected intents.",
)
metric_columns[2].metric("Micro precision", f"{report['intent_micro_precision']:.1%}")
metric_columns[3].metric("Micro recall", f"{report['intent_micro_recall']:.1%}")
metric_columns[4].metric("Micro F1", f"{report['intent_micro_f1']:.1%}")
metric_columns[5].metric("Invalid plans", report["invalid_plan_count"])

st.subheader("Routing scores by intent")
intent_rows = []
for intent, scores in sorted(report["per_intent"].items()):
    if not isinstance(scores, dict):
        st.error(f"Invalid per-intent metrics for {intent!r}.")
        st.stop()
    score_values = {}
    for metric in ("precision", "recall", "f1"):
        value = scores.get(metric)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            st.error(f"Invalid {metric} score for intent {intent!r}.")
            st.stop()
        score_values[metric] = value * 100
    intent_rows.append({"Intent": intent, **score_values})

if intent_rows:
    intent_scores = pd.DataFrame(intent_rows)
    st.bar_chart(
        intent_scores,
        x="Intent",
        y=["precision", "recall", "f1"],
    )
    st.dataframe(intent_scores, hide_index=True, width="stretch")
else:
    st.info("This report contains no per-intent scores.")

st.subheader("Mismatches")
mismatches = report["mismatches"]
if mismatches:
    try:
        golden_prompts = _load_golden_prompts()
    except (OSError, json.JSONDecodeError, KeyError) as error:
        golden_prompts = {}
        st.warning(f"Could not load golden prompts for mismatch details: {error}")

    mismatch_rows = []
    for mismatch in mismatches:
        if not isinstance(mismatch, dict):
            st.error("Report contains an invalid mismatch entry.")
            st.stop()
        mismatch_rows.append(
            {
                "Case": mismatch["id"],
                "Prompt": golden_prompts.get(mismatch.get("id"), ""),
                "Expected intents": ", ".join(
                    mismatch["expected_intents"]
                ),
                "Predicted intents": ", ".join(
                    mismatch["predicted_intents"]
                ),
                "Error": mismatch.get("error", ""),
            }
        )
    st.dataframe(
        pd.DataFrame(mismatch_rows),
        hide_index=True,
        width="stretch",
    )
else:
    st.success("All prompts matched their expected intents.")
