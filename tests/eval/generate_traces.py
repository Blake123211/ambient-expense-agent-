"""Trace generator for local evaluation of the ambient expense agent.

Runs each scenario in tests/eval/datasets/basic-dataset.json through the local
ADK workflow runner, automating human-in-the-loop approval decisions:
  - Clean requests are approved.
  - Prompt injection attempts are rejected.

Serializes the resulting traces to artifacts/traces/generated_traces.json for
grading by the LLM-as-judge in eval_config.yaml.
"""

import json
from pathlib import Path

from google.adk.runners import Runner

from agent import app

DATASET_PATH = Path("tests/eval/datasets/basic-dataset.json")
OUTPUT_PATH = Path("artifacts/traces/generated_traces.json")


def automated_decision(case_id: str) -> dict:
    """Simulate a human approval decision based on the scenario type."""
    if "injection" in case_id or case_id.startswith("prompt_injection"):
        return {"decision": "reject"}
    return {"decision": "approve"}


def run_scenario(runner: Runner, case: dict) -> dict:
    """Run a single scenario through the workflow, handling any HITL pause."""
    payload = json.dumps(case["input"])
    trace = runner.run(app, input=payload)

    if trace.is_paused_for_input():
        decision = automated_decision(case["case_id"])
        trace = runner.resume(trace, input=decision)

    return {
        "case_id": case["case_id"],
        "description": case["description"],
        "input": case["input"],
        "trace": trace.to_dict(),
    }


def main() -> None:
    dataset = json.loads(DATASET_PATH.read_text())
    runner = Runner()

    traces = [run_scenario(runner, case) for case in dataset]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(traces, indent=2))
    print(f"✓ Generated {len(traces)} traces → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
