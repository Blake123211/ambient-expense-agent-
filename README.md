# Ambient Expense Approval Agent

A production-style AI agent built on Google's Agent Development Kit (ADK) that
automates expense report approval routing — with security containment for PII
and prompt injection attacks built directly into the workflow.

## What it does

The agent receives expense report events (e.g., from a Pub/Sub trigger) and
routes each one through a graph-based workflow:

- **Under $100** → auto-approved immediately, no human involvement.
- **$100 or more** → passes through a **Security Checkpoint** first:
  - Personal data (SSNs, credit card numbers) is detected via regex and
    **redacted** before any LLM ever sees it.
  - **Prompt injection attempts** are detected via keyword signature matching
    and **bypass the LLM entirely** — routed straight to a human for
    approval, never auto-approved.
  - Clean, non-injected requests are reviewed by an LLM agent, which flags
    risk factors before routing to a human for final approval.

Every decision is logged as a structured JSON log entry for auditability.

## Evaluation

Covered by a local evaluation suite (`tests/eval/`) with 5 synthetic
scenarios, each graded by two custom LLM-as-judge metrics:

- **routing_correctness** — did the agent apply the $100 threshold correctly?
- **security_containment** — was PII redacted and was injection contained?

**Result: 5.0 / 5.0 mean score on both metrics, across all 5 scenarios.**

## Tech stack

Google ADK, Gemini, Pydantic, agents-cli

## Running it locally

```bash
cp .env.example .env
pip install -r requirements.txt

make playground
make generate-traces
make grade
```
