# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ambient agent that processes expense report emails.

This agent receives expense events and routes them through a graph-based workflow:
- Expenses under config.review_threshold are auto-approved immediately.
- Expenses of config.review_threshold or more pass through a Security Checkpoint.
  - Personal data (SSNs and CCs) is redacted.
  - Prompt injection attempts bypass the LLM and are routed directly to human approval.
  - Clean expenses are reviewed by the LLM, then route to human approval.
"""

import base64
import json
import re

from functools import cached_property
from google.adk import Agent, Context, Event, Workflow
from google.adk.apps import App
from google.adk.events import RequestInput
from google.adk.models import Gemini
from google.genai import Client
import os
from pydantic import BaseModel, Field

# Load centralized configuration (sets auth environment variables)
from .config import config


class AIStudioGemini(Gemini):
    """Gemini model class configured explicitly for Google AI Studio to avoid ADC errors."""

    @cached_property
    def api_client(self) -> Client:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get("GEMINI_API_KEY")
        return Client(
            api_key=api_key,
            vertexai=False,
        )


SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CREDIT_CARD_PATTERN = re.compile(
    r"\b(?:\d[ -]?){13,16}\b"
)

PROMPT_INJECTION_KEYWORDS = [
    "ignore",
    "bypass",
    "override",
    "system instructions",
    "system prompt",
    "auto-approve",
    "force approval",
    "you must approve",
    "do not check",
    "skip review",
    "approve this expense",
]


class ExpenseData(BaseModel):
    """Expense report data extracted from the incoming email event."""

    amount: float = Field(description="Expense amount in USD")
    submitter: str = Field(description="Email of the person who submitted")
    category: str = Field(description="Expense category, e.g. travel, meals")
    description: str = Field(description="What the expense is for")
    date: str = Field(description="Date of the expense (YYYY-MM-DD)")


def parse_expense_email(node_input: str) -> Event:
    """Parse a Pub/Sub trigger event and extract expense data."""
    try:
        event = json.loads(node_input)
    except json.JSONDecodeError:
        return Event(output={"error": f"Invalid JSON: {node_input[:200]}"})

    data = event.get("data", {})

    if isinstance(data, str):
        try:
            data = json.loads(base64.b64decode(data))
        except Exception:
            return Event(output={"error": f"Failed to decode data: {data[:200]}"})

    return Event(
        output={
            "amount": float(data.get("amount", 0)),
            "submitter": data.get("submitter", "unknown"),
            "category": data.get("category", "other"),
            "description": data.get("description", ""),
            "date": data.get("date", ""),
        }
    )


def route_by_amount(node_input: dict, ctx: Context) -> Event:
    """Route expenses based on the configured dollar threshold in code."""
    ctx.state["expense_data"] = node_input
    amount = node_input.get("amount", 0)
    if amount >= config.review_threshold:
        return Event(route="NEEDS_REVIEW", output=node_input)
    return Event(route="AUTO_APPROVE", output=node_input)


def security_checkpoint(node_input: dict, ctx: Context) -> Event:
    """Security Checkpoint: scrubs PII and checks for prompt injection."""
    description = node_input.get("description", "")
    redacted = []

    if SSN_PATTERN.search(description):
        description = SSN_PATTERN.sub("[SSN_REDACTED]", description)
        redacted.append("SSN")

    if CREDIT_CARD_PATTERN.search(description):
        description = CREDIT_CARD_PATTERN.sub("[CREDIT_CARD_REDACTED]", description)
        redacted.append("Credit Card")

    cleaned_input = dict(node_input)
    cleaned_input["description"] = description
    ctx.state["expense_data"] = cleaned_input
    if redacted:
        ctx.state["redacted_categories"] = redacted

    desc_lower = description.lower()
    is_injection = any(kw in desc_lower for kw in PROMPT_INJECTION_KEYWORDS)

    if is_injection:
        ctx.state["security_event"] = True
        return Event(route="PROMPT_INJECTION", output=cleaned_input)

    return Event(route="CLEAN", output=cleaned_input)


def auto_approve(node_input: dict) -> Event:
    """Auto-approve a low-value expense and log the decision."""
    log_entry = {
        "severity": "INFO",
        "message": (
            f"Expense auto-approved: ${node_input['amount']:.2f}"
            f" from {node_input['submitter']}"
        ),
        "decision": "approved",
        "amount": node_input["amount"],
        "submitter": node_input["submitter"],
        "category": node_input["category"],
    }
    print(json.dumps(log_entry), flush=True)
    msg = f"Expense auto-approved: ${node_input['amount']:.2f} from {node_input['submitter']}"
    return Event(output={"status": "approved", **node_input}, message=msg)


def emit_expense_alert(
    submitter: str,
    amount: float,
    category: str,
    risk_summary: str,
) -> dict:
    """Emit a structured log alerting finance to review a high-value expense."""
    log_entry = {
        "severity": "WARNING",
        "message": (
            f"Expense review alert: ${amount:.2f} from {submitter} — {risk_summary}"
        ),
        "alert_type": "expense_review",
        "submitter": submitter,
        "amount": amount,
        "category": category,
        "risk_summary": risk_summary,
    }
    print(json.dumps(log_entry), flush=True)
    return {"status": "alert_emitted", "submitter": submitter, "amount": amount}


review_agent = Agent(
    name="review_agent",
    model=AIStudioGemini(
        model=config.model,
    ),
    mode="single_turn",
    instruction="""You are an expense review agent. You receive expense reports
of $100 or more that need review before approval.

Analyze the expense and:
1. Check for risk factors: unusual category for the amount, vague description,
   suspiciously round numbers, very high value (>$1000), or potential policy
   violations.
2. Call the `emit_expense_alert` tool with the submitter, amount, category,
   and a brief risk summary explaining why this expense needs human review.
3. Return a structured review.

Your review MUST include:
- **Amount**: The expense amount
- **Submitter**: Who submitted it
- **Category**: The expense category
- **Risk level**: low, medium, or high
- **Risk factors**: What flags you found (if any)
- **Recommendation**: approve, request-more-info, or escalate""",
    input_schema=ExpenseData,
    tools=[emit_expense_alert],
)


def request_approval(node_input, ctx: Context):  # type: ignore[no-untyped-def]
    """Pause the workflow and wait for a human to approve or reject."""
    expense = ctx.state.get("expense_data", {})
    is_security_event = ctx.state.get("security_event", False)

    if is_security_event:
        msg = "⚠️ SECURITY WARNING: Potential prompt injection detected in description. Routed directly to manager for safety."
    else:
        msg = "Expense requires manager approval. Approve or reject."

    yield RequestInput(
        message=msg,
        payload=expense,
    )


def process_decision(node_input, ctx: Context) -> Event:  # type: ignore[no-untyped-def]
    """Process the human's approval decision and log the outcome."""
    decision = "unknown"
    if isinstance(node_input, dict):
        decision = node_input.get("decision", "unknown")
    elif isinstance(node_input, str):
        decision = "approve" if "approve" in node_input.lower() else "reject"

    approved = decision == "approve"
    expense = ctx.state.get("expense_data", {})
    status = "approved" if approved else "rejected"
    is_security_event = ctx.state.get("security_event", False)
    redacted = ctx.state.get("redacted_categories", [])

    log_severity = "INFO"
    if not approved:
        log_severity = "WARNING"
    if is_security_event:
        log_severity = "CRITICAL"

    log_entry = {
        "severity": log_severity,
        "message": f"Expense {status} by manager" + (" (FLAGGED SECURITY EVENT)" if is_security_event else ""),
        "decision": status,
        "security_event": is_security_event,
        "redacted_categories": redacted,
    }
    print(json.dumps(log_entry), flush=True)

    submitter = expense.get("submitter", "unknown")
    amount = expense.get("amount", 0)
    category = expense.get("category", "")
    description = expense.get("description", "")
    date = expense.get("date", "")

    parts = []
    if is_security_event:
        parts.append("⚠️ [SECURITY WARNING] This expense was flagged for potential prompt injection.")

    parts.append(f"${amount:.2f} expense from {submitter} has been {status}.")

    if description:
        parts.append(f'"{description}" ({category}) on {date}.')

    if redacted:
        parts.append(f"(PII Redacted: {', '.join(redacted)})")

    if approved:
        parts.append(
            "The expense has been logged and will be processed for reimbursement."
        )
    else:
        parts.append(
            "The submitter will be notified and may resubmit with additional documentation."
        )

    msg = " ".join(parts)
    return Event(output={"status": status, "message": msg}, message=msg)


def end_workflow(node_input: dict) -> Event:
    """Final node of the workflow that outputs the result."""
    return Event(output=node_input)


root_agent = Workflow(
    name="ambient_expense_agent",
    edges=[
        ("START", parse_expense_email, route_by_amount),
        (
            route_by_amount,
            {
                "AUTO_APPROVE": auto_approve,
                "NEEDS_REVIEW": security_checkpoint,
            },
        ),
        (
            security_checkpoint,
            {
                "PROMPT_INJECTION": request_approval,
                "CLEAN": review_agent,
            },
        ),
        (review_agent, request_approval),
        (request_approval, process_decision),
        (auto_approve, end_workflow),
        (process_decision, end_workflow),
    ],
)

from google.adk.apps import ResumabilityConfig

app = App(
    root_agent=root_agent,
    name="expense_agent",
    resumability_config=ResumabilityConfig(is_resumable=True),
)
