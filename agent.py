from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import assemblyai, cartesia, openai, silero

from prompting import build_system_prompt

load_dotenv()


CUSTOMER_PROFILE: dict[str, Any] = {
    "campaign_type": "pre_approved",  # pre_approved | cold
    "script_version": "A",  # A | B (pre-approved only)
    "customer_name": "Jordan Alvarez",
    "vehicle_year": 2021,
    "vehicle_make": "Toyota",
    "vehicle_model": "Camry",
    "state": "FL",
    "advance_amount_usd": 2400.00,
    "max_term_months": 12,
    "vdc_total_usd": 324.00,
    "estimated_monthly_payment_usd": 261.75,
    "estimated_total_interest_usd": 417.00,
    "lead_source": "web inquiry",
    "phone_number": "+1-415-555-0199",
    "contact_channel_preference": "phone",
    "language_preference": "English",
}


@dataclass
class CallDispositionState:
    customer_reached: bool = False
    customer_interested: bool | None = None
    not_interested_reason: str | None = None
    requested_loan_amount_usd: float | None = None
    referred_to_loan_officer: bool = False
    qualification_notes: str | None = None
    next_step: str | None = None
    next_step_datetime: str | None = None
    best_phone_number: str | None = None
    notes: str = ""


def _as_money(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"${value:.2f}"
    return "N/A"


def build_customer_context(customer: dict[str, Any]) -> str:
    campaign_type = str(customer.get("campaign_type", "pre_approved")).lower()
    script_version = str(customer.get("script_version", "A")).upper()

    lines = [
        f"- Customer Name: {customer['customer_name']}",
        (
            "- Vehicle: "
            f"{customer['vehicle_year']} {customer['vehicle_make']} "
            f"{customer['vehicle_model']}"
        ),
        f"- Campaign Type: {campaign_type}",
        f"- Script Version: {script_version if campaign_type == 'pre_approved' else 'N/A'}",
        f"- State: {customer.get('state', 'Unknown')}",
        f"- Lead Source: {customer.get('lead_source', 'N/A')}",
        f"- Phone Number: {customer['phone_number']}",
        f"- Contact Preference: {customer['contact_channel_preference']}",
        f"- Language Preference: {customer['language_preference']}",
        f"- Today's Date: {date.today().isoformat()}",
    ]
    if campaign_type == "pre_approved":
        lines.extend(
            [
                f"- Pre-Approved Amount: {_as_money(customer.get('advance_amount_usd'))}",
                f"- Max Term Months: {customer.get('max_term_months', 12)}",
                f"- VDC Total (12 months): {_as_money(customer.get('vdc_total_usd'))}",
                (
                    "- Estimated Monthly Payment (principal + VDC + interest): "
                    f"{_as_money(customer.get('estimated_monthly_payment_usd'))}"
                ),
                (
                    "- Estimated Total Interest (12 months): "
                    f"{_as_money(customer.get('estimated_total_interest_usd'))}"
                ),
            ]
        )

    return "\n".join(lines)


class SalesAgent(Agent):
    def __init__(self, instructions: str, disposition: CallDispositionState) -> None:
        super().__init__(instructions=instructions)
        self._disposition = disposition

    @function_tool
    async def mark_interest_outcome(
        self,
        context: RunContext,
        reached_customer: bool,
        interested: bool | None = None,
        reason: str = "",
        note: str = "",
    ) -> str:
        """Record whether customer was reached and whether they are interested."""
        self._disposition.customer_reached = reached_customer
        self._disposition.customer_interested = interested
        if interested is False and reason:
            self._disposition.not_interested_reason = reason
        if note:
            self._append_note(note)
        return "Interest outcome recorded."

    @function_tool
    async def mark_requested_loan_amount(
        self,
        context: RunContext,
        requested_loan_amount_usd: float,
        note: str = "",
    ) -> str:
        """Record requested loan amount."""
        self._disposition.requested_loan_amount_usd = requested_loan_amount_usd
        if note:
            self._append_note(note)
        return "Requested loan amount recorded."

    @function_tool
    async def mark_loan_officer_referral(
        self,
        context: RunContext,
        reason: str = "",
    ) -> str:
        """Record that customer should be referred to a loan officer."""
        self._disposition.referred_to_loan_officer = True
        if reason:
            self._append_note(reason)
        return "Loan officer referral recorded."

    @function_tool
    async def mark_qualification_notes(
        self,
        context: RunContext,
        qualification_notes: str,
    ) -> str:
        """Record qualification answers as concise notes."""
        self._disposition.qualification_notes = qualification_notes.strip()
        return "Qualification notes recorded."

    @function_tool
    async def mark_next_step(
        self,
        context: RunContext,
        next_step: str,
        next_step_datetime: str,
        best_phone_number: str,
        note: str = "",
    ) -> str:
        """Record agreed next step, date/time, and best phone number."""
        self._disposition.next_step = next_step.strip().lower()
        self._disposition.next_step_datetime = next_step_datetime
        self._disposition.best_phone_number = best_phone_number
        if note:
            self._append_note(note)
        return "Next-step details recorded."

    def _append_note(self, note: str) -> None:
        clean_note = note.strip()
        if not clean_note:
            return
        if self._disposition.notes:
            self._disposition.notes += f" | {clean_note}"
        else:
            self._disposition.notes = clean_note


def build_agent() -> tuple[SalesAgent, CallDispositionState]:
    campaign_type = str(CUSTOMER_PROFILE.get("campaign_type", "pre_approved")).lower()
    script_version = str(CUSTOMER_PROFILE.get("script_version", "A")).upper()
    if campaign_type != "pre_approved":
        script_version = "N/A"

    customer_context = build_customer_context(CUSTOMER_PROFILE)
    prompt = build_system_prompt(
        customer_context=customer_context,
        campaign_type=campaign_type,
        script_version=script_version,
        customer_name=CUSTOMER_PROFILE["customer_name"],
    )
    disposition = CallDispositionState()
    return SalesAgent(instructions=prompt, disposition=disposition), disposition


async def entrypoint(ctx: JobContext) -> None:
    agent, disposition = build_agent()
    await ctx.connect()

    session = AgentSession(
        stt=assemblyai.STT(model="universal-streaming-multilingual"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(model="sonic-3", language="en"),
        vad=silero.VAD.load(),
    )

    customer_name = str(CUSTOMER_PROFILE.get("customer_name", "there"))
    await session.start(room=ctx.room, agent=agent)
    await session.generate_reply(
        instructions=(
            "Start the call now with this line: "
            f"\"Hello, this is Abby! I'm a virtual representative calling from "
            f"Simple Loans. May I please speak with {customer_name}?\" "
            "Then follow the required sales flow naturally."
        )
    )

    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        print("=== CALL DISPOSITION SUMMARY ===")
        print(json.dumps(asdict(disposition), indent=2))


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
