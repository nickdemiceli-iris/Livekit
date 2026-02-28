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

from prompting import CAMPAIGN_PRE_APPROVED, build_system_prompt, resolve_campaign_policy

load_dotenv()


LEAD_PROFILE: dict[str, Any] = {
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
class SalesCallDispositionState:
    intended_customer_reached: bool = False
    callback_requested_with_non_customer: bool = False
    callback_time_for_customer: str | None = None
    customer_interested: bool | None = None
    not_interested_reason: str | None = None
    requested_loan_amount_usd: float | None = None
    requested_more_than_preapproved: bool = False
    referred_to_loan_officer: bool = False
    still_driving_vehicle: bool | None = None
    owns_vehicle_free_and_clear: bool | None = None
    current_lender: str | None = None
    payoff_amount_usd: float | None = None
    florida_resident: bool | None = None
    vin_available: bool | None = None
    mileage_available: bool | None = None
    inspection_scheduled: bool = False
    inspection_datetime: str | None = None
    human_callback_scheduled: bool = False
    human_callback_datetime: str | None = None
    confirmed_best_phone_number: str | None = None
    notes: str = ""


def _as_money(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"${value:.2f}"
    return "N/A"


def build_lead_context(
    lead: dict[str, Any],
    campaign_type: str,
    script_version: str | None,
) -> str:
    lines = [
        f"- Customer Name: {lead['customer_name']}",
        (
            "- Vehicle: "
            f"{lead.get('vehicle_year', 'Unknown')} "
            f"{lead.get('vehicle_make', 'Unknown')} "
            f"{lead.get('vehicle_model', 'Unknown')}"
        ),
        f"- State: {lead.get('state', 'Unknown')}",
        f"- Campaign Type: {campaign_type}",
        f"- Script Version: {script_version or 'N/A'}",
        f"- Lead Source: {lead.get('lead_source', 'N/A')}",
        f"- Preferred Phone: {lead.get('phone_number', 'N/A')}",
        f"- Contact Preference: {lead.get('contact_channel_preference', 'phone')}",
        f"- Language Preference: {lead.get('language_preference', 'English')}",
        f"- Today's Date: {date.today().isoformat()}",
    ]
    if campaign_type == CAMPAIGN_PRE_APPROVED:
        lines.extend(
            [
                f"- Pre-Approved Amount: {_as_money(lead.get('advance_amount_usd'))}",
                f"- Max Term Months: {lead.get('max_term_months', 12)}",
                f"- VDC Total (12 months): {_as_money(lead.get('vdc_total_usd'))}",
                (
                    "- Estimated Monthly Payment (principal + VDC + interest): "
                    f"{_as_money(lead.get('estimated_monthly_payment_usd'))}"
                ),
                (
                    "- Estimated Total Interest (12 months): "
                    f"{_as_money(lead.get('estimated_total_interest_usd'))}"
                ),
            ]
        )
    return "\n".join(lines)


class SalesAgent(Agent):
    def __init__(
        self, instructions: str, disposition: SalesCallDispositionState
    ) -> None:
        super().__init__(instructions=instructions)
        self._disposition = disposition

    @function_tool
    async def mark_intended_customer_reached(
        self,
        context: RunContext,
        reached: bool = True,
        note: str = "",
    ) -> str:
        """Record whether the intended customer was reached."""
        self._disposition.intended_customer_reached = reached
        if note:
            self._append_note(note)
        return "Intended customer reach status recorded."

    @function_tool
    async def mark_non_customer_callback(
        self,
        context: RunContext,
        callback_time: str,
        note: str = "",
    ) -> str:
        """Record callback time provided by a non-customer."""
        self._disposition.callback_requested_with_non_customer = True
        self._disposition.callback_time_for_customer = callback_time
        if note:
            self._append_note(note)
        return "Non-customer callback details recorded."

    @function_tool
    async def mark_interest_outcome(
        self,
        context: RunContext,
        interested: bool,
        reason: str = "",
    ) -> str:
        """Record customer interest outcome and reason when not interested."""
        self._disposition.customer_interested = interested
        if not interested and reason:
            self._disposition.not_interested_reason = reason
        elif reason:
            self._append_note(reason)
        return "Interest outcome recorded."

    @function_tool
    async def mark_requested_amount(
        self,
        context: RunContext,
        requested_loan_amount_usd: float,
        exceeds_preapproved: bool = False,
    ) -> str:
        """Record requested loan amount and whether it exceeds pre-approved amount."""
        self._disposition.requested_loan_amount_usd = requested_loan_amount_usd
        self._disposition.requested_more_than_preapproved = exceeds_preapproved
        return "Requested amount recorded."

    @function_tool
    async def mark_qualification_answers(
        self,
        context: RunContext,
        still_driving_vehicle: bool,
        owns_vehicle_free_and_clear: bool,
        florida_resident: bool,
        current_lender: str = "",
        payoff_amount_usd: float | None = None,
    ) -> str:
        """Record qualification responses and payoff details when applicable."""
        self._disposition.still_driving_vehicle = still_driving_vehicle
        self._disposition.owns_vehicle_free_and_clear = owns_vehicle_free_and_clear
        self._disposition.florida_resident = florida_resident
        if not owns_vehicle_free_and_clear:
            self._disposition.current_lender = current_lender or None
            self._disposition.payoff_amount_usd = payoff_amount_usd
        else:
            self._disposition.current_lender = None
            self._disposition.payoff_amount_usd = None
        return "Qualification answers recorded."

    @function_tool
    async def mark_vehicle_data_readiness(
        self,
        context: RunContext,
        vin_available: bool,
        mileage_available: bool,
        note: str = "",
    ) -> str:
        """Record whether VIN and mileage are available."""
        self._disposition.vin_available = vin_available
        self._disposition.mileage_available = mileage_available
        if note:
            self._append_note(note)
        return "VIN and mileage readiness recorded."

    @function_tool
    async def mark_referral_to_loan_officer(
        self,
        context: RunContext,
        reason: str = "",
    ) -> str:
        """Record that the lead should be referred to a loan officer."""
        self._disposition.referred_to_loan_officer = True
        if reason:
            self._append_note(reason)
        return "Loan officer referral recorded."

    @function_tool
    async def mark_inspection_schedule(
        self,
        context: RunContext,
        inspection_datetime: str,
        best_phone_number: str,
        note: str = "",
    ) -> str:
        """Record scheduled inspection time and confirmed phone number."""
        self._disposition.inspection_scheduled = True
        self._disposition.inspection_datetime = inspection_datetime
        self._disposition.confirmed_best_phone_number = best_phone_number
        if note:
            self._append_note(note)
        return "Inspection schedule recorded."

    @function_tool
    async def mark_human_callback_schedule(
        self,
        context: RunContext,
        callback_datetime: str,
        best_phone_number: str,
        note: str = "",
    ) -> str:
        """Record scheduled callback by a human representative."""
        self._disposition.human_callback_scheduled = True
        self._disposition.human_callback_datetime = callback_datetime
        self._disposition.confirmed_best_phone_number = best_phone_number
        if note:
            self._append_note(note)
        return "Human callback schedule recorded."

    @function_tool
    async def add_call_note(
        self,
        context: RunContext,
        note: str,
    ) -> str:
        """Add a freeform note to call disposition."""
        self._append_note(note)
        return "Call note added."

    def _append_note(self, note: str) -> None:
        clean_note = note.strip()
        if not clean_note:
            return
        if self._disposition.notes:
            self._disposition.notes += f" | {clean_note}"
        else:
            self._disposition.notes = clean_note


def build_agent() -> tuple[SalesAgent, SalesCallDispositionState]:
    script_version_value = LEAD_PROFILE.get("script_version")
    policy = resolve_campaign_policy(
        campaign_type=str(LEAD_PROFILE.get("campaign_type", "cold")),
        script_version=(
            str(script_version_value) if script_version_value is not None else None
        ),
    )
    lead_context = build_lead_context(
        LEAD_PROFILE,
        campaign_type=policy.campaign_type,
        script_version=policy.script_version,
    )
    customer_name = str(LEAD_PROFILE.get("customer_name", "there"))
    prompt = build_system_prompt(
        lead_context=lead_context,
        customer_name=customer_name,
        campaign_type=policy.campaign_type,
        script_version=policy.script_version,
        terms_review_mode=policy.terms_review_mode,
    )
    disposition = SalesCallDispositionState()
    return SalesAgent(instructions=prompt, disposition=disposition), disposition


async def entrypoint(ctx: JobContext) -> None:
    agent, disposition = build_agent()
    customer_name = str(LEAD_PROFILE.get("customer_name", "there"))

    await ctx.connect()

    session = AgentSession(
        # Streaming STT + Silero VAD is a reliable baseline for telephony calls.
        stt=assemblyai.STT(model="universal-streaming-multilingual"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(model="sonic-3", language="en"),
        vad=silero.VAD.load(),
    )

    await session.start(room=ctx.room, agent=agent)
    await session.generate_reply(
        instructions=(
            "Start the outbound call now with this exact first line: "
            f"\"Hello, this is Abby! I'm a virtual representative calling from "
            f"Simple Loans. May I please speak with {customer_name}?\" "
            "After that opening line, continue naturally using the configured sales flow. "
            "Keep responses concise, ask one question at a time, and use the tools "
            "to capture call disposition details."
        )
    )

    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        print("=== SALES CALL DISPOSITION SUMMARY ===")
        print(json.dumps(asdict(disposition), indent=2))


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
