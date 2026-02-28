from __future__ import annotations

import asyncio
import json
import os
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

try:
    # Optional but useful for snappier conversational turns on phone calls.
    from livekit.plugins.turn_detector.multilingual import MultilingualModel
except Exception:  # pragma: no cover - safe fallback when plugin is unavailable
    MultilingualModel = None

load_dotenv()


CAMPAIGN_PREAPPROVED_AI_TERMS = "preapproved_ai_terms"
CAMPAIGN_PREAPPROVED_HUMAN_TERMS = "preapproved_human_terms"
CAMPAIGN_COLD_CLIENT = "cold_client"
VALID_CAMPAIGNS = {
    CAMPAIGN_PREAPPROVED_AI_TERMS,
    CAMPAIGN_PREAPPROVED_HUMAN_TERMS,
    CAMPAIGN_COLD_CLIENT,
}


def _campaign_from_env() -> str:
    raw_mode = os.getenv("SALES_CAMPAIGN_MODE", CAMPAIGN_PREAPPROVED_AI_TERMS).strip()
    return raw_mode if raw_mode in VALID_CAMPAIGNS else CAMPAIGN_PREAPPROVED_AI_TERMS


# Example lead profile. In production, replace this with your CRM/lead API payload.
LEAD_PROFILE: dict[str, Any] = {
    "campaign_mode": _campaign_from_env(),
    "client_name": "Jordan Alvarez",
    "preapproved_amount_usd": 2500.0,
    "vehicle_year": 2021,
    "vehicle_make": "Toyota",
    "vehicle_model": "Camry",
    "state": "FL",
    "phone_number": "+1-415-555-0199",
    "vdc_total_usd": 288.0,
    "monthly_payment_usd": 243.92,
    "total_interest_usd": 139.04,
    "lead_source_note": "Client previously requested title loan information.",
}


@dataclass
class SalesDispositionState:
    right_party_confirmed: bool = False
    wrong_party_or_unavailable: bool = False
    callback_window: str | None = None
    interested_in_loan: bool | None = None
    uninterested_reason: str | None = None
    requested_amount_usd: float | None = None
    requested_above_preapproved_usd: float | None = None
    referred_to_loan_officer: bool = False
    still_driving_vehicle: bool | None = None
    owns_vehicle_free_and_clear: bool | None = None
    current_lender_name: str | None = None
    amount_owed_to_lender_usd: float | None = None
    florida_resident: bool | None = None
    vin_and_mileage_ready: bool | None = None
    follow_up_type: str | None = None
    follow_up_datetime: str | None = None
    best_phone_number: str | None = None
    human_handoff_requested: bool = False
    call_outcome: str | None = None
    notes: str = ""


def _fmt_money(value: float | None) -> str:
    return f"${value:,.2f}" if value is not None else "N/A"


def _build_lead_context(lead: dict[str, Any]) -> str:
    lines = [
        f"- Campaign Mode: {lead.get('campaign_mode', 'unknown')}",
        f"- Client Name: {lead.get('client_name', 'Unknown')}",
        (
            "- Vehicle: "
            f"{lead.get('vehicle_year', 'Unknown')} "
            f"{lead.get('vehicle_make', 'Unknown')} "
            f"{lead.get('vehicle_model', 'Unknown')}"
        ),
        f"- State on File: {lead.get('state', 'Unknown')}",
        f"- Phone Number: {lead.get('phone_number', 'Unknown')}",
        f"- Pre-Approved Amount: {_fmt_money(lead.get('preapproved_amount_usd'))}",
        f"- VDC Total (12 months): {_fmt_money(lead.get('vdc_total_usd'))}",
        f"- Estimated Monthly Payment: {_fmt_money(lead.get('monthly_payment_usd'))}",
        f"- Estimated Total Interest (12 months): {_fmt_money(lead.get('total_interest_usd'))}",
        f"- Lead Source Note: {lead.get('lead_source_note', 'N/A')}",
        f"- Today's Date: {date.today().isoformat()}",
    ]
    return "\n".join(lines)


def _build_campaign_flow(campaign_mode: str, lead: dict[str, Any]) -> str:
    preapproved_amount = _fmt_money(lead.get("preapproved_amount_usd"))
    vehicle = (
        f"{lead.get('vehicle_year', 'Unknown')} "
        f"{lead.get('vehicle_make', 'Unknown')} "
        f"{lead.get('vehicle_model', 'Unknown')}"
    )
    vdc_total = _fmt_money(lead.get("vdc_total_usd"))
    monthly_payment = _fmt_money(lead.get("monthly_payment_usd"))
    total_interest = _fmt_money(lead.get("total_interest_usd"))

    if campaign_mode == CAMPAIGN_PREAPPROVED_AI_TERMS:
        return f"""
Campaign script: Pre-Approved Client - Version A (AI reviews loan terms)
1) Opening:
   - Start with: "Hello, this is Abby, a virtual representative calling from Simple Loans. May I please speak with {lead.get('client_name')}?"
   - If a non-client answers, ask if the client is available now.
   - If unavailable, ask for the best callback time, call record_wrong_party_or_unavailable, then close politely.
2) Once the client is confirmed:
   - Reason for call: pre-approved title loan for {preapproved_amount}.
   - Ask if they want to hear loan details.
3) If not interested:
   - Ask for the reason so it can be documented.
   - Call record_not_interested and record_call_outcome.
4) If they want more than pre-approved:
   - Ask exact amount needed.
   - Inform them a loan officer will review that request.
   - Call record_requested_amount_above_preapproved.
5) Ask qualifying questions one at a time:
   - Are you still driving the {vehicle}?
   - Do you own your vehicle free and clear?
   - If not free and clear, ask lender name and how much is still owed.
   - Do you live in Florida? If not, explain we currently lend only to Florida residents.
   - Ask how much they want to borrow (up to {preapproved_amount}).
6) Summarize terms naturally (not as a robotic monologue):
   - Twelve-month term, no prepayment penalty.
   - VDC explanation:
     We offer Voluntary Debt Cancellation (VDC), which can forgive the loan if the vehicle is stolen or totaled
     and allows funding without six months of prepaid comprehensive/collision insurance with a one-thousand-dollar deductible.
     Total VDC cost for twelve months: {vdc_total}.
   - Estimated monthly payment including principal, VDC, and interest: {monthly_payment}.
   - Estimated total interest for twelve months: {total_interest}.
   - Clarify there is no prepayment penalty and interest is only paid for months the loan is active.
7) Schedule next step:
   - Ask if they are available today or tomorrow for a virtual vehicle inspection.
   - Confirm best phone number.
   - Give a concise recap with date/time.
   - Call schedule_follow_up with follow_up_type="virtual_inspection".
   - Call record_call_outcome with outcome="inspection_scheduled".
""".strip()

    if campaign_mode == CAMPAIGN_PREAPPROVED_HUMAN_TERMS:
        return f"""
Campaign script: Pre-Approved Client - Version B (human reviews terms on callback)
1) Opening:
   - Start with: "Hello, this is Abby, a virtual representative calling from Simple Loans. May I please speak with {lead.get('client_name')}?"
   - If non-client answers, ask if the client is available.
   - If unavailable, ask best callback time, call record_wrong_party_or_unavailable, and close politely.
2) Once client is confirmed:
   - Share reason for call: pre-approved title loan for {preapproved_amount}.
   - Ask if they want to hear details.
3) If not interested:
   - Ask reason for documentation.
   - Call record_not_interested and record_call_outcome.
4) If asking for more than pre-approved:
   - Ask requested amount and explain a loan officer will review.
   - Call record_requested_amount_above_preapproved.
5) Ask qualifying questions one at a time:
   - Still driving the listed vehicle.
   - Own free and clear, or collect lender and balance owed.
   - Florida residency.
   - Borrow amount request (up to pre-approved amount).
6) Schedule next step:
   - Explain that a human agent will review terms during the virtual inspection callback.
   - Ask if they are available today or tomorrow.
   - Confirm best phone number.
   - Recap appointment date/time.
   - Call schedule_follow_up with follow_up_type="virtual_inspection_human_terms".
   - Call record_call_outcome with outcome="inspection_scheduled_human_terms".
""".strip()

    return f"""
Campaign script: Cold Client (not pre-approved)
1) Opening:
   - Start with: "Hello, this is Abby, a virtual representative calling from Simple Loans. May I please speak with {lead.get('client_name')}?"
   - If non-client answers and client is unavailable, ask best callback time, call record_wrong_party_or_unavailable, then close politely.
2) Reason for call:
   - Explain they previously showed interest in a title loan.
   - Ask whether they are interested in moving forward.
3) If not interested:
   - Ask reason for documentation.
   - Call record_not_interested and record_call_outcome.
4) If interested:
   - Ask how much they are looking to borrow and call record_requested_amount.
   - Ask qualifying questions one at a time:
     a) Confirm year, make, model of vehicle.
     b) Ask if they own it free and clear. If no, capture lender and amount owed.
     c) Confirm Florida residency.
   - Ask if mileage and VIN are readily available; call record_vin_and_mileage_ready.
5) Set handoff:
   - Confirm best callback time for a human representative to finalize terms.
   - Confirm best phone number.
   - Call schedule_follow_up with follow_up_type="human_callback".
   - Call record_call_outcome with outcome="human_callback_scheduled".
""".strip()


def build_system_prompt(lead: dict[str, Any]) -> str:
    campaign_mode = lead.get("campaign_mode", CAMPAIGN_PREAPPROVED_AI_TERMS)
    client_name = lead.get("client_name", "the client")
    campaign_flow = _build_campaign_flow(campaign_mode, lead)
    lead_context = _build_lead_context(lead)

    return f"""
You are Abby, a virtual sales representative for Simple Loans, handling outbound title-loan calls.

Primary objective:
- Run the correct sales script for this lead.
- Sound natural, calm, and professional.
- Capture accurate qualification details and schedule the next step.

Conversation style:
- Warm, confident, and concise.
- Use natural spoken phrasing with contractions.
- Ask one clear question at a time.
- Keep turns short (usually one to three sentences).
- Vary acknowledgments so you do not sound repetitive or robotic.
- Do not read long blocks; break complex details into smaller pieces and check for understanding.

Compliance and safety:
- Verify you are speaking to "{client_name}" before discussing offer details.
- If a non-client answers, do not discuss loan details.
- If asked, be transparent that you are a virtual representative.
- Never pressure, threaten, or argue.
- If the person asks not to be called, acknowledge and close politely.
- Do not invent rates, fees, or terms outside provided context.
- If unsure, offer a human follow-up instead of guessing.

Required tool usage:
- After right-party confirmation, call mark_right_party_confirmed.
- If wrong party or client unavailable, call record_wrong_party_or_unavailable.
- If not interested, call record_not_interested.
- If interested and amount is provided, call record_requested_amount.
- If amount requested is above pre-approved, call record_requested_amount_above_preapproved.
- After qualification answers, call record_qualification_answers.
- For cold-client VIN/mileage availability, call record_vin_and_mileage_ready.
- For any scheduled inspection/callback, call schedule_follow_up.
- Set final disposition with record_call_outcome before ending.
- If customer requests human now, call record_human_handoff.

Campaign flow to follow:
{campaign_flow}

Lead context:
{lead_context}
""".strip()


class SalesAgent(Agent):
    def __init__(self, instructions: str, disposition: SalesDispositionState) -> None:
        super().__init__(instructions=instructions)
        self._disposition = disposition

    @function_tool
    async def mark_right_party_confirmed(
        self,
        context: RunContext,
        note: str = "",
    ) -> str:
        """Call when the intended client identity is confirmed."""
        self._disposition.right_party_confirmed = True
        if note:
            self._disposition.notes = note
        return "Right-party confirmation recorded."

    @function_tool
    async def record_wrong_party_or_unavailable(
        self,
        context: RunContext,
        callback_window: str,
        note: str = "",
    ) -> str:
        """Call when a non-client answers or the client is unavailable."""
        self._disposition.wrong_party_or_unavailable = True
        self._disposition.callback_window = callback_window
        self._disposition.call_outcome = "callback_needed"
        if note:
            self._disposition.notes = note
        return "Wrong-party/unavailable callback details recorded."

    @function_tool
    async def record_not_interested(
        self,
        context: RunContext,
        reason: str,
        note: str = "",
    ) -> str:
        """Call when the client declines moving forward with the loan."""
        self._disposition.interested_in_loan = False
        self._disposition.uninterested_reason = reason
        if note:
            self._disposition.notes = note
        return "Not-interested reason recorded."

    @function_tool
    async def record_requested_amount(
        self,
        context: RunContext,
        requested_amount_usd: float,
        note: str = "",
    ) -> str:
        """Call when client provides desired loan amount."""
        self._disposition.interested_in_loan = True
        self._disposition.requested_amount_usd = requested_amount_usd
        if note:
            self._disposition.notes = note
        return "Requested amount recorded."

    @function_tool
    async def record_requested_amount_above_preapproved(
        self,
        context: RunContext,
        requested_amount_usd: float,
        note: str = "",
    ) -> str:
        """Call when client asks for more than pre-approved amount."""
        self._disposition.interested_in_loan = True
        self._disposition.requested_above_preapproved_usd = requested_amount_usd
        self._disposition.referred_to_loan_officer = True
        if note:
            self._disposition.notes = note
        return "Above-preapproved request recorded for loan officer review."

    @function_tool
    async def record_qualification_answers(
        self,
        context: RunContext,
        still_driving_vehicle: bool,
        owns_vehicle_free_and_clear: bool,
        florida_resident: bool,
        current_lender_name: str = "",
        amount_owed_to_lender_usd: float | None = None,
        note: str = "",
    ) -> str:
        """Call after capturing qualification responses."""
        self._disposition.still_driving_vehicle = still_driving_vehicle
        self._disposition.owns_vehicle_free_and_clear = owns_vehicle_free_and_clear
        self._disposition.florida_resident = florida_resident
        self._disposition.current_lender_name = current_lender_name or None
        self._disposition.amount_owed_to_lender_usd = amount_owed_to_lender_usd
        if note:
            self._disposition.notes = note
        return "Qualification answers recorded."

    @function_tool
    async def record_vin_and_mileage_ready(
        self,
        context: RunContext,
        ready: bool,
        note: str = "",
    ) -> str:
        """Call in cold-client flow after asking if VIN and mileage are available."""
        self._disposition.vin_and_mileage_ready = ready
        if note:
            self._disposition.notes = note
        return "VIN/mileage readiness recorded."

    @function_tool
    async def schedule_follow_up(
        self,
        context: RunContext,
        follow_up_type: str,
        follow_up_datetime: str,
        best_phone_number: str,
        note: str = "",
    ) -> str:
        """Call when inspection or human callback is scheduled."""
        self._disposition.follow_up_type = follow_up_type.strip()
        self._disposition.follow_up_datetime = follow_up_datetime.strip()
        self._disposition.best_phone_number = best_phone_number.strip()
        if note:
            self._disposition.notes = note
        return "Follow-up schedule recorded."

    @function_tool
    async def record_human_handoff(
        self,
        context: RunContext,
        reason: str = "",
    ) -> str:
        """Call when the customer asks to continue with a human representative."""
        self._disposition.human_handoff_requested = True
        if reason:
            self._disposition.notes = reason
        return "Human handoff request recorded."

    @function_tool
    async def record_call_outcome(
        self,
        context: RunContext,
        outcome: str,
        note: str = "",
    ) -> str:
        """Call near the end to store the final call outcome."""
        self._disposition.call_outcome = outcome.strip()
        if note:
            self._disposition.notes = note
        return "Final call outcome recorded."


def build_agent() -> tuple[SalesAgent, SalesDispositionState]:
    prompt = build_system_prompt(LEAD_PROFILE)
    disposition = SalesDispositionState()
    return SalesAgent(instructions=prompt, disposition=disposition), disposition


def _build_start_instruction(lead: dict[str, Any]) -> str:
    return (
        f'Start the call now by saying: "Hello, this is Abby, a virtual representative '
        f'calling from Simple Loans. May I please speak with {lead.get("client_name", "the client")}?" '
        "Then continue naturally using the campaign flow and ask one clear question at a time."
    )


async def entrypoint(ctx: JobContext) -> None:
    agent, disposition = build_agent()
    await ctx.connect()

    session_kwargs: dict[str, Any] = {
        "stt": assemblyai.STT(model="universal-streaming-multilingual"),
        "llm": openai.LLM(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
        "tts": cartesia.TTS(model="sonic-3", language="en"),
        "vad": silero.VAD.load(),
    }

    if MultilingualModel is not None:
        session_kwargs.update(
            {
                "turn_detection": MultilingualModel(),
                "min_endpointing_delay": 0.5,
                "max_endpointing_delay": 1.8,
                "preemptive_generation": True,
            }
        )

    session = AgentSession(**session_kwargs)

    await session.start(room=ctx.room, agent=agent)
    await session.generate_reply(instructions=_build_start_instruction(LEAD_PROFILE))

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
