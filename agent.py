from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Any

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    RunContext,
    function_tool,
    inference,
)
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()


# Set CALL_FLOW in env to one of:
# - preapproved_a (AI agent reviews terms)
# - preapproved_b (human agent reviews terms on callback)
# - cold_client   (not pre-approved)
CALL_FLOW_PREAPPROVED_A = "preapproved_a"
CALL_FLOW_PREAPPROVED_B = "preapproved_b"
CALL_FLOW_COLD_CLIENT = "cold_client"
VALID_CALL_FLOWS = {
    CALL_FLOW_PREAPPROVED_A,
    CALL_FLOW_PREAPPROVED_B,
    CALL_FLOW_COLD_CLIENT,
}
CALL_FLOW = os.getenv("CALL_FLOW", CALL_FLOW_PREAPPROVED_A).strip().lower()
if CALL_FLOW not in VALID_CALL_FLOWS:
    CALL_FLOW = CALL_FLOW_PREAPPROVED_A


# Lead profile: replace this with CRM/API payload in production.
LEAD_PROFILE: dict[str, Any] = {
    "client_name": "Jordan Alvarez",
    "phone_number": "+1-415-555-0199",
    "vehicle_year": 2021,
    "vehicle_make": "Toyota",
    "vehicle_model": "Camry",
    "state": "Florida",
    "preapproved_amount_usd": 2600.00,
    "vdc_total_usd": 315.00,
    "monthly_payment_usd": 284.75,
    "total_interest_usd": 426.00,
}


@dataclass
class SalesCallState:
    script_version: str = ""
    reached_intended_client: bool = False
    interested_in_loan: bool | None = None
    decline_reason: str = ""
    requested_amount_usd: float | None = None
    requested_more_than_preapproved: bool = False
    referred_to_loan_officer: bool = False
    still_driving_vehicle: bool | None = None
    owns_free_and_clear: bool | None = None
    lender_name: str = ""
    lender_balance_usd: float | None = None
    florida_resident: bool | None = None
    vin_available: bool | None = None
    mileage_available: bool | None = None
    next_step_type: str = ""
    next_step_day: str = ""
    next_step_time: str = ""
    best_phone_number: str = ""
    outcome: str = "in_progress"
    notes: str = ""


def _format_currency(amount: float | None) -> str:
    if amount is None:
        return "N/A"
    return f"${amount:,.2f}"


def _vehicle_label(lead: dict[str, Any]) -> str:
    year = lead.get("vehicle_year", "unknown year")
    make = lead.get("vehicle_make", "unknown make")
    model = lead.get("vehicle_model", "unknown model")
    return f"{year} {make} {model}"


def _call_flow_label(call_flow: str) -> str:
    if call_flow == CALL_FLOW_PREAPPROVED_A:
        return "Pre-Approved Script A (AI terms review)"
    if call_flow == CALL_FLOW_PREAPPROVED_B:
        return "Pre-Approved Script B (Human terms review)"
    return "Cold Client Script"


def _append_note(state: SalesCallState, note: str) -> None:
    cleaned = note.strip()
    if not cleaned:
        return
    if state.notes:
        state.notes = f"{state.notes} | {cleaned}"
    else:
        state.notes = cleaned


def build_lead_context(lead: dict[str, Any], call_flow: str) -> str:
    rows = [
        f"- Script Version: {_call_flow_label(call_flow)}",
        f"- Client Name: {lead.get('client_name', 'Unknown')}",
        f"- Vehicle: {_vehicle_label(lead)}",
        f"- State on File: {lead.get('state', 'Unknown')}",
        f"- Best Phone on File: {lead.get('phone_number', 'Unknown')}",
        f"- Today's Date: {date.today().isoformat()}",
    ]
    if call_flow in {CALL_FLOW_PREAPPROVED_A, CALL_FLOW_PREAPPROVED_B}:
        rows.extend(
            [
                f"- Pre-Approved Amount: {_format_currency(lead.get('preapproved_amount_usd'))}",
                f"- VDC Total (12 months): {_format_currency(lead.get('vdc_total_usd'))}",
                f"- Monthly Payment Estimate: {_format_currency(lead.get('monthly_payment_usd'))}",
                f"- Total Interest (12 months): {_format_currency(lead.get('total_interest_usd'))}",
            ]
        )
    return "\n".join(rows)


def build_system_prompt(call_flow: str, lead: dict[str, Any]) -> str:
    client_name = str(lead.get("client_name", "the client"))
    vehicle = _vehicle_label(lead)
    preapproved_amount = _format_currency(lead.get("preapproved_amount_usd"))
    vdc_total = _format_currency(lead.get("vdc_total_usd"))
    monthly_payment = _format_currency(lead.get("monthly_payment_usd"))
    total_interest = _format_currency(lead.get("total_interest_usd"))

    universal = f"""
You are Abby, a virtual representative from Simple Loans.

Primary objective:
- Run a professional outbound sales call for a title loan.
- Follow the required business flow in order.
- Keep the call natural, concise, and respectful.
- Capture key outcomes with tools.

Natural voice requirements (high priority):
- Sound human, warm, and confident. Never sound robotic.
- Use conversational phrasing and contractions naturally.
- Ask one clear question at a time, then wait for a response.
- Keep most turns to one or two short sentences.
- Vary sentence openings; do not repeat canned filler lines.
- If interrupted, acknowledge briefly and continue smoothly.
- Do not use markdown, lists, or formatting in spoken responses.

Universal flow rules:
1) Start by asking: "Hello, this is Abby! I'm a virtual representative calling from Simple Loans. May I please speak with {client_name}?"
2) If the person is not {client_name}, ask if {client_name} is available.
3) If unavailable, ask for the best callback time, call mark_wrong_party_callback, and close politely.
4) Do not continue loan details with a non-client.
5) Confirm best phone number before close whenever possible.
6) End with a clear recap of the next step day/time.

Tool usage requirements:
- Use mark_intended_client_reached once identity is confirmed.
- Use record_interest after the client answers interest.
- If not interested, ask why and call record_decline_reason.
- Use record_requested_amount when customer gives desired amount.
- If requested amount is above approved cap, call mark_loan_officer_referral.
- Use record_qualification_answers after qualification responses are collected.
- In cold flow, use record_vin_mileage_availability after asking VIN and mileage readiness.
- Use schedule_next_step for inspection or human callback scheduling.
- Use confirm_best_phone_number once number is confirmed.
- Use finalize_call before ending.

Lead context:
{build_lead_context(lead, call_flow)}
""".strip()

    if call_flow == CALL_FLOW_PREAPPROVED_A:
        campaign = f"""
Campaign: Pre-Approved Client Script Version A (AI reviews terms now).

Required sequence:
1) After identity confirmation, state reason:
   "Hi, {client_name}. I'm calling to let you know that you have been pre-approved for a {preapproved_amount} title loan with us. Are you interested in hearing the details of your loan?"
2) If not interested:
   - Ask: "May I ask why you are not interested in the pre-qualified loan? We will want to document the reason."
   - Record reason with record_decline_reason and close politely.
3) If client asks for more than {preapproved_amount}:
   - Ask how much they need.
   - Explain that you will refer this to a loan officer.
   - Call mark_loan_officer_referral.
4) Ask qualifying questions one at a time:
   - Are you still driving the {vehicle}?
   - Do you own your car free and clear?
     If no, ask lender name and balance owed.
   - Do you live in Florida?
     If no, explain lending is currently only for Florida residents and close politely.
   - Ask how much they want to borrow (up to approved amount).
5) Review terms naturally in short chunks:
   - 12-month term, no prepayment penalty.
   - VDC explanation:
     "Our system shows you may not have six months of prepaid comprehensive and collision insurance with a one-thousand-dollar deductible. We offer Voluntary Debt Cancellation, called VDC, which forgives the loan if your car is stolen or totaled and allows us to provide a loan without that prepaid insurance requirement."
   - Total VDC for 12 months: {vdc_total}
   - Monthly payment including principal, VDC, and interest: {monthly_payment}
   - Total 12-month interest: {total_interest}; interest only applies for months they keep the loan because no prepayment penalty.
6) Schedule inspection:
   "The next step in getting your loan funded is for one of our agents to conduct a virtual inspection of the vehicle. Are you available today or tomorrow?"
7) Confirm best phone number.
8) Recap with exact callback date/time and a professional close.
""".strip()
    elif call_flow == CALL_FLOW_PREAPPROVED_B:
        campaign = f"""
Campaign: Pre-Approved Client Script Version B (human agent reviews terms on callback).

Required sequence:
1) After identity confirmation, state reason:
   "Hi, {client_name}. I'm calling to let you know that you have been pre-approved for a {preapproved_amount} title loan with us. Are you interested in hearing the details of your loan?"
2) If not interested:
   - Ask: "May I ask why you are not interested in the pre-qualified loan? We will want to document the reason."
   - Record reason with record_decline_reason and close politely.
3) If client asks for more than {preapproved_amount}:
   - Ask how much they need.
   - Explain that you will refer this to a loan officer.
   - Call mark_loan_officer_referral.
4) Ask qualifying questions one at a time:
   - Are you still driving the {vehicle}?
   - Do you own your car free and clear?
     If no, ask lender name and balance owed.
   - Do you live in Florida?
     If no, explain lending is currently only for Florida residents and close politely.
   - Ask how much they want to borrow (up to approved amount).
5) Schedule inspection with human terms review:
   "The next step in getting your loan funded is for one of our agents to conduct a virtual inspection of the vehicle and review the loan terms with you. Are you available today or tomorrow?"
6) Confirm best phone number.
7) Recap with exact callback date/time and a professional close.
""".strip()
    else:
        campaign = f"""
Campaign: Cold Client (not pre-approved).

Required sequence:
1) Greet and state reason:
   - The client expressed interest in a title loan.
2) Ask if they are interested in moving forward.
3) Ask how much they are looking to borrow and record it.
4) Ask qualifying questions one at a time:
   - Confirm year, make, and model of their car (reference expected: {vehicle}).
   - Do they own the car free and clear? If no, ask lender and payoff balance.
   - Do they live in Florida? If no, explain lending is currently Florida-only and close politely.
5) Ask if they have mileage and VIN readily available.
   - Record with record_vin_mileage_availability.
6) Confirm best time for a human representative callback to finalize terms.
7) Confirm best phone number and give concise recap before close.
""".strip()

    return f"{universal}\n\n{campaign}".strip()


class AbbySalesAgent(Agent):
    def __init__(self, instructions: str, call_flow: str, client_name: str) -> None:
        super().__init__(instructions=instructions)
        self._call_flow = call_flow
        self._client_name = client_name

    async def on_enter(self) -> None:
        if self._call_flow == CALL_FLOW_COLD_CLIENT:
            start_instructions = (
                f"Start the call now with the approved greeting and ask for {self._client_name}. "
                "After identity is confirmed, explain they previously showed interest in a title loan "
                "and ask if they want to move forward."
            )
        else:
            start_instructions = (
                f"Start the call now with the approved greeting and ask for {self._client_name}. "
                "Once identity is confirmed, deliver the pre-approved call opening naturally."
            )
        await self.session.generate_reply(instructions=start_instructions)

    @function_tool()
    async def mark_intended_client_reached(
        self, context: RunContext[SalesCallState]
    ) -> str:
        """Call when the intended customer identity is confirmed on this call."""
        context.userdata.reached_intended_client = True
        return "Intended client marked as reached."

    @function_tool()
    async def mark_wrong_party_callback(
        self, context: RunContext[SalesCallState], best_time_to_call_back: str, note: str = ""
    ) -> str:
        """Call when customer is unavailable and callback timing is provided by another person."""
        context.userdata.reached_intended_client = False
        context.userdata.next_step_type = "wrong_party_callback"
        context.userdata.next_step_day = best_time_to_call_back
        _append_note(context.userdata, note)
        return "Wrong-party callback timing recorded."

    @function_tool()
    async def record_interest(
        self, context: RunContext[SalesCallState], interested: bool, note: str = ""
    ) -> str:
        """Record whether customer is interested in continuing loan conversation."""
        context.userdata.interested_in_loan = interested
        _append_note(context.userdata, note)
        return "Interest status recorded."

    @function_tool()
    async def record_decline_reason(
        self, context: RunContext[SalesCallState], reason: str
    ) -> str:
        """Record reason the customer is not interested in moving forward."""
        context.userdata.interested_in_loan = False
        context.userdata.decline_reason = reason.strip()
        return "Decline reason recorded."

    @function_tool()
    async def record_requested_amount(
        self, context: RunContext[SalesCallState], amount_usd: float
    ) -> str:
        """Record the customer's requested borrowing amount in USD."""
        context.userdata.requested_amount_usd = amount_usd
        return "Requested amount recorded."

    @function_tool()
    async def mark_loan_officer_referral(
        self,
        context: RunContext[SalesCallState],
        requested_amount_usd: float,
        reason: str = "",
    ) -> str:
        """Record that customer requested above approved amount and needs loan officer review."""
        context.userdata.requested_more_than_preapproved = True
        context.userdata.referred_to_loan_officer = True
        context.userdata.requested_amount_usd = requested_amount_usd
        _append_note(context.userdata, reason)
        return "Loan officer referral recorded."

    @function_tool()
    async def record_qualification_answers(
        self,
        context: RunContext[SalesCallState],
        still_driving_vehicle: bool,
        owns_free_and_clear: bool,
        florida_resident: bool,
        lender_name: str = "",
        lender_balance_usd: float | None = None,
    ) -> str:
        """Record core qualification answers and lender details if vehicle has a lien."""
        context.userdata.still_driving_vehicle = still_driving_vehicle
        context.userdata.owns_free_and_clear = owns_free_and_clear
        context.userdata.florida_resident = florida_resident
        if owns_free_and_clear:
            context.userdata.lender_name = ""
            context.userdata.lender_balance_usd = None
        else:
            context.userdata.lender_name = lender_name.strip()
            context.userdata.lender_balance_usd = lender_balance_usd
        return "Qualification answers recorded."

    @function_tool()
    async def record_vin_mileage_availability(
        self,
        context: RunContext[SalesCallState],
        vin_available: bool,
        mileage_available: bool,
    ) -> str:
        """Record whether VIN and mileage are readily available for the cold-client flow."""
        context.userdata.vin_available = vin_available
        context.userdata.mileage_available = mileage_available
        return "VIN and mileage availability recorded."

    @function_tool()
    async def schedule_next_step(
        self,
        context: RunContext[SalesCallState],
        step_type: str,
        day: str,
        time: str,
        note: str = "",
    ) -> str:
        """Record scheduled next step such as inspection or human callback."""
        context.userdata.next_step_type = step_type.strip().lower()
        context.userdata.next_step_day = day.strip()
        context.userdata.next_step_time = time.strip()
        _append_note(context.userdata, note)
        return "Next step scheduled."

    @function_tool()
    async def confirm_best_phone_number(
        self, context: RunContext[SalesCallState], phone_number: str
    ) -> str:
        """Record confirmed best phone number for follow-up."""
        context.userdata.best_phone_number = phone_number.strip()
        return "Best phone number confirmed."

    @function_tool()
    async def finalize_call(
        self, context: RunContext[SalesCallState], outcome: str, note: str = ""
    ) -> str:
        """
        Mark final call outcome.
        Example outcomes: scheduled_inspection, human_callback_scheduled, not_interested,
        ineligible_non_florida, wrong_party_callback, referred_to_loan_officer.
        """
        context.userdata.outcome = outcome.strip().lower()
        _append_note(context.userdata, note)
        return "Call finalized."


server = AgentServer()


@server.rtc_session()
async def sales_session(ctx: agents.JobContext) -> None:
    script_version = _call_flow_label(CALL_FLOW)
    session = AgentSession[SalesCallState](
        userdata=SalesCallState(script_version=script_version),
        stt="assemblyai/universal-streaming-multilingual",
        llm="openai/gpt-4o-mini",
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="f9836c6e-a0bd-460e-9d3c-f7299fa60f94",
            language="en",
            extra_kwargs={
                "speed": 1.0,
                "volume": 1.0,
            },
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        min_endpointing_delay=0.25,
        max_endpointing_delay=1.8,
        preemptive_generation=True,
    )

    prompt = build_system_prompt(CALL_FLOW, LEAD_PROFILE)
    agent = AbbySalesAgent(
        instructions=prompt,
        call_flow=CALL_FLOW,
        client_name=str(LEAD_PROFILE.get("client_name", "the client")),
    )
    await session.start(room=ctx.room, agent=agent)


if __name__ == "__main__":
    agents.cli.run_app(server)
