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

from prompting import build_system_prompt, get_delinquency_policy

load_dotenv()


CUSTOMER_PROFILE: dict[str, Any] = {
    "customer_name": "Jordan Alvarez",
    "vehicle_year": 2021,
    "vehicle_make": "Toyota",
    "vehicle_model": "Camry",
    "loan_origination_date": "2024-01-15",
    "original_payment_due_date": "2026-02-01",
    "amount_outstanding_usd": 486.32,
    "days_past_due": 33,
    "loan_reference_id": "VX-20491",
    "phone_number": "+1-415-555-0199",
    "contact_channel_preference": "phone",
    "language_preference": "English",
}


@dataclass
class CallDispositionState:
    full_payment_committed: bool = False
    partial_payment_committed: bool = False
    partial_payment_amount_usd: float | None = None
    remaining_balance_commitment_date: str | None = None
    payment_plan_proposed: bool = False
    payment_plan_terms: str | None = None
    human_agent_handoff_requested: bool = False
    notes: str = ""


def build_customer_context(customer: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"- Customer Name: {customer['customer_name']}",
            (
                "- Vehicle: "
                f"{customer['vehicle_year']} {customer['vehicle_make']} "
                f"{customer['vehicle_model']}"
            ),
            f"- Loan Origination Date: {customer['loan_origination_date']}",
            f"- Original Payment Due Date: {customer['original_payment_due_date']}",
            f"- Amount Outstanding: ${customer['amount_outstanding_usd']:.2f}",
            f"- Days Past Due: {customer['days_past_due']}",
            f"- Loan ID: {customer.get('loan_reference_id', 'N/A')}",
            f"- Phone Number: {customer['phone_number']}",
            f"- Contact Preference: {customer['contact_channel_preference']}",
            f"- Language Preference: {customer['language_preference']}",
            f"- Today's Date: {date.today().isoformat()}",
        ]
    )


class CollectionsAgent(Agent):
    def __init__(self, instructions: str, disposition: CallDispositionState) -> None:
        super().__init__(instructions=instructions)
        self._disposition = disposition

    @function_tool
    async def mark_full_payment_committed(
        self,
        context: RunContext,
        note: str = "",
    ) -> str:
        """Record that customer commits to full payment now."""
        self._disposition.full_payment_committed = True
        self._disposition.partial_payment_committed = False
        self._disposition.payment_plan_proposed = False
        if note:
            self._disposition.notes = note
        return "Full-payment commitment recorded."

    @function_tool
    async def mark_partial_payment_committed(
        self,
        context: RunContext,
        partial_payment_amount_usd: float,
        remaining_balance_commitment_date: str,
        note: str = "",
    ) -> str:
        """Record partial payment and date commitment for remaining balance."""
        self._disposition.partial_payment_committed = True
        self._disposition.partial_payment_amount_usd = partial_payment_amount_usd
        self._disposition.remaining_balance_commitment_date = (
            remaining_balance_commitment_date
        )
        if note:
            self._disposition.notes = note
        return "Partial-payment commitment recorded."

    @function_tool
    async def mark_payment_plan_proposed(
        self,
        context: RunContext,
        payment_plan_terms: str,
        note: str = "",
    ) -> str:
        """Record proposed payment plan terms accepted by the customer."""
        self._disposition.payment_plan_proposed = True
        self._disposition.payment_plan_terms = payment_plan_terms
        if note:
            self._disposition.notes = note
        return "Payment-plan terms recorded."

    @function_tool
    async def mark_human_handoff_requested(
        self,
        context: RunContext,
        reason: str = "",
    ) -> str:
        """Record customer agreement to continue with a human specialist."""
        self._disposition.human_agent_handoff_requested = True
        if reason:
            self._disposition.notes = reason
        return "Human-handoff request recorded."


def build_agent() -> tuple[CollectionsAgent, CallDispositionState]:
    policy = get_delinquency_policy(int(CUSTOMER_PROFILE["days_past_due"]))
    customer_context = build_customer_context(CUSTOMER_PROFILE)
    prompt = build_system_prompt(
        customer_context=customer_context,
        consequence_message=policy.consequence_message,
        customer_name=CUSTOMER_PROFILE["customer_name"],
    )
    disposition = CallDispositionState()
    return CollectionsAgent(instructions=prompt, disposition=disposition), disposition


async def entrypoint(ctx: JobContext) -> None:
    agent, disposition = build_agent()
    await ctx.connect()

    session = AgentSession(
        stt=assemblyai.STT(model="universal-streaming-multilingual"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(model="sonic-3", language="en"),
        vad=silero.VAD.load(),
    )

    await session.start(room=ctx.room, agent=agent)
    await session.generate_reply(
        instructions=(
            "Start the call now by verifying whether you are speaking with the customer. "
            "Do not introduce yourself or share any account details until identity is confirmed. "
            "Then follow the required collections flow."
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
