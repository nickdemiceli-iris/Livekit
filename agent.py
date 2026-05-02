from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from typing import Any
from urllib import request

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

from knowledgebase import GCSKnowledgebase, create_knowledgebase_from_env
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


@dataclass
class TranscriptTurn:
    timestamp_utc: str
    role: str
    text: str
    interrupted: bool = False


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


def _to_role_string(role: Any) -> str:
    value = getattr(role, "value", role)
    return str(value).lower()


def _extract_item_text(item: Any) -> str:
    text_content = getattr(item, "text_content", None)
    if text_content:
        return str(text_content).strip()

    content = getattr(item, "content", None)
    if not content:
        return ""

    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            if part.strip():
                parts.append(part.strip())
            continue
        transcript = getattr(part, "transcript", None)
        if transcript and str(transcript).strip():
            parts.append(str(transcript).strip())
    return " ".join(parts).strip()


def _is_user_role(role: str) -> bool:
    return role in {"user", "human", "customer"}


def _classify_objection(reason_or_text: str) -> str:
    text = reason_or_text.lower()
    if not text.strip():
        return "none"

    if any(k in text for k in ["fuck off", "stop calling", "leave me alone"]):
        return "hostile_rejection"
    if any(k in text for k in ["rate", "interest", "apr", "payment too high", "cost"]):
        return "rate_or_payment"
    if any(k in text for k in ["already got", "already funded", "already have a loan"]):
        return "already_solved"
    if any(k in text for k in ["not interested", "no thanks", "don't want", "do not want"]):
        return "not_interested"
    if any(k in text for k in ["busy", "call me later", "another time", "not a good time"]):
        return "timing"
    if any(k in text for k in ["scam", "trust", "legit", "real company"]):
        return "trust"
    if any(k in text for k in ["not in florida", "outside florida", "don't live in florida"]):
        return "eligibility_location"
    if any(k in text for k in ["more than", "higher amount", "need more", "larger amount"]):
        return "amount_too_low"
    return "other"


def _derive_outcome(disposition: CallDispositionState) -> str:
    if not disposition.customer_reached:
        return "not_reached"
    if disposition.customer_interested is False:
        return "not_interested"
    if disposition.referred_to_loan_officer:
        return "loan_officer_referral"
    if disposition.next_step:
        if "inspection" in disposition.next_step:
            return "inspection_scheduled"
        if "callback" in disposition.next_step:
            return "callback_scheduled"
    if disposition.customer_interested is True:
        return "interested_pending_next_step"
    return "incomplete"


def _derive_callback_needed(disposition: CallDispositionState) -> bool:
    if not disposition.customer_reached:
        return True
    if disposition.next_step and "callback" in disposition.next_step:
        return True
    return False


def _extract_requested_amount_from_transcript(transcript: list[TranscriptTurn]) -> float | None:
    amount_pattern = re.compile(r"\$?\s*(\d{2,6}(?:\.\d{1,2})?)")
    for turn in transcript:
        if not _is_user_role(turn.role):
            continue
        if any(k in turn.text.lower() for k in ["borrow", "need", "amount"]):
            match = amount_pattern.search(turn.text.replace(",", ""))
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
    return None


def _infer_disposition_from_transcript(
    disposition: CallDispositionState,
    transcript: list[TranscriptTurn],
    customer_name: str,
) -> None:
    user_turns = [t for t in transcript if _is_user_role(t.role)]
    if not user_turns:
        return

    user_texts = [t.text.lower() for t in user_turns if t.text.strip()]
    customer_first = customer_name.split()[0].lower() if customer_name.strip() else ""

    if not disposition.customer_reached:
        reached_markers = ["this is", "speaking", "that's me", "yes this is", "yeah this is"]
        if any(m in txt for txt in user_texts for m in reached_markers):
            disposition.customer_reached = True
        elif customer_first and any(customer_first in txt for txt in user_texts):
            disposition.customer_reached = True

    if disposition.customer_interested is None:
        negative_markers = [
            "not interested",
            "don't need",
            "do not need",
            "no thanks",
            "fuck off",
            "stop calling",
        ]
        positive_markers = ["sure", "yes", "yeah", "yep", "interested", "okay", "ok"]
        if any(m in txt for txt in user_texts for m in negative_markers):
            disposition.customer_interested = False
        elif any(re.search(rf"\b{re.escape(m)}\b", txt) for txt in user_texts for m in positive_markers):
            disposition.customer_interested = True

    if disposition.customer_interested is False and not disposition.not_interested_reason:
        for txt in user_texts:
            if any(
                m in txt
                for m in ["not interested", "don't need", "do not need", "no thanks", "fuck off", "stop calling"]
            ):
                disposition.not_interested_reason = txt
                break

    if disposition.requested_loan_amount_usd is None:
        disposition.requested_loan_amount_usd = _extract_requested_amount_from_transcript(
            transcript
        )


def _build_post_call_report(
    *,
    customer_profile: dict[str, Any],
    disposition: CallDispositionState,
    transcript: list[TranscriptTurn],
    call_started_at: datetime,
    call_ended_at: datetime,
    room_name: str,
) -> dict[str, Any]:
    user_transcript_text = " ".join(
        t.text for t in transcript if _is_user_role(t.role)
    ).strip()
    objection_source = disposition.not_interested_reason or (
        user_transcript_text if disposition.customer_interested is False else ""
    )
    requested_amount = disposition.requested_loan_amount_usd

    outcome = _derive_outcome(disposition)
    callback_needed = _derive_callback_needed(disposition)

    return {
        "call_metadata": {
            "customer_name": customer_profile.get("customer_name"),
            "campaign_type": customer_profile.get("campaign_type"),
            "script_version": customer_profile.get("script_version"),
            "room_name": room_name,
            "started_at_utc": call_started_at.isoformat(),
            "ended_at_utc": call_ended_at.isoformat(),
            "duration_seconds": max(
                0, int((call_ended_at - call_started_at).total_seconds())
            ),
        },
        "transcript": [asdict(turn) for turn in transcript],
        "analysis": {
            "outcome": outcome,
            "requested_amount_usd": requested_amount,
            "callback_needed": callback_needed,
            "callback_datetime": disposition.next_step_datetime
            if disposition.next_step and "callback" in disposition.next_step
            else None,
            "objection_category": _classify_objection(objection_source),
            "objection_reason": disposition.not_interested_reason,
        },
        "disposition": asdict(disposition),
    }


def _persist_post_call_report(report: dict[str, Any]) -> str:
    output_dir = os.getenv("POST_CALL_OUTPUT_DIR", "post_call_reports")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    customer_name = str(
        report.get("call_metadata", {}).get("customer_name", "unknown")
    ).lower()
    safe_name = re.sub(r"[^a-z0-9]+", "_", customer_name).strip("_") or "unknown"
    filename = f"{timestamp}_{safe_name}.json"
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return os.path.abspath(path)


def _post_report_to_webhook(report: dict[str, Any]) -> tuple[bool, str]:
    webhook_url = os.getenv("POST_CALL_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return False, "POST_CALL_WEBHOOK_URL not set; skipped webhook post."
    payload = json.dumps(report).encode("utf-8")
    req = request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=10) as resp:
            return True, f"Webhook post succeeded with status {resp.status}."
    except Exception as exc:
        return False, f"Webhook post failed: {exc}"


def _finalize_post_call(
    *,
    customer_profile: dict[str, Any],
    disposition: CallDispositionState,
    transcript: list[TranscriptTurn],
    call_started_at: datetime,
    room_name: str,
    trigger: str,
) -> None:
    customer_name = str(customer_profile.get("customer_name", "")).strip()
    _infer_disposition_from_transcript(disposition, transcript, customer_name)

    call_ended_at = datetime.now(tz=timezone.utc)
    report = _build_post_call_report(
        customer_profile=customer_profile,
        disposition=disposition,
        transcript=transcript,
        call_started_at=call_started_at,
        call_ended_at=call_ended_at,
        room_name=room_name,
    )
    report["call_metadata"]["finalization_trigger"] = trigger
    report_path = _persist_post_call_report(report)
    posted, webhook_message = _post_report_to_webhook(report)

    print("=== CALL DISPOSITION SUMMARY ===", flush=True)
    print(json.dumps(asdict(disposition), indent=2), flush=True)
    print("=== POST CALL REPORT ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved report: {report_path}", flush=True)
    print(webhook_message, flush=True)
    if posted:
        print("Webhook delivery: success", flush=True)


class SalesAgent(Agent):
    def __init__(
        self,
        instructions: str,
        disposition: CallDispositionState,
        knowledgebase: GCSKnowledgebase | None,
    ) -> None:
        super().__init__(instructions=instructions)
        self._disposition = disposition
        self._knowledgebase = knowledgebase

    @function_tool
    async def lookup_knowledgebase(
        self,
        context: RunContext,
        question: str,
    ) -> str:
        """Retrieve policy or process details from the client's GCS knowledge base.

        Use this tool whenever the customer asks about company policies, process,
        fees, timelines, eligibility, documentation, payment terms, or disputes.
        """
        if self._knowledgebase is None:
            return (
                "Knowledge base is not configured. "
                "KB_GCS_BUCKET (and optionally KB_GCS_PREFIX / KB_CLIENT_ID) must be set."
            )
        return await self._knowledgebase.query(question)

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


def build_agent() -> tuple[SalesAgent, CallDispositionState, GCSKnowledgebase | None]:
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
    knowledgebase = create_knowledgebase_from_env()
    disposition = CallDispositionState()
    return (
        SalesAgent(
            instructions=prompt,
            disposition=disposition,
            knowledgebase=knowledgebase,
        ),
        disposition,
        knowledgebase,
    )


async def entrypoint(ctx: JobContext) -> None:
    agent, disposition, knowledgebase = build_agent()
    if knowledgebase is not None:
        # Warm cache in parallel so first KB lookup is fast.
        asyncio.create_task(knowledgebase.warm())
    await ctx.connect()
    call_started_at = datetime.now(tz=timezone.utc)
    transcript: list[TranscriptTurn] = []
    finalized = False

    session = AgentSession(
        stt=assemblyai.STT(model="universal-streaming-multilingual"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(model="sonic-3", language="en"),
        vad=silero.VAD.load(),
    )

    @session.on("conversation_item_added")
    def _on_conversation_item_added(event: Any) -> None:
        item = getattr(event, "item", None)
        if item is None:
            return
        text = _extract_item_text(item)
        if not text:
            return
        transcript.append(
            TranscriptTurn(
                timestamp_utc=datetime.now(tz=timezone.utc).isoformat(),
                role=_to_role_string(getattr(item, "role", "unknown")),
                text=text,
                interrupted=bool(getattr(item, "interrupted", False)),
            )
        )

    def _finalize_once(trigger: str) -> None:
        nonlocal finalized
        if finalized:
            return
        finalized = True
        room_name = str(getattr(ctx.room, "name", "unknown"))
        try:
            _finalize_post_call(
                customer_profile=CUSTOMER_PROFILE,
                disposition=disposition,
                transcript=transcript,
                call_started_at=call_started_at,
                room_name=room_name,
                trigger=trigger,
            )
        except Exception as exc:
            print(f"Post-call finalization error: {exc}", flush=True)

    @session.on("close")
    def _on_session_close(event: Any) -> None:
        reason = str(getattr(event, "reason", "unknown"))
        _finalize_once(f"session_close:{reason}")

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
        _finalize_once("entrypoint_finally")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
