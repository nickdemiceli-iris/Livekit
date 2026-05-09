from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_KB_PREFIX = "knowledgebase/"
_DEFAULT_CACHE_TTL_SECONDS = 300
_DEFAULT_MAX_TOTAL_BYTES = 256 * 1024
_DEFAULT_MAX_FILE_BYTES = 64 * 1024
_DEFAULT_MAX_CHUNK_CHARS = 1200

_TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".json",
    ".csv",
    ".yaml",
    ".yml",
}


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) > 1
    }


def _is_text_blob(path: str) -> bool:
    extension = PurePosixPath(path).suffix.lower()
    return extension in _TEXT_EXTENSIONS or extension == ""


def _chunk_text(content: str, *, max_chunk_chars: int) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
    chunks: list[str] = []
    current = ""

    for part in parts:
        if len(part) > max_chunk_chars:
            start = 0
            while start < len(part):
                end = start + max_chunk_chars
                chunks.append(part[start:end].strip())
                start = end
            continue

        candidate = f"{current}\n\n{part}".strip() if current else part
        if len(candidate) <= max_chunk_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = part

    if current:
        chunks.append(current)
    return [c for c in chunks if c]


def _normalize_domain(raw: str) -> str:
    value = raw.strip().rstrip(".,;")
    value = re.sub(r"^https?://", "", value, flags=re.IGNORECASE)
    return value.strip()


def _extract_payment_portal(snapshot: "_Snapshot") -> tuple[str, str] | None:
    patterns = (
        re.compile(
            r"(?:pay online at|payment portal)\s*[:\-]?\s*(https?://\S+|[a-z0-9.-]+\.[a-z]{2,}(?:/\S*)?)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\b([a-z0-9.-]+\.[a-z]{2,}(?:/\S*)?)\b",
            flags=re.IGNORECASE,
        ),
    )
    for chunk in snapshot.chunks:
        low = chunk.text.lower()
        if "payment portal" not in low and "pay online" not in low:
            continue
        for idx, pattern in enumerate(patterns):
            match = pattern.search(chunk.text)
            if not match:
                continue
            value = _normalize_domain(match.group(1))
            if idx == 1 and "." not in value:
                continue
            return value, chunk.source
    return None


def _extract_phone_payment_hours(snapshot: "_Snapshot") -> tuple[str, str] | None:
    pattern = re.compile(r"hours\s*:\s*([^\n]+)", flags=re.IGNORECASE)
    for chunk in snapshot.chunks:
        low = chunk.text.lower()
        if "payment by phone" not in low and "hours" not in low:
            continue
        match = pattern.search(chunk.text)
        if match:
            return match.group(1).strip(), chunk.source
    return None


def _extract_phone_payment_number(snapshot: "_Snapshot") -> tuple[str, str] | None:
    pattern = re.compile(
        r"(?:calling|call(?:ing)?)\s*:\s*([+()0-9\-\s]{10,})",
        flags=re.IGNORECASE,
    )
    for chunk in snapshot.chunks:
        low = chunk.text.lower()
        if "payment by phone" not in low and "call" not in low:
            continue
        match = pattern.search(chunk.text)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip(), chunk.source
    return None


@dataclass(frozen=True)
class _Chunk:
    source: str
    text: str
    token_set: set[str]


@dataclass(frozen=True)
class _Snapshot:
    loaded_at: float
    files: list[str]
    chunks: list[_Chunk]


class GCSKnowledgebase:
    def __init__(
        self,
        *,
        storage_client: Any,
        bucket_name: str,
        prefix: str,
        cache_ttl_seconds: int = _DEFAULT_CACHE_TTL_SECONDS,
        max_total_bytes: int = _DEFAULT_MAX_TOTAL_BYTES,
        max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
        max_chunk_chars: int = _DEFAULT_MAX_CHUNK_CHARS,
    ) -> None:
        self._storage_client = storage_client
        self._bucket_name = bucket_name.strip()
        self._prefix = prefix.strip().lstrip("/")
        self._cache_ttl_seconds = max(1, cache_ttl_seconds)
        self._max_total_bytes = max(1024, max_total_bytes)
        self._max_file_bytes = max(1024, max_file_bytes)
        self._max_chunk_chars = max(400, max_chunk_chars)

        self._snapshot: _Snapshot | None = None
        self._lock = asyncio.Lock()

    async def warm(self) -> None:
        await self._get_snapshot()

    async def query(self, question: str, *, top_k: int = 4) -> str:
        prompt = question.strip()
        if not prompt:
            return "Knowledge base query is empty."

        snapshot = await self._get_snapshot()
        if not snapshot.chunks:
            return "No client knowledge base content is available."

        query_tokens = _tokenize(prompt)
        lowered = prompt.lower()

        # Fast-path deterministic answers for operational questions that
        # frequently get hallucinated (portal domain, hours, phone payment line).
        asks_portal = any(
            k in lowered
            for k in (
                "pay online",
                "payment portal",
                "portal",
                "domain",
                "website",
                "site",
            )
        )
        asks_hours = any(
            k in lowered
            for k in (
                "hours",
                "open",
                "saturday",
                "weekend",
            )
        )
        asks_phone_payment = any(
            k in lowered
            for k in (
                "pay by phone",
                "payment by phone",
                "call to pay",
                "phone number",
            )
        )

        deterministic: list[str] = []
        sources: set[str] = set()
        if asks_portal:
            portal = _extract_payment_portal(snapshot)
            if portal:
                value, source = portal
                deterministic.append(f"Payment portal: {value}")
                sources.add(source)
        if asks_hours:
            hours = _extract_phone_payment_hours(snapshot)
            if hours:
                value, source = hours
                deterministic.append(f"Phone payment hours: {value}")
                sources.add(source)
        if asks_phone_payment:
            phone = _extract_phone_payment_number(snapshot)
            if phone:
                value, source = phone
                deterministic.append(f"Phone payment number: {value}")
                sources.add(source)

        if deterministic:
            ordered_sources = ", ".join(sorted(sources))
            return (
                "Knowledge base exact matches:\n"
                + "\n".join(f"- {line}" for line in deterministic)
                + f"\nSources: {ordered_sources}"
            )

        scored: list[tuple[float, _Chunk]] = []
        for chunk in snapshot.chunks:
            overlap = len(query_tokens & chunk.token_set)
            if query_tokens and overlap == 0:
                substring_bonus = 1.0 if any(t in chunk.text.lower() for t in query_tokens) else 0.0
                if substring_bonus == 0.0:
                    continue
                score = substring_bonus
            else:
                score = float(overlap)
            if lowered in chunk.text.lower():
                score += 3.0
            scored.append((score, chunk))

        if not scored:
            return (
                "No direct match found in the knowledge base for that question. "
                "Ask a follow-up question with more specific terms."
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = scored[: max(1, min(top_k, 8))]

        sections = []
        for index, (_, chunk) in enumerate(selected, start=1):
            sections.append(f"[{index}] Source: {chunk.source}\n{chunk.text}")

        sources = ", ".join(sorted(set(chunk.source for _, chunk in selected)))
        return (
            f"Knowledge base matches for: {prompt}\n"
            f"Sources: {sources}\n\n"
            + "\n\n".join(sections)
        )

    async def _get_snapshot(self) -> _Snapshot:
        now = time.monotonic()
        snapshot = self._snapshot
        if snapshot and (now - snapshot.loaded_at) < self._cache_ttl_seconds:
            return snapshot

        async with self._lock:
            now = time.monotonic()
            snapshot = self._snapshot
            if snapshot and (now - snapshot.loaded_at) < self._cache_ttl_seconds:
                return snapshot

            loaded = await asyncio.to_thread(self._load_snapshot_sync)
            self._snapshot = loaded
            return loaded

    def _load_snapshot_sync(self) -> _Snapshot:
        if not self._bucket_name or not self._prefix:
            return _Snapshot(loaded_at=time.monotonic(), files=[], chunks=[])

        files: list[str] = []
        chunks: list[_Chunk] = []
        total_bytes = 0
        try:
            blobs = sorted(
                self._storage_client.list_blobs(self._bucket_name, prefix=self._prefix),
                key=lambda b: b.name,
            )
        except Exception as exc:
            logger.warning("knowledgebase: failed listing GCS blobs: %s", exc)
            return _Snapshot(loaded_at=time.monotonic(), files=[], chunks=[])

        for blob in blobs:
            if blob.name.endswith("/") or not _is_text_blob(blob.name):
                continue
            if total_bytes >= self._max_total_bytes:
                break

            try:
                raw = blob.download_as_text(encoding="utf-8")
            except Exception as exc:
                logger.warning("knowledgebase: failed downloading %s: %s", blob.name, exc)
                continue

            if not raw.strip():
                continue

            content_bytes = raw.encode("utf-8")
            if len(content_bytes) > self._max_file_bytes:
                raw = content_bytes[: self._max_file_bytes].decode("utf-8", errors="ignore")

            remaining = self._max_total_bytes - total_bytes
            if remaining <= 0:
                break

            truncated = raw.encode("utf-8")[:remaining].decode("utf-8", errors="ignore")
            total_bytes += len(truncated.encode("utf-8"))

            source = blob.name.split("/")[-1]
            files.append(source)
            for piece in _chunk_text(truncated.strip(), max_chunk_chars=self._max_chunk_chars):
                chunks.append(_Chunk(source=source, text=piece, token_set=_tokenize(piece)))

        logger.info(
            "knowledgebase: loaded %d file(s) and %d chunk(s) from gs://%s/%s",
            len(files),
            len(chunks),
            self._bucket_name,
            self._prefix,
        )
        return _Snapshot(loaded_at=time.monotonic(), files=files, chunks=chunks)


def create_knowledgebase_from_env() -> GCSKnowledgebase | None:
    bucket_name = os.getenv("KB_GCS_BUCKET", "").strip()
    if not bucket_name:
        return None

    prefix = os.getenv("KB_GCS_PREFIX", "").strip()
    client_id = os.getenv("KB_CLIENT_ID", "").strip()
    if not prefix:
        prefix = (
            f"clients/{client_id}/knowledgebase/"
            if client_id
            else _DEFAULT_KB_PREFIX
        )

    try:
        from google.cloud import storage
    except Exception as exc:
        logger.warning("knowledgebase: google-cloud-storage unavailable: %s", exc)
        return None

    cache_ttl = int(os.getenv("KB_CACHE_TTL_SECONDS", str(_DEFAULT_CACHE_TTL_SECONDS)))
    max_total = int(os.getenv("KB_MAX_TOTAL_BYTES", str(_DEFAULT_MAX_TOTAL_BYTES)))
    max_file = int(os.getenv("KB_MAX_FILE_BYTES", str(_DEFAULT_MAX_FILE_BYTES)))
    max_chunk = int(os.getenv("KB_MAX_CHUNK_CHARS", str(_DEFAULT_MAX_CHUNK_CHARS)))

    return GCSKnowledgebase(
        storage_client=storage.Client(),
        bucket_name=bucket_name,
        prefix=prefix,
        cache_ttl_seconds=cache_ttl,
        max_total_bytes=max_total,
        max_file_bytes=max_file,
        max_chunk_chars=max_chunk,
    )
