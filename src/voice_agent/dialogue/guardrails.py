"""Deterministic safety guardrails, checked on every caller turn before
any dialogue logic runs.

- Distress / emergency language  -> escalate to a human, high priority.
- Medical-advice requests        -> escalate to a human.
- Social engineering (authority claims + data demands) -> refuse *and*
  escalate as suspected fraud.
- Requests for another person's information -> refuse. The agent holds
  other callers' data in its stores; none of it may ever be disclosed.

These are pure functions of the caller's text (plus the caller's own
name, to avoid flagging self-references), so they are exactly testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DISTRESS_PHRASES = [
    "kill myself", "hurt myself", "end my life", "don't want to live",
    "do not want to live", "want to die", "suicidal", "suicide",
    "overdose", "can't breathe", "cannot breathe", "can not breathe",
    "chest pain", "chest hurts", "bleeding heavily", "going to hurt myself",
    "hurt someone", "don't want to be here anymore",
    "do not want to be here anymore",
]

MEDICAL_ADVICE_PHRASES = [
    "should i take", "what should i take", "should i double",
    "double my dose", "increase my dose", "my dose", "what medicine",
    "what medication", "is it serious", "is this serious",
    "diagnose me", "what's wrong with me", "what is wrong with me",
    "do i need surgery", "can i stop taking", "should i stop taking",
]

AUTHORITY_PHRASES = [
    "i'm dr", "i am dr", "this is dr", "this is doctor", "i'm doctor",
    "i work at the clinic", "i work here", "i'm from the clinic",
    "calling from the clinic", "i'm a nurse", "i am a nurse",
    "i'm the manager", "i am the manager", "staff member",
    "i work for the clinic",
]

DATA_DEMAND_PHRASES = [
    "patient list", "list of patients", "all appointments",
    "all patients", "read me the", "give me the list",
    "every patient's", "all the appointments",
]

FAMILY_POSSESSIVES = [
    "my husband's", "my wife's", "my son's", "my daughter's",
    "my mother's", "my father's", "my brother's", "my sister's",
    "my partner's", "my mom's", "my dad's", "my child's",
]

_INFO_CONTEXT = ("appointment", "visit", "record", "information", "schedule", "phone", "time")
_NAME_POSSESSIVE_RE = re.compile(
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'s\s+"
    r"(appointment|visit|record|information|schedule|phone|time)"
)


@dataclass
class GuardrailHit:
    kind: str  # distress | medical_advice | social_engineering | privacy_other_info
    action: str  # escalate | refuse | refuse_and_escalate
    priority: str = "normal"
    detail: str = ""


def _contains_any(text: str, phrases: list[str]) -> str | None:
    for phrase in phrases:
        if phrase in text:
            return phrase
    return None


def check_caller_text(text: str, *, caller_name: str | None = None) -> GuardrailHit | None:
    lower = text.lower()

    hit = _contains_any(lower, DISTRESS_PHRASES)
    if hit:
        return GuardrailHit("distress", "escalate", priority="high", detail=hit)

    hit = _contains_any(lower, MEDICAL_ADVICE_PHRASES)
    if hit:
        return GuardrailHit("medical_advice", "escalate", detail=hit)

    authority = _contains_any(lower, AUTHORITY_PHRASES)
    demand = _contains_any(lower, DATA_DEMAND_PHRASES)
    if authority or demand:
        return GuardrailHit(
            "social_engineering",
            "refuse_and_escalate",
            priority="high",
            detail=authority or demand or "",
        )

    # Requests about another person's information.
    if any(p in lower for p in FAMILY_POSSESSIVES) and any(
        ctx in lower for ctx in _INFO_CONTEXT
    ):
        return GuardrailHit("privacy_other_info", "refuse", detail="family possessive")
    m = _NAME_POSSESSIVE_RE.search(text)
    if m:
        mentioned = m.group(1).strip().lower()
        if caller_name is None or mentioned != caller_name.strip().lower():
            return GuardrailHit(
                "privacy_other_info", "refuse", detail=f"name possessive: {m.group(1)}"
            )

    return None
