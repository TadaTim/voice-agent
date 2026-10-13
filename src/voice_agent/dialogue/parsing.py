"""Deterministic extraction of slots (name, phone, service, date/time)
from caller text. Everything here is rule-based and unit-testable; the
same parsers run on real STT output or simulated STT output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from ..tools.booking import SERVICE_SYNONYMS, canonical_service

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}

ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "twenty-first": 21,
    "twenty first": 21, "twenty-second": 22, "twenty second": 22,
    "twenty-third": 23, "twenty third": 23, "twenty-fourth": 24,
    "twenty fourth": 24, "twenty-fifth": 25, "twenty fifth": 25,
    "twenty-sixth": 26, "twenty sixth": 26, "twenty-seventh": 27,
    "twenty seventh": 27, "twenty-eighth": 28, "twenty eighth": 28,
    "twenty-ninth": 29, "twenty ninth": 29, "thirtieth": 30,
    "thirty-first": 31, "thirty first": 31,
}

HOUR_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}

WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

_ORDINAL_PATTERN = "|".join(sorted(ORDINALS, key=len, reverse=True))
_MONTH_PATTERN = "|".join(MONTHS)


@dataclass
class WhenParts:
    day: date | None = None
    tod: time | None = None

    def combine(self) -> datetime | None:
        if self.day is None or self.tod is None:
            return None
        return datetime.combine(self.day, self.tod)


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_when(
    text: str, now: datetime, *, default_from: datetime | None = None
) -> WhenParts:
    """Extract a (date, time) pair from free text. Either may be missing."""
    lower = text.lower()
    found_day: date | None = None
    found_time: time | None = None

    # --- date ---
    iso = re.search(r"(\d{4})-(\d{2})-(\d{2})", lower)
    if iso:
        found_day = _safe_date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
    if found_day is None:
        m = re.search(
            rf"\b({_MONTH_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", lower
        )
        if m:
            found_day = _safe_date(now.year, MONTHS[m.group(1)], int(m.group(2)))
        else:
            m = re.search(
                rf"\b({_MONTH_PATTERN})\s+({_ORDINAL_PATTERN})\b", lower
            )
            if m:
                found_day = _safe_date(now.year, MONTHS[m.group(1)], ORDINALS[m.group(2)])
            else:
                m = re.search(
                    rf"\b({_ORDINAL_PATTERN})\s+of\s+({_MONTH_PATTERN})\b", lower
                )
                if m:
                    found_day = _safe_date(now.year, MONTHS[m.group(2)], ORDINALS[m.group(1)])
    if found_day is None and "tomorrow" in lower:
        found_day = now.date() + timedelta(days=1)
    if found_day is None and "today" in lower:
        found_day = now.date()
    if found_day is None:
        for name, weekday in WEEKDAYS.items():
            if re.search(rf"\b{name}\b", lower):
                delta = (weekday - now.weekday()) % 7
                found_day = now.date() + timedelta(days=delta)
                break
    if found_day is None:
        # Day-of-month only (common in corrections: "No, the fifteenth").
        day_num: int | None = None
        m = re.search(r"\bthe\s+(\d{1,2})(?:st|nd|rd|th)\b", lower)
        if m:
            day_num = int(m.group(1))
        else:
            m = re.search(rf"\b({_ORDINAL_PATTERN})\b", lower)
            if m:
                day_num = ORDINALS[m.group(1)]
        if day_num is not None:
            base = default_from or now
            found_day = _safe_date(base.year, base.month, day_num)
            if found_day is not None and default_from is None and found_day < now.date():
                month = found_day.month % 12 + 1
                year = found_day.year + (1 if month == 1 else 0)
                found_day = _safe_date(year, month, day_num)

    # --- time ---
    m = re.search(r"\b(\d{1,2}):(\d{2})\s*(am|pm)?\b", lower)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
        meridiem = m.group(3)
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        if hour <= 23 and minute <= 59:
            found_time = time(hour, minute)
    if found_time is None:
        m = re.search(r"\b(\d{1,2})\s*(am|pm)\b", lower)
        if m:
            hour = int(m.group(1)) % 12
            if m.group(2) == "pm":
                hour += 12
            found_time = time(hour, 0)
    if found_time is None:
        m = re.search(
            r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
            r"\s*(am|pm|o'clock)\b",
            lower,
        )
        if m:
            hour = HOUR_WORDS[m.group(1)] % 12
            if m.group(2) == "pm":
                hour += 12
            found_time = time(hour, 0)
    if found_time is None and "noon" in lower:
        found_time = time(12, 0)
    if found_time is None and "midnight" in lower:
        found_time = time(0, 0)
    if found_time is None:
        m = re.search(r"\bat\s+(\d{1,2})\b(?!\s*(?:am|pm|:))", lower)
        if m and default_from is not None:
            found_time = time(int(m.group(1)) % 24, 0)

    return WhenParts(day=found_day, tod=found_time)


_PHONE_RE = re.compile(r"(\d{3})[-.\s]?(\d{3})[-.\s]?(\d{4})")
_LAST4_RE = re.compile(
    r"(?:ends?\s+(?:in|with)|last\s+four(?:\s+digits)?(?:\s+is|\s+are)?)\s*(\d{4})"
)

DIGIT_WORDS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9",
}
_DIGIT_WORD_PATTERN = "|".join(DIGIT_WORDS)


def _spelled_digit_run(text: str, start: int = 0) -> str:
    """Convert a run of spelled digit words ('zero one zero one') to digits.
    This is the classic recovery path when a recognizer keeps mishearing
    the same spoken digits: the caller spells them instead."""
    words = re.findall(r"[a-z]+", text.lower()[start:])
    digits: list[str] = []
    for word in words:
        if word in DIGIT_WORDS:
            digits.append(DIGIT_WORDS[word])
        elif digits:
            break
    return "".join(digits)


def extract_phone(text: str) -> str | None:
    m = _PHONE_RE.search(text)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    digits = _spelled_digit_run(text)
    if len(digits) == 10:
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    return None


def extract_phone_last4(text: str) -> str | None:
    lower = text.lower()
    m = _LAST4_RE.search(lower)
    if m:
        return m.group(1)
    cue = re.search(
        r"(?:ends?\s+(?:in|with)|last\s+four(?:\s+digits)?(?:\s+is|\s+are)?)",
        lower,
    )
    if cue:
        digits = _spelled_digit_run(lower, cue.end())
        if len(digits) >= 4:
            return digits[:4]
    return None


_NAME_INTRO_RE = re.compile(
    r"(?:my name is|this is|i am|i'm|name is)\s+"
    r"([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+){0,2})",
    re.IGNORECASE,
)
_NAME_STOPWORDS = {"and", "i", "the", "a", "an", "calling", "here", "from"}


def extract_name(text: str) -> str | None:
    m = _NAME_INTRO_RE.search(text)
    if m:
        words = [w for w in m.group(1).split() if w.lower() not in _NAME_STOPWORDS]
        if words:
            return " ".join(w.capitalize() for w in words[:3])
    # Bare name (an answer to "may I have your name?"): 1-3 plain words.
    stripped = text.strip().strip(".")
    if re.fullmatch(r"[A-Za-z]+(\s+[A-Za-z]+){0,2}", stripped) and len(stripped) >= 2:
        words = stripped.split()
        if all(w.lower() not in _NAME_STOPWORDS for w in words):
            return " ".join(w.capitalize() for w in words)
    return None


def extract_service(text: str) -> str | None:
    lower = text.lower()
    for synonym in sorted(SERVICE_SYNONYMS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(synonym)}\b", lower):
            return canonical_service(synonym)
    return None


def format_dt(dt: datetime) -> str:
    """'Tuesday, October 20 at 10:00 AM' -- deterministic, locale-free."""
    hour = dt.strftime("%I").lstrip("0") or "12"
    return (
        f"{dt.strftime('%A, %B')} {dt.day} at "
        f"{hour}:{dt.strftime('%M')} {dt.strftime('%p')}"
    )


def format_price(cents: int) -> str:
    if cents % 100 == 0:
        return f"${cents // 100}"
    return f"${cents / 100:.2f}"
