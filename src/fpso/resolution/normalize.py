"""Cheap, source-agnostic normalizers used to find candidate links.

These are general text rules (separators, case, digit runs), not knowledge of how the
mock data was corrupted. Anything source-specific belongs in a confirmed pattern.
"""

from __future__ import annotations

import re
from datetime import date, datetime

DASHES = str.maketrans({"–": "-", "—": "-", "−": "-"})

# Equipment tags: 1-2 letters, optional separator, 3-4 digits, optional separator, train letter.
TAG_RE = re.compile(r"\b([A-Za-z]{1,2})[\s\-]?(\d{3,4})(?:[\s\-]?([A-Ca-c]))?\b")
PO_RE = re.compile(r"(?<!\d)0*(45\d{8})(?!\d)")
SIX_DIGITS_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")
EIGHT_DIGITS_RE = re.compile(r"(?<!\d)(\d{8})(?!\d)")


def norm_tag(tag: str) -> str:
    """'p-2301-a', 'P 2301A' and 'P2301A' all become 'P2301A'."""
    return re.sub(r"[\s\-]", "", tag.translate(DASHES)).upper()


def tag_group(tag: str) -> str:
    """The tag without its train letter: 'P2301A' becomes 'P2301'."""
    t = norm_tag(tag)
    return t[:-1] if t and t[-1] in "ABC" and t[-2:-1].isdigit() else t


def find_tags(text: str) -> set[str]:
    """Normalized equipment tags mentioned anywhere in free text."""
    out = set()
    for prefix, num, train in TAG_RE.findall(text.translate(DASHES)):
        out.add(f"{prefix}{num}{train}".upper())
    return out


def find_po_numbers(text: str) -> set[str]:
    return set(PO_RE.findall(text or ""))


def find_six_digit_runs(text: str) -> set[str]:
    return set(SIX_DIGITS_RE.findall(text or ""))


def find_eight_digit_runs(text: str) -> set[str]:
    return set(EIGHT_DIGITS_RE.findall(text or ""))


def parse_date(value: str) -> date | None:
    """Dates in the forms the sources use: 20270104, 2027-01-04, '2027-01-04 07:00'."""
    v = (value or "").strip()[:10]
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(v[: 8 if fmt == "%Y%m%d" else 10], fmt).date()
        except ValueError:
            continue
    return None


def trigrams(text: str) -> set[str]:
    t = re.sub(r"[^A-Z0-9 ]", " ", (text or "").upper())
    t = re.sub(r"\s+", " ", t).strip()
    return {t[i : i + 3] for i in range(len(t) - 2)} if len(t) >= 3 else {t}


def similarity(a: str, b: str) -> float:
    """Trigram Jaccard similarity; tolerant of abbreviation and typos."""
    ta, tb = trigrams(a), trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def leading_int(text: str) -> int | None:
    m = re.match(r"\s*(\d+)", text or "")
    return int(m.group(1)) if m else None


def containment(needle: str, haystack: str) -> float:
    """Share of the needle's trigrams found in the haystack; suits short text vs long."""
    tn, th = trigrams(needle), trigrams(haystack)
    return len(tn & th) / len(tn) if tn else 0.0
