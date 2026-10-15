"""Corruption profiles: how the clean world turns into messy extracts.

Each profile is a named, versioned set of probabilities. Profile ``A`` is the development
profile the pipeline is tuned on. Profile ``B`` uses different messiness rules and is held
back for the final Phase 0 measurement, so the result is not just overfitting to ``A``.
Change a profile only by adding a new version; results cite the version they ran on.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

PROFILES: dict[str, dict] = {
    "A": {
        "version": 1,
        "description": "Development profile: separator and case drift, missing references, "
        "abbreviated manifests, PO numbers in email subjects.",
        "typo_rate": 0.05,
        "erp_techident_missing": 0.10,
        "erp_po_wo_missing": 0.30,
        "p6_tag_variant": 0.50,  # 'P-2301A' written as 'P2301A', 'P-2301-A', 'p-2301a'
        "p6_tag_drop_train": 0.0,
        "p6_wo_udf_missing": 0.40,
        "p6_wo_in_name": 0.15,
        "manifest_po_clean": 0.50,
        "manifest_po_variant": 0.20,  # 'PO#4500210033', '4500210033-10'
        "manifest_po_leading_zeros": 0.0,
        "manifest_merge_lines": 0.0,
        "manifest_abbreviate": 0.70,
        "email_po_in_subject": 0.60,
        "email_po_in_body": 0.25,  # remainder: no PO number at all
        "email_po_quoted_only": 0.0,
        "email_date_format": "%d %b %Y",
        "unicode_dashes": 0.0,
    },
    "B": {
        "version": 1,
        "description": "Held-back profile: train suffixes dropped, merged manifest lines, "
        "zero-padded PO numbers, PO only in quoted replies, day-first dates.",
        "typo_rate": 0.08,
        "erp_techident_missing": 0.20,
        "erp_po_wo_missing": 0.45,
        "p6_tag_variant": 0.30,
        "p6_tag_drop_train": 0.40,
        "p6_wo_udf_missing": 0.60,
        "p6_wo_in_name": 0.05,
        "manifest_po_clean": 0.35,
        "manifest_po_variant": 0.10,
        "manifest_po_leading_zeros": 0.20,
        "manifest_merge_lines": 0.15,
        "manifest_abbreviate": 0.85,
        "email_po_in_subject": 0.40,
        "email_po_in_body": 0.15,
        "email_po_quoted_only": 0.30,
        "email_date_format": "%d/%m/%Y",
        "unicode_dashes": 0.10,
    },
}

ABBREVIATIONS = {
    "MECHANICAL": "MECH",
    "SEAL": "SL",
    "BEARING": "BRG",
    "GASKET": "GSKT",
    "ASSEMBLY": "ASSY",
    "ELEMENT": "ELEM",
    "CARTRIDGE": "CART",
    "HYDRAULIC": "HYD",
    "FILTER": "FILT",
    "ACTUATOR": "ACT",
    "PACKING": "PKG",
    "CONTROL": "CTRL",
    "COMPRESSOR": "COMP",
    "CENTRIFUGAL": "CENT",
    "EXCHANGER": "EXCH",
    "SACRIFICIAL": "SACR",
    "TRANSFER": "TRANS",
    "JOURNAL": "JNL",
}


@dataclass
class Corruptor:
    profile_name: str
    seed: int
    rng: random.Random = field(init=False)

    def __post_init__(self):
        if self.profile_name not in PROFILES:
            raise ValueError(f"unknown profile {self.profile_name!r}; choose from {list(PROFILES)}")
        self.p = PROFILES[self.profile_name]
        # Offset so profile A and B corrupt the same world differently.
        self.rng = random.Random(self.seed * 1009 + ord(self.profile_name))

    def chance(self, key: str) -> bool:
        return self.rng.random() < self.p[key]

    def typo(self, text: str) -> str:
        """Swap two adjacent letters in one word, at the profile's typo rate."""
        if not text or self.rng.random() >= self.p["typo_rate"]:
            return text
        idx = [i for i in range(len(text) - 1) if text[i].isalpha() and text[i + 1].isalpha()]
        if not idx:
            return text
        i = self.rng.choice(idx)
        return text[:i] + text[i + 1] + text[i] + text[i + 2 :]

    def dash(self, text: str) -> str:
        if self.p["unicode_dashes"] and self.rng.random() < self.p["unicode_dashes"]:
            return text.replace("-", "–")
        return text

    def tag(self, tag: str) -> str:
        """How a human writes an equipment tag in a schedule or manifest."""
        m = re.fullmatch(r"([A-Z]+)-(\d+)([A-C]?)", tag)
        if not m:
            return tag
        prefix, num, train = m.groups()
        if train and self.chance("p6_tag_drop_train"):
            train = ""
        if self.chance("p6_tag_variant"):
            style = self.rng.choice(["nodash", "trainsep", "lower", "space"])
            if style == "nodash":
                return f"{prefix}{num}{train}"
            if style == "trainsep" and train:
                return f"{prefix}-{num}-{train}"
            if style == "lower":
                return f"{prefix}-{num}{train}".lower()
            return f"{prefix} {num}{train}"
        return self.dash(f"{prefix}-{num}{train}")

    def po_ref(self, po_no: str, item: str) -> str:
        """A PO reference as written on a manifest line, or '' when it is missing."""
        r = self.rng.random()
        clean = self.p["manifest_po_clean"]
        variant = clean + self.p["manifest_po_variant"]
        zeros = variant + self.p["manifest_po_leading_zeros"]
        if r < clean:
            return po_no
        if r < variant:
            return self.rng.choice([f"PO#{po_no}", f"{po_no}-{int(item)}", f"PO {po_no}/{item}"])
        if r < zeros:
            return self.rng.choice([f"00{po_no}", po_no[-6:]])
        return ""

    def abbreviate(self, text: str) -> str:
        if not self.chance("manifest_abbreviate"):
            return text
        words = re.split(r"(\W+)", text)
        return "".join(ABBREVIATIONS.get(w, w) if self.rng.random() < 0.7 else w for w in words)

    def email_po_placement(self) -> str:
        """Where the PO number appears in a supplier email: subject, body, quoted, or none."""
        r = self.rng.random()
        subject = self.p["email_po_in_subject"]
        body = subject + self.p["email_po_in_body"]
        quoted = body + self.p["email_po_quoted_only"]
        if r < subject:
            return "subject"
        if r < body:
            return "body"
        if r < quoted:
            return "quoted"
        return "none"
