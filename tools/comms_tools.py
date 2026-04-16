"""Deterministic communications preprocessing tools.

Deviation from design brief: spaCy (en_core_web_sm) failed to install on the
host Python 3.9. Entity extraction is implemented with compiled regex patterns
instead. Output schema is identical to what the design brief specifies.
No LLM calls inside any function.
"""

import json
import re
from typing import Any, Dict, List

from langchain_core.tools import tool


# ── Compiled patterns ─────────────────────────────────────────────────────────

_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b")
_AMOUNT_RE = re.compile(
    r"[€$£]\s*[\d,]+(?:\.\d{1,2})?|[\d,]+(?:\.\d{1,2})?\s*(?:EUR|USD|GBP|CHF|GBP)"
)
_URL_RE = re.compile(r"https?://[^\s\"<>]+")
_URGENCY_PHRASES = [
    "urgent", "immediately", "right now", "asap", "act now",
    "suspended", "blocked", "verify your account", "verify your identity",
    "unusual activity", "click here", "confirm your identity",
    "account will be closed", "limited time", "wire transfer",
    "send money", "gift card", "unusual login", "unauthorised access",
]


@tool
def extract_comms_entities(sms_texts_json: str, mail_texts_json: str) -> Dict[str, Any]:
    """Extract fraud-relevant entities from SMS and email content using regex patterns.

    Note: Design brief specified spaCy en_core_web_sm. This implementation uses
    compiled regex instead because spaCy could not be installed on the host
    Python 3.9. Entity coverage (IBANs, amounts, URLs, urgency phrases) is the same.

    Args:
        sms_texts_json: JSON list of SMS thread strings.
        mail_texts_json: JSON list of email/HTML thread strings.

    Returns:
        {"ibans": list, "amounts": list, "urls": list, "urgency_phrases": list,
         "n_sms": int, "n_mails": int}
    """
    try:
        sms_texts: List[str] = json.loads(sms_texts_json)
        mail_texts: List[str] = json.loads(mail_texts_json)
    except (json.JSONDecodeError, TypeError):
        sms_texts, mail_texts = [], []

    combined_raw = " ".join(sms_texts + mail_texts)
    combined_lower = combined_raw.lower()

    ibans = list(set(_IBAN_RE.findall(combined_raw)))
    amounts = list(set(_AMOUNT_RE.findall(combined_lower)))
    urls = list(set(_URL_RE.findall(combined_raw)))
    urgency_found = [p for p in _URGENCY_PHRASES if p in combined_lower]

    return {
        "ibans": ibans[:20],         # cap list size for prompt token budget
        "amounts": amounts[:10],
        "urls": urls[:10],
        "urgency_phrases": urgency_found,
        "n_sms": len(sms_texts),
        "n_mails": len(mail_texts),
    }


def extract_comms_entities_direct(
    sms_texts: List[str],
    mail_texts: List[str],
) -> Dict[str, Any]:
    """Direct version that accepts lists — used by featurizer."""
    combined_raw = " ".join(sms_texts + mail_texts)
    combined_lower = combined_raw.lower()
    return {
        "ibans": list(set(_IBAN_RE.findall(combined_raw)))[:20],
        "amounts": list(set(_AMOUNT_RE.findall(combined_lower)))[:10],
        "urls": list(set(_URL_RE.findall(combined_raw)))[:10],
        "urgency_phrases": [p for p in _URGENCY_PHRASES if p in combined_lower],
        "n_sms": len(sms_texts),
        "n_mails": len(mail_texts),
    }


@tool
def find_amount_iban_mismatch(
    entities_json: str,
    sender_iban: str,
    recipient_iban: str,
    tx_amount: float,
) -> Dict[str, Any]:
    """Find mismatches between entities extracted from comms and the transaction record.

    Args:
        entities_json: JSON dict from extract_comms_entities output.
        sender_iban: sender_iban from the transaction (empty string if not present).
        recipient_iban: recipient_iban from the transaction (empty string if not present).
        tx_amount: Amount of the transaction.

    Returns:
        {"flags": list[str], "n_flags": int}
    """
    try:
        entities = json.loads(entities_json)
    except (json.JSONDecodeError, TypeError):
        return {"flags": [], "n_flags": 0}

    flags: List[str] = []
    tx_ibans = {i for i in [sender_iban, recipient_iban] if i and i.strip()}

    # IBAN in comms not matching the transaction IBANs
    for iban in entities.get("ibans", []):
        if tx_ibans and iban not in tx_ibans:
            flags.append(f"IBAN in comms ({iban[:8]}...) does not match transaction IBANs")

    # Urgency language always flagged
    for phrase in entities.get("urgency_phrases", []):
        flags.append(f"Urgency language detected: '{phrase}'")

    # Suspicious short-link URLs
    for url in entities.get("urls", []):
        if any(s in url for s in ["bit.ly", "tinyurl", "t.co", "goo.gl", "ow.ly"]):
            flags.append(f"Short-link URL in comms: {url}")

    return {"flags": flags, "n_flags": len(flags)}


def find_amount_iban_mismatch_direct(
    entities: Dict[str, Any],
    sender_iban: str,
    recipient_iban: str,
    tx_amount: float,
) -> List[str]:
    """Direct version that accepts a dict — used by featurizer."""
    flags: List[str] = []
    tx_ibans = {i for i in [sender_iban, recipient_iban] if i and str(i).strip()}
    for iban in entities.get("ibans", []):
        if tx_ibans and iban not in tx_ibans:
            flags.append(f"IBAN in comms ({iban[:8]}...) does not match transaction IBANs")
    for phrase in entities.get("urgency_phrases", []):
        flags.append(f"Urgency language detected: '{phrase}'")
    for url in entities.get("urls", []):
        if any(s in url for s in ["bit.ly", "tinyurl", "t.co", "goo.gl", "ow.ly"]):
            flags.append(f"Short-link URL: {url}")
    return flags
