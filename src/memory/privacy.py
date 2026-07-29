from __future__ import annotations

import re
from enum import Enum

from src.memory.models import CandidateClaim, MemoryClaim, MemoryEvent


class LearningMode(str, Enum):
    AUTOMATIC = "automatic"
    EXPLICIT_PRIVATE = "explicit_private"
    EXPLICIT_GROUP = "explicit_group"
    EXPLICIT_GLOBAL = "explicit_global"


class Sensitivity(str, Enum):
    SAFE = "safe"
    SENSITIVE = "sensitive"
    HARD_SECRET = "hard_secret"


_PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN [^-\r\n]*PRIVATE KEY-----"
    r".*?"
    r"(?:-----END [^-\r\n]*PRIVATE KEY-----|$)",
    re.IGNORECASE | re.DOTALL,
)
_DATA_URL_SECRET_PATTERN = re.compile(
    r"data:image/[a-z0-9.+-]+;base64,[a-z0-9+/=\r\n]*",
    re.IGNORECASE,
)
_KNOWN_IMAGE_BASE64_PATTERN = re.compile(
    r"(?:iVBORw0KGgo|/9j/|R0lGOD(?:lh|dh)|UklGR)[a-z0-9+/=]{12,}",
    re.IGNORECASE,
)
_BEARER_PATTERN = re.compile(
    r"\bBearer\s+[a-z0-9._~+/=-]+",
    re.IGNORECASE,
)
_SECRET_TOKEN_PATTERN = re.compile(
    r"\b(?:sk-[a-z0-9_-]{8,}|gh[pousr]_[a-z0-9]{20,}|"
    r"AKIA[A-Z0-9]{16})\b",
    re.IGNORECASE,
)
_GEMINI_API_KEY_PATTERN = re.compile(
    r"(?<![a-z0-9_-])AIza[a-z0-9_-]{20,60}(?![a-z0-9_-])",
    re.IGNORECASE,
)
_COMPACT_OTP_PATTERN = re.compile(
    r"(?:(?<![a-z0-9_])OTP|验证码|校验码)\d{4,10}(?!\d)",
    re.IGNORECASE,
)
_COMPACT_PASSWORD_PATTERN = re.compile(
    r"(?:密码|口令)"
    r"(?=[a-z0-9._~!@#$%^&*+/=-]{6,64})"
    r"(?=[a-z0-9._~!@#$%^&*+/=-]{0,63}\d)"
    r"[a-z0-9._~!@#$%^&*+/=-]{6,64}"
    r"(?![a-z0-9._~!@#$%^&*+/=-])",
    re.IGNORECASE,
)
_COMPACT_PAYMENT_PASSWORD_PATTERN = re.compile(
    r"支付密码\d{4,12}(?!\d)",
)
_COMPACT_PAYMENT_ACCOUNT_PATTERN = re.compile(
    r"(?:银行(?:账号|帐号|账户)|银行卡号|支付账号)"
    r"\d{6,32}(?!\d)",
)
_COMPACT_CARD_VERIFICATION_PATTERN = re.compile(
    r"(?<![a-z0-9_])(?:CVV|CVC)\d{3,4}(?!\d)",
    re.IGNORECASE,
)
_HARD_SECRET_SEPARATOR = (
    r"(?:\s*(?:[:=：]|\bis\b|是|为)\s*|\s+)"
)
_CREDENTIAL_LABEL_PATTERN = (
    r"(?:"
    r"\b(?:api[_ -]?key|secret|access[_ -]?token|token|password|"
    r"passwd|credential|cookie|otp|verification[_ -]?code|"
    r"authorization)\b"
    r"|api\s*密钥|密钥|密码|口令|验证码|校验码"
    r")"
)
_PAYMENT_LABEL_PATTERN = (
    r"(?:"
    r"\b(?:payment[_ -]?(?:token|credential|data)|cvv|cvc|"
    r"card[_ -]?number|bank[_ -]?account)\b"
    r"|支付凭据|支付密码|支付信息|支付账号|"
    r"银行卡号|银行账户|银行账号|银行帐号"
    r")"
)
_LABELED_CREDENTIAL_PATTERN = re.compile(
    _CREDENTIAL_LABEL_PATTERN
    + _HARD_SECRET_SEPARATOR
    + r"[^\s,;，；]+",
    re.IGNORECASE,
)
_LABELED_PAYMENT_PATTERN = re.compile(
    _PAYMENT_LABEL_PATTERN
    + _HARD_SECRET_SEPARATOR
    + r"[^\r\n,;，；]+",
    re.IGNORECASE,
)
_COOKIE_HEADER_PATTERN = re.compile(
    r"(?im)\b(?:cookie|set-cookie)\s*[:=]\s*[^\r\n，；]+"
)
_PAYMENT_NUMBER_PATTERN = re.compile(
    r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"
)
_EMAIL_PATTERN = re.compile(
    r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[a-z]{2,}(?![\w.-])",
    re.IGNORECASE,
)
_PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:(?:\+?86[- ]?)?1[3-9]\d{9})(?!\d)"
)
_IDENTIFIER_PATTERN = re.compile(
    r"(?:身份证|身份证号|护照|passport|id[_ -]?card)"
    r"\s*(?:[:=：]|\bis\b|是|为)?\s*[a-z0-9-]{6,}",
    re.IGNORECASE,
)
_PRECISE_ADDRESS_PATTERN = re.compile(
    r"(?:"
    r"(?:住址|家庭住址|家庭地址|详细地址|门牌号|home[_ -]?address|"
    r"street[_ -]?address|lives?[_ -]?at)\s*(?:[:=：]|\bis\b|是|为)?\s*\S+"
    r"|(?:省|自治区|市|区|县).{0,40}(?:路|街|道|巷|弄)"
    r".{0,20}\d{1,6}(?:号|室|栋|单元)"
    r"|\d{1,6}\s+[\w .'-]{2,40}\s+"
    r"(?:street|st|road|rd|avenue|ave|lane|ln)\b"
    r")",
    re.IGNORECASE,
)
_HEALTH_PATTERN = re.compile(
    r"(?:"
    r"\b(?:HIV|AIDS)\b|艾滋|阳性|确诊|诊断|病史|病历|患有|"
    r"糖尿病|高血压|抑郁症|焦虑症|癌症|肿瘤|过敏|"
    r"medical|diagnosis|health[_ -]?condition|condition\s*[:=]"
    r")",
    re.IGNORECASE,
)
_FINANCIAL_PATTERN = re.compile(
    r"(?:收入|工资|薪资|负债|存款|财务状况|salary|income|debt|"
    r"bank[_ -]?account)",
    re.IGNORECASE,
)
_SENSITIVE_LABEL_PATTERN = re.compile(
    r"(?:home_address|street_address|location_exact|contact_point|"
    r"id_card|passport|medical|diagnosis|health_condition|"
    r"salary|bank_account|住址|身份证|病史)",
    re.IGNORECASE,
)
_URL_PATTERN = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
_IDENTITY_ASSERTION_PATTERN = re.compile(
    r"(?:真实姓名|身份证|护照|年龄|出生|职业|雇主|公司|学校|"
    r"(?:我|本人)\s*(?:是|叫))"
)
_RELATIONSHIP_ASSERTION_PATTERN = re.compile(
    r"(?:妻子|丈夫|老婆|老公|男友|女友|父亲|母亲|爸爸|妈妈|"
    r"儿子|女儿|亲属|伴侣|结婚|已婚|relationship)"
)
_UNSAFE_NAME_PUNCTUATION = re.compile(r"[。！？?；;：:=，,\r\n]")

MAX_CLAIM_EXCERPT_CHARS = 160
MAX_SAFE_NAME_CHARS = 32
MAX_SAFE_STYLE_CHARS = 80


def classify_sensitive_text(text: str) -> Sensitivity:
    value = str(text or "")
    if any(
        pattern.search(value)
        for pattern in (
            _PRIVATE_KEY_PATTERN,
            _DATA_URL_SECRET_PATTERN,
            _KNOWN_IMAGE_BASE64_PATTERN,
            _BEARER_PATTERN,
            _SECRET_TOKEN_PATTERN,
            _GEMINI_API_KEY_PATTERN,
            _COMPACT_OTP_PATTERN,
            _COMPACT_PASSWORD_PATTERN,
            _COMPACT_PAYMENT_PASSWORD_PATTERN,
            _COMPACT_PAYMENT_ACCOUNT_PATTERN,
            _COMPACT_CARD_VERIFICATION_PATTERN,
            _LABELED_CREDENTIAL_PATTERN,
            _LABELED_PAYMENT_PATTERN,
            _COOKIE_HEADER_PATTERN,
        )
    ):
        return Sensitivity.HARD_SECRET
    if any(
        _passes_luhn(match.group(0))
        for match in _PAYMENT_NUMBER_PATTERN.finditer(value)
    ):
        return Sensitivity.HARD_SECRET
    if any(
        pattern.search(value)
        for pattern in (
            _EMAIL_PATTERN,
            _PHONE_PATTERN,
            _IDENTIFIER_PATTERN,
            _PRECISE_ADDRESS_PATTERN,
            _HEALTH_PATTERN,
            _FINANCIAL_PATTERN,
            _SENSITIVE_LABEL_PATTERN,
        )
    ):
        return Sensitivity.SENSITIVE
    return Sensitivity.SAFE


def redact_hard_secrets(text: str) -> str:
    value = str(text or "")
    value = _PRIVATE_KEY_PATTERN.sub("[redacted:credential]", value)
    value = _DATA_URL_SECRET_PATTERN.sub("[redacted:image-data]", value)
    value = _KNOWN_IMAGE_BASE64_PATTERN.sub(
        "[redacted:image-data]",
        value,
    )
    value = _BEARER_PATTERN.sub("[redacted:credential]", value)
    value = _COOKIE_HEADER_PATTERN.sub("[redacted:credential]", value)
    value = _COMPACT_PAYMENT_PASSWORD_PATTERN.sub(
        "[redacted:payment-data]",
        value,
    )
    value = _COMPACT_PAYMENT_ACCOUNT_PATTERN.sub(
        "[redacted:payment-data]",
        value,
    )
    value = _COMPACT_CARD_VERIFICATION_PATTERN.sub(
        "[redacted:payment-data]",
        value,
    )
    value = _LABELED_PAYMENT_PATTERN.sub(
        "[redacted:payment-data]",
        value,
    )
    value = _COMPACT_OTP_PATTERN.sub("[redacted:credential]", value)
    value = _COMPACT_PASSWORD_PATTERN.sub(
        "[redacted:credential]",
        value,
    )
    value = _LABELED_CREDENTIAL_PATTERN.sub(
        "[redacted:credential]",
        value,
    )
    value = _GEMINI_API_KEY_PATTERN.sub(
        "[redacted:credential]",
        value,
    )
    value = _SECRET_TOKEN_PATTERN.sub("[redacted:credential]", value)
    return _PAYMENT_NUMBER_PATTERN.sub(
        lambda match: (
            "[redacted:payment-data]"
            if _passes_luhn(match.group(0))
            else match.group(0)
        ),
        value,
    )


def claim_contains_hard_secret(claim: MemoryClaim) -> bool:
    return any(
        classify_sensitive_text(value) is Sensitivity.HARD_SECRET
        for value in (
            f"{claim.predicate}={claim.value}",
            claim.source_excerpt,
        )
    )


def safe_group_personalization(claim: MemoryClaim) -> bool:
    if (
        claim.subject_type != "qq_user"
        or claim.subject_id != claim.speaker_qq
    ):
        return False
    value = str(claim.value or "").strip()
    if not value or "\n" in value or "\r" in value:
        return False
    if _URL_PATTERN.search(value):
        return False
    if classify_sensitive_text(
        f"{claim.predicate}={value}"
    ) is not Sensitivity.SAFE:
        return False
    if (
        _IDENTITY_ASSERTION_PATTERN.search(value)
        or _RELATIONSHIP_ASSERTION_PATTERN.search(value)
    ):
        return False

    preferred_name = (
        claim.predicate == "preferred_name"
        or claim.memory_type == "preferred_name"
    )
    response_style = claim.predicate == "response_style"
    if preferred_name:
        return (
            len(value) <= MAX_SAFE_NAME_CHARS
            and _UNSAFE_NAME_PUNCTUATION.search(value) is None
        )
    if response_style:
        return len(value) <= MAX_SAFE_STYLE_CHARS
    return False


def minimal_claim_excerpt(
    _event: MemoryEvent,
    candidate: CandidateClaim,
) -> str:
    value = re.sub(r"\s+", " ", str(candidate.value or "")).strip()
    if classify_sensitive_text(value) is Sensitivity.HARD_SECRET:
        return ""
    return value[:MAX_CLAIM_EXCERPT_CHARS]


def shared_claim_is_safe(claim: MemoryClaim) -> bool:
    if claim_contains_hard_secret(claim):
        return False
    sensitivity = classify_sensitive_text(
        f"{claim.predicate}={claim.value}"
    )
    if sensitivity is Sensitivity.SENSITIVE:
        return (
            claim.scope_type == "group"
            and claim.source_kind.startswith(
                f"command:{LearningMode.EXPLICIT_GROUP.value}:"
            )
        )
    return True


def _passes_luhn(value: str) -> bool:
    digits = [int(character) for character in value if character.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0
