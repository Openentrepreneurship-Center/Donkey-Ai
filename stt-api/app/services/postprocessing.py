"""Faster-Whisper 전사 후처리: 의료 용어 딕셔너리 교정 + 환각 제거."""

import re

# 정형외과 의료 용어 교정 매핑 (Whisper 오인식 → 올바른 용어)
MEDICAL_DICT = [
    ("전체 환수로", "전치환술 후"),
    ("전체 환술의", "전치환술의"),
    ("전체 환술을", "전치환술을"),
    ("전체 환술", "전치환술"),
    ("전체환술", "전치환술"),
    ("전체 환수", "전치환술"),
    ("전체환수", "전치환술"),
    ("무혈설 계세사", "무혈성 괴사"),
    ("무혈성 계세사", "무혈성 괴사"),
    ("무혈성계세사", "무혈성 괴사"),
    ("무혈성 계서", "무혈성 괴사"),
    ("계세사증", "괴사증"),
    ("계세사", "괴사"),
    ("관절념", "관절염"),
    ("관절륨", "관절염"),
    ("관질염", "관절염"),
    ("고관질", "고관절"),
    ("이용성증", "이형성증"),
    ("이영성증", "이형성증"),
    ("스테로지를", "스테로이드를"),
    ("스테로지", "스테로이드"),
    ("스테로에즈", "스테로이드"),
    ("릴리카", "리리카"),
    ("니리카", "리리카"),
    ("트리드로", "트리돌"),
    ("세레네스", "세레브렉스"),
    ("코피바", "본비바"),
    ("연고라 골절", "연골하 골절"),
    ("손통제", "진통제"),
    ("고기 고른 즙", "고름집"),
    ("구름집같이", "고름집같이"),
    ("구름집", "고름집"),
    ("액저리", "엑스레이"),
    ("이명옥", "임영욱"),
]


def apply_medical_dict(text: str) -> str:
    """딕셔너리 기반 의료 용어 교정."""
    for wrong, correct in MEDICAL_DICT:
        text = text.replace(wrong, correct)
    return text


def remove_hallucinations(text: str) -> str:
    """반복 환각 패턴 제거."""
    text = re.sub(r"(.{10,}?)\1{1,}", r"\1", text)
    patterns = [
        r"(감사합니다\.?\s*){3,}",
        r"(네\.?\s*){5,}",
        r"(MBC 뉴스.?\s*){2,}",
        r"(KBS 뉴스.?\s*){2,}",
        r"(시청해 주셔서 감사합니다.?\s*){2,}",
        r"(구독과 좋아요.?\s*){2,}",
    ]
    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    sentences = re.split(r"(?<=[.?!])\s+", text)
    if len(sentences) > 1:
        deduped = [sentences[0]]
        for s in sentences[1:]:
            if s != deduped[-1]:
                deduped.append(s)
        text = " ".join(deduped)
    return text.strip()


def deduplicate_segments(segments: list) -> list:
    """연속 중복 세그먼트 제거."""
    if not segments:
        return segments
    deduped = [segments[0]]
    for seg in segments[1:]:
        if seg.get("text", "").strip() != deduped[-1].get("text", "").strip():
            deduped.append(seg)
    return deduped


def postprocess_text(text: str) -> str:
    """환각 제거 → 딕셔너리 교정 순차 적용."""
    text = remove_hallucinations(text)
    text = apply_medical_dict(text)
    return text
