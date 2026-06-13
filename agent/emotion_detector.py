"""LLM-based emotion detection via auxiliary model. ~200 tokens/call."""

from typing import Tuple

# Valid emotion labels
VALID_EMOTIONS = ("happy", "sad", "anxious", "angry", "tired", "neutral")

_EMOTION_PROMPT = """判断以下用户消息的情绪状态。只输出一个词，必须是以下之一：
happy, sad, anxious, angry, tired, neutral

用户消息：{message}

情绪："""


def detect_emotion_prompt(user_message: str) -> str:
    """Build the emotion detection prompt for auxiliary model."""
    # Truncate to save tokens
    msg = user_message[:200] if len(user_message) > 200 else user_message
    return _EMOTION_PROMPT.format(message=msg)


def parse_emotion_response(response: str) -> Tuple[str, float]:
    """Parse LLM response into (emotion_label, confidence).
    
    Returns (label, 0.8) for valid labels, ("neutral", 0.0) for invalid.
    """
    if not response:
        return ("neutral", 0.0)
    cleaned = response.strip().lower().split()[0] if response.strip() else ""
    # Remove punctuation
    cleaned = cleaned.rstrip(".,;!?。，；！？")
    if cleaned in VALID_EMOTIONS:
        return (cleaned, 0.8 if cleaned != "neutral" else 0.0)
    # Fuzzy match
    for emo in VALID_EMOTIONS:
        if emo in cleaned or cleaned in emo:
            return (emo, 0.6)
    return ("neutral", 0.0)


# Tone guidance map — injected into system prompt as emotion response directive
TONE_GUIDANCE: dict[str, str] = {
    "happy": "用户心情不错，回复可以轻松活泼一些",
    "sad": "用户似乎心情低落，回复语气温柔体贴，多一些关怀和陪伴感",
    "anxious": "用户可能感到焦虑，回复平静稳定，给予安全感和鼓励",
    "angry": "用户情绪激动，回复冷静不火上浇油，适当共情",
    "tired": "用户很疲惫，回复简短温暖，不制造额外信息负担",
    "neutral": "",
}
