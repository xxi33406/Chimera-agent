"""Smart reply gating — decide whether to reply or stay silent."""

import re
from typing import List, Dict, Optional

# --- Fast-pass rules (always reply, skip LLM check) ---

_QUESTION_INDICATORS = re.compile(
    r'[?？]|'
    r'吗$|呢$|么$|吧$|'
    r'怎么|为什么|什么|哪|多少|几|'
    r'能不能|可以|帮我|请|'
    r'是不是',
    re.IGNORECASE
)

_TASK_INDICATORS = re.compile(
    r'帮我|请你|麻烦|能不能|'
    r'写一个|做一个|改一下|看一下|查一下|'
    r'please|help|can you|could you|'
    r'给我|告诉我|发给我',
    re.IGNORECASE
)

_MIN_LENGTH_FOR_FAST_PASS = 30  # Messages longer than this always get a reply


def needs_llm_check(user_message: str) -> bool:
    """Determine if this message needs LLM-based reply gating.
    
    Returns False (= always reply) for clear questions/tasks/long messages.
    Returns True (= needs LLM judgment) for short/ambiguous messages.
    """
    if not user_message or not user_message.strip():
        return True  # Empty message — let LLM decide
    
    text = user_message.strip()
    
    # Long messages always get a reply
    if len(text) > _MIN_LENGTH_FOR_FAST_PASS:
        return False
    
    # Questions always get a reply
    if _QUESTION_INDICATORS.search(text):
        return False
    
    # Task instructions always get a reply
    if _TASK_INDICATORS.search(text):
        return False
    
    # Short ambiguous message — needs LLM check
    return True


# --- LLM prompt for reply decision ---

_GATE_PROMPT = """你是一个对话意图判断助手。根据对话上下文，判断用户最新这条消息是否需要你回复。

判断标准：
- 如果用户在结束对话（如重复说晚安、拜拜）→ silent
- 如果用户只是简短回应表示知道了（嗯、哦、好的）且之前没有未完成的话题 → silent  
- 如果用户情绪不好且在冷淡回应（可能生气/不想聊）→ silent
- 如果用户在表达新想法、分享事情、寻求互动 → reply
- 如果不确定 → reply

最近对话：
{context}

用户最新消息：{message}

只输出一个词：reply 或 silent"""


def build_gate_prompt(user_message: str, recent_messages: List[Dict[str, str]]) -> str:
    """Build the reply-gating prompt with conversation context."""
    # Format recent messages (last 3 pairs max)
    context_lines = []
    for msg in recent_messages[-6:]:  # Last 6 messages (3 pairs)
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Multipart content — extract text
            content = " ".join(
                p.get("text", "") for p in content 
                if isinstance(p, dict) and p.get("type") == "text"
            )
        # Truncate long messages in context
        if len(content) > 100:
            content = content[:97] + "..."
        if role == "user":
            context_lines.append(f"用户: {content}")
        elif role == "assistant":
            context_lines.append(f"助手: {content}")
    
    context = "\n".join(context_lines) if context_lines else "(无上下文)"
    msg = user_message[:200] if len(user_message) > 200 else user_message
    return _GATE_PROMPT.format(context=context, message=msg)


def parse_gate_response(response: str) -> bool:
    """Parse LLM response. Returns True = should reply, False = stay silent."""
    if not response:
        return True  # Default to reply on error
    cleaned = response.strip().lower()
    if "silent" in cleaned:
        return False
    return True  # Default reply
