"""Data Shield — reversible data redaction for external API calls.

Replaces sensitive content with unique placeholders before sending to
external LLM/embedding APIs, and restores them from responses.
All mapping state lives only in memory — never persisted to disk.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ShieldContext:
    """Per-call mapping table: placeholder -> original text."""

    def __init__(self):
        self._map: Dict[str, str] = {}  # placeholder -> original
        self._reverse: Dict[str, str] = {}  # original -> placeholder (for dedup)
        self._counters: Dict[str, int] = {}  # type -> next counter

    def add(self, category: str, original: str) -> str:
        """Register a sensitive value. Returns its placeholder.

        If the same original was already registered, returns the existing placeholder.
        """
        if original in self._reverse:
            return self._reverse[original]

        n = self._counters.get(category, 0)
        self._counters[category] = n + 1
        placeholder = f"__SHIELD_{category.upper()}_{n}__"

        self._map[placeholder] = original
        self._reverse[original] = placeholder
        return placeholder

    @property
    def has_replacements(self) -> bool:
        return bool(self._map)

    def get_original(self, placeholder: str) -> Optional[str]:
        return self._map.get(placeholder)


class DataShield:
    """Reversible data redaction engine.

    Supports two policies:
    - "auto": Redact API keys, emails, phone numbers, IPs, file paths
    - "strict": auto + custom keywords + code variable patterns
    - "off": No redaction (passthrough)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.enabled: bool = config.get("enabled", True)
        self.policy: str = config.get("policy", "auto")
        self.shield_auxiliary: bool = config.get("shield_auxiliary", True)
        self.shield_embedding: bool = config.get("shield_embedding", False)
        self._custom_keywords: List[str] = config.get("custom_keywords", [])
        self._preserve_code_structure: bool = config.get(
            "preserve_code_structure", True
        )
        self._log_shielded: bool = config.get("log_shielded", False)

        # Compile patterns
        self._patterns = self._build_patterns()

        # Statistics tracking — backward compatible (default empty/zero counters).
        self._stats: Dict[str, Any] = {
            "total_calls": 0,
            "total_redactions": 0,
            "by_category": {},
            "chars_redacted": 0,
        }

    def _build_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """Build regex patterns for sensitive content detection.

        Returns list of (compiled_pattern, category) tuples.
        Order matters — earlier patterns take priority.
        """
        patterns: List[Tuple[re.Pattern, str]] = []

        # --- API Keys (highest priority) ---
        # Common API key prefixes (from agent/redact.py reference)
        api_prefixes = [
            r"sk-[a-zA-Z0-9]{20,}",  # OpenAI
            r"sk-proj-[a-zA-Z0-9\-_]{20,}",  # OpenAI project
            r"sk-ant-[a-zA-Z0-9\-_]{20,}",  # Anthropic
            r"ghp_[a-zA-Z0-9]{36,}",  # GitHub PAT
            r"gho_[a-zA-Z0-9]{36,}",  # GitHub OAuth
            r"github_pat_[a-zA-Z0-9_]{20,}",  # GitHub fine-grained
            r"glpat-[a-zA-Z0-9\-_]{20,}",  # GitLab
            r"xoxb-[a-zA-Z0-9\-]+",  # Slack bot
            r"xoxp-[a-zA-Z0-9\-]+",  # Slack user
            r"AKIA[A-Z0-9]{16}",  # AWS access key
            r"AIza[a-zA-Z0-9\-_]{35}",  # Google API
            r"ya29\.[a-zA-Z0-9\-_]+",  # Google OAuth
            r"[a-f0-9]{32,40}",  # Generic hex tokens (32-40 chars, conservative)
        ]
        # Combine into one pattern with word boundaries
        api_pattern = (
            r"(?<![a-zA-Z0-9_])(" + "|".join(api_prefixes) + r")(?![a-zA-Z0-9_])"
        )
        patterns.append((re.compile(api_pattern), "KEY"))

        # --- Named secrets (key=value patterns) ---
        secret_names = (
            r"(?:api[_-]?key|secret[_-]?key|access[_-]?token|"
            r"auth[_-]?token|password|passwd|credential)"
        )
        named_secret = re.compile(
            rf"({secret_names})(\s*[=:]\s*[\"']?)([^\s\"']+)([\"']?)",
            re.IGNORECASE,
        )
        patterns.append((named_secret, "SECRET"))

        # --- Email addresses ---
        patterns.append(
            (
                re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
                "EMAIL",
            )
        )

        # --- Phone numbers ---
        # Chinese mobile: 1[3-9]xxxxxxxxx
        # International: +CC XXXXXXXX
        patterns.append(
            (
                re.compile(
                    r"(?<!\d)(?:(?:\+\d{1,3}[\s\-]?)?\d{10,11}|1[3-9]\d{9})(?!\d)"
                ),
                "PHONE",
            )
        )

        # --- IP addresses ---
        patterns.append(
            (
                re.compile(
                    r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
                    r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?!\d)"
                ),
                "IP",
            )
        )

        # --- File paths (absolute only) ---
        # Unix: /home/user/..., /etc/...
        # Windows: C:\Users\..., D:\...
        patterns.append(
            (
                re.compile(
                    r'(?:[A-Z]:\\(?:[^\s\\:*?"<>|]+\\)*[^\s\\:*?"<>|]+'
                    r"|/(?:home|Users|etc|var|tmp|opt|usr)/[^\s]+)"
                ),
                "PATH",
            )
        )

        # --- URL credentials ---
        # Match password in URLs like https://user:password@host
        patterns.append(
            (
                re.compile(r"://([^:@\s]+):([^@\s]+)@"),
                "URL_CRED",
            )
        )

        # --- Strict mode: custom keywords ---
        if self.policy == "strict" and self._custom_keywords:
            for kw in self._custom_keywords:
                if kw and len(kw) >= 2:
                    # Escape regex special chars in keywords
                    escaped = re.escape(kw)
                    patterns.append(
                        (
                            re.compile(
                                rf"(?<![a-zA-Z0-9_]){escaped}(?![a-zA-Z0-9_])",
                                re.IGNORECASE,
                            ),
                            "CUSTOM",
                        )
                    )

        return patterns

    def shield_text(
        self, text: str, ctx: ShieldContext = None
    ) -> Tuple[str, ShieldContext]:
        """Redact sensitive content in text.

        Args:
            text: Input text to redact
            ctx: Optional existing context to reuse (for multi-field consistency)

        Returns:
            (redacted_text, context) tuple. Context needed for unshield.
        """
        if not self.enabled or self.policy == "off" or not text:
            ctx = ctx or ShieldContext()
            return text, ctx

        ctx = ctx or ShieldContext()
        _prev_map_size = len(ctx._map)
        _prev_total_chars = sum(len(v) for v in ctx._map.values())
        _prev_counters = dict(ctx._counters)
        result = text

        for pattern, category in self._patterns:
            if category == "SECRET":
                # Special handling for key=value pairs — only redact the value
                def _replace_secret(m: re.Match) -> str:
                    key_name = m.group(1)
                    separator = m.group(2)
                    value = m.group(3)
                    closing_quote = m.group(4)
                    if len(value) < 4:  # Too short to be a real secret
                        return m.group(0)
                    placeholder = ctx.add(category, value)
                    return f"{key_name}{separator}{placeholder}{closing_quote}"

                result = pattern.sub(_replace_secret, result)
            elif category == "URL_CRED":
                # Special: only redact the password part
                def _replace_url_cred(m: re.Match) -> str:
                    user = m.group(1)
                    password = m.group(2)
                    placeholder = ctx.add(category, password)
                    return f"://{user}:{placeholder}@"

                result = pattern.sub(_replace_url_cred, result)
            else:
                # General replacement — use factory to avoid late-binding issue
                def _make_replacer(cat: str):
                    def _replace(m: re.Match) -> str:
                        original = m.group(0)
                        # Skip very short matches (likely false positives)
                        if len(original) < 4 and cat not in ("IP",):
                            return original
                        return ctx.add(cat, original)

                    return _replace

                result = pattern.sub(_make_replacer(category), result)

        # --- Stats update (after all replacements) ---
        _new_count = len(ctx._map) - _prev_map_size
        if _new_count > 0:
            self._stats["total_redactions"] += _new_count
            for cat, cnt in ctx._counters.items():
                added = cnt - _prev_counters.get(cat, 0)
                if added > 0:
                    self._stats["by_category"][cat] = (
                        self._stats["by_category"].get(cat, 0) + added
                    )
            _new_chars = sum(len(v) for v in ctx._map.values()) - _prev_total_chars
            if _new_chars > 0:
                self._stats["chars_redacted"] += _new_chars

        if self._log_shielded and ctx.has_replacements:
            logger.debug("DataShield: redacted %d items", len(ctx._map))

        return result, ctx

    def shield_messages(
        self, messages: List[Dict[str, Any]], ctx: ShieldContext = None
    ) -> Tuple[List[Dict[str, Any]], ShieldContext]:
        """Redact sensitive content in OpenAI-format message list.

        Only processes string 'content' fields. Preserves message structure.
        Uses a shared context across all messages for consistent replacement.
        """
        if not self.enabled or self.policy == "off" or not messages:
            ctx = ctx or ShieldContext()
            return messages, ctx

        self._stats["total_calls"] += 1
        ctx = ctx or ShieldContext()
        shielded: List[Dict[str, Any]] = []

        for msg in messages:
            new_msg = dict(msg)  # Shallow copy
            content = msg.get("content")

            if isinstance(content, str) and content:
                new_msg["content"], ctx = self.shield_text(content, ctx)
            elif isinstance(content, list):
                # Multi-modal: list of content parts
                new_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text", "")
                        if text:
                            shielded_text, ctx = self.shield_text(text, ctx)
                            new_parts.append({**part, "text": shielded_text})
                        else:
                            new_parts.append(part)
                    else:
                        new_parts.append(part)
                new_msg["content"] = new_parts

            shielded.append(new_msg)

        return shielded, ctx

    def get_stats(self) -> Dict[str, Any]:
        """Return a copy of current shielding statistics.

        Keys: total_calls, total_redactions, by_category (dict), chars_redacted.
        Safe to call regardless of whether shielding is enabled.
        """
        return {
            "total_calls": self._stats["total_calls"],
            "total_redactions": self._stats["total_redactions"],
            "by_category": dict(self._stats["by_category"]),
            "chars_redacted": self._stats["chars_redacted"],
        }

    def unshield_text(self, text: str, ctx: ShieldContext) -> str:
        """Restore placeholders in text using the mapping from shield phase.

        Args:
            text: Text potentially containing __SHIELD_*__ placeholders
            ctx: The ShieldContext from the corresponding shield call

        Returns:
            Text with placeholders restored to original values.
        """
        if not text or not ctx or not ctx.has_replacements:
            return text

        result = text
        # Replace all known placeholders
        for placeholder, original in ctx._map.items():
            result = result.replace(placeholder, original)

        return result


# --- Module-level singleton ---

_shield_instance: Optional[DataShield] = None
_shield_initialized: bool = False


def init_data_shield(config: Optional[Dict[str, Any]] = None) -> DataShield:
    """Initialize the module-level DataShield singleton.

    Call this once at startup with the security.data_shield config section.
    """
    global _shield_instance, _shield_initialized
    _shield_instance = DataShield(config)
    _shield_initialized = True
    return _shield_instance


def get_data_shield() -> Optional[DataShield]:
    """Get the module-level DataShield singleton.

    Returns None if not initialized or if shielding is disabled.
    """
    if not _shield_initialized:
        return None
    return _shield_instance
