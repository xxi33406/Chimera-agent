#!/usr/bin/env python3

from __future__ import annotations

import sqlite3

from active_message_lib import (
    build_context_payload,
    format_dt,
    format_recent_messages,
    format_recent_outputs,
    hermes_home,
    load_feature_config,
)


def _load_interests_from_core() -> str:
    """Load user interests from Core Memory in memory.db."""
    try:
        db_path = hermes_home() / "memory.db"
        if not db_path.exists():
            return "none"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT value FROM memory_blocks WHERE label = 'interests'"
        ).fetchone()
        conn.close()
        if row and row["value"]:
            return row["value"]
    except Exception:
        pass
    return "none"


def _load_mood_from_core() -> str:
    """Load current user mood from Core Memory in memory.db."""
    try:
        db_path = hermes_home() / "memory.db"
        if not db_path.exists():
            return "neutral"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT value FROM memory_blocks WHERE label = 'mood'"
        ).fetchone()
        conn.close()
        if row and row["value"]:
            # Value format: "happy (conf=0.8)" — extract just the label
            return row["value"].split("(")[0].strip()
    except Exception:
        pass
    return "neutral"


def main() -> None:
    config = load_feature_config()
    payload = build_context_payload(config)
    latest_session = payload["latest_session"]

    print("ACTIVE_MESSAGE_CONTEXT_START")
    print("FEATURE=active-message")
    print(f"SEND_DECISION={payload['decision']}")
    print(f"REASON={','.join(payload['reasons'])}")
    print(f"NEXT_ELIGIBLE_AT={format_dt(payload['next_eligible_at'])}")
    print(f"NOW={format_dt(payload['now'])}")
    print(f"TARGET_PLATFORM={config['target_platform']}")
    print(f"TARGET_CHAT_ID={config['target_chat_id']}")
    print(f"TARGET_USER_ID={config['target_user_id']}")
    print(f"LAST_USER_MESSAGE_AT={format_dt(payload['last_user_message_at'])}")
    print(f"LAST_PROACTIVE_MESSAGE_AT={format_dt(payload['last_proactive_at'])}")
    print(f"TODAY_PROACTIVE_COUNT={payload['today_count']}")
    print(f"RECENT_SESSION_ID={latest_session['id'] if latest_session else 'N/A'}")
    print(f"RECENT_SESSION_TITLE={(latest_session['title'] or '').strip() if latest_session else 'N/A'}")
    print("RECENT_MESSAGES_START")
    print(format_recent_messages(payload["recent_messages"], payload["timezone"]))
    print("RECENT_MESSAGES_END")
    print("RECENT_PROACTIVE_OUTPUTS_START")
    print(format_recent_outputs(payload["recent_outputs"]))
    print("RECENT_PROACTIVE_OUTPUTS_END")
    # Load interests and mood from Core Memory
    print("USER_INTERESTS_START")
    print(_load_interests_from_core())
    print("USER_INTERESTS_END")
    print(f"USER_MOOD={_load_mood_from_core()}")
    print("ACTIVE_MESSAGE_CONTEXT_END")


if __name__ == "__main__":
    main()
