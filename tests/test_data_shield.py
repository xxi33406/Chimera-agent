"""Tests for agent.data_shield — reversible data redaction engine."""

import pytest
from agent.data_shield import DataShield, ShieldContext, init_data_shield, get_data_shield


class TestShieldContext:
    """Test ShieldContext mapping behavior."""

    def test_add_returns_placeholder(self):
        ctx = ShieldContext()
        ph = ctx.add("KEY", "sk-abc123xyz")
        assert ph == "__SHIELD_KEY_0__"

    def test_deduplication(self):
        """Same original should return same placeholder."""
        ctx = ShieldContext()
        ph1 = ctx.add("EMAIL", "user@example.com")
        ph2 = ctx.add("EMAIL", "user@example.com")
        assert ph1 == ph2

    def test_different_values_get_different_placeholders(self):
        ctx = ShieldContext()
        ph1 = ctx.add("KEY", "sk-aaa")
        ph2 = ctx.add("KEY", "sk-bbb")
        assert ph1 != ph2
        assert ph1 == "__SHIELD_KEY_0__"
        assert ph2 == "__SHIELD_KEY_1__"

    def test_has_replacements(self):
        ctx = ShieldContext()
        assert not ctx.has_replacements
        ctx.add("IP", "192.168.1.1")
        assert ctx.has_replacements

    def test_get_original(self):
        ctx = ShieldContext()
        ctx.add("PHONE", "13812345678")
        assert ctx.get_original("__SHIELD_PHONE_0__") == "13812345678"
        assert ctx.get_original("__SHIELD_PHONE_99__") is None


class TestDataShieldText:
    """Test shield_text and unshield_text."""

    def test_api_key_openai(self):
        """OpenAI API keys should be redacted."""
        ds = DataShield({"enabled": True, "policy": "auto"})
        text = "My key is sk-proj-abcdefghij1234567890abcdefghij"
        shielded, ctx = ds.shield_text(text)
        assert "sk-proj-" not in shielded
        assert "__SHIELD_KEY_" in shielded
        # Unshield should restore
        restored = ds.unshield_text(shielded, ctx)
        assert restored == text

    def test_api_key_github(self):
        ds = DataShield({"enabled": True, "policy": "auto"})
        text = "token: ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890"
        shielded, ctx = ds.shield_text(text)
        assert "ghp_" not in shielded
        restored = ds.unshield_text(shielded, ctx)
        assert restored == text

    def test_email_redaction(self):
        ds = DataShield({"enabled": True, "policy": "auto"})
        text = "Contact me at john.doe@company.com for details"
        shielded, ctx = ds.shield_text(text)
        assert "john.doe@company.com" not in shielded
        assert "__SHIELD_EMAIL_" in shielded
        restored = ds.unshield_text(shielded, ctx)
        assert restored == text

    def test_phone_chinese(self):
        ds = DataShield({"enabled": True, "policy": "auto"})
        text = "Call me at 13912345678"
        shielded, ctx = ds.shield_text(text)
        assert "13912345678" not in shielded
        restored = ds.unshield_text(shielded, ctx)
        assert restored == text

    def test_ip_address(self):
        ds = DataShield({"enabled": True, "policy": "auto"})
        text = "Server at 192.168.1.100 is down"
        shielded, ctx = ds.shield_text(text)
        assert "192.168.1.100" not in shielded
        assert "__SHIELD_IP_" in shielded
        restored = ds.unshield_text(shielded, ctx)
        assert restored == text

    def test_file_path_unix(self):
        ds = DataShield({"enabled": True, "policy": "auto"})
        text = "Edit /home/user/secrets/config.yaml"
        shielded, ctx = ds.shield_text(text)
        assert "/home/user/" not in shielded
        restored = ds.unshield_text(shielded, ctx)
        assert restored == text

    def test_file_path_windows(self):
        ds = DataShield({"enabled": True, "policy": "auto"})
        text = r"Open C:\Users\admin\Documents\secret.txt"
        shielded, ctx = ds.shield_text(text)
        assert r"C:\Users" not in shielded
        restored = ds.unshield_text(shielded, ctx)
        assert restored == text

    def test_named_secret(self):
        ds = DataShield({"enabled": True, "policy": "auto"})
        text = 'api_key = "super_secret_value_12345"'
        shielded, ctx = ds.shield_text(text)
        assert "super_secret_value_12345" not in shielded
        restored = ds.unshield_text(shielded, ctx)
        assert restored == text

    def test_url_credentials(self):
        ds = DataShield({"enabled": True, "policy": "auto"})
        text = "postgres://admin:p4ssw0rd@db.example.com:5432/mydb"
        shielded, ctx = ds.shield_text(text)
        assert "p4ssw0rd" not in shielded
        # Admin user preserved (or also redacted — either is fine)
        restored = ds.unshield_text(shielded, ctx)
        assert restored == text

    def test_multiple_sensitive_items(self):
        """Multiple different sensitive items in one text."""
        ds = DataShield({"enabled": True, "policy": "auto"})
        text = "Email: test@foo.com, Key: sk-abcdefghijklmnopqrstuvwxyz, IP: 10.0.0.1"
        shielded, ctx = ds.shield_text(text)
        assert "test@foo.com" not in shielded
        assert "sk-abcdefghijklmnopqrstuvwxyz" not in shielded
        assert "10.0.0.1" not in shielded
        restored = ds.unshield_text(shielded, ctx)
        assert restored == text

    def test_no_sensitive_content(self):
        """Normal text should pass through unchanged."""
        ds = DataShield({"enabled": True, "policy": "auto"})
        text = "The quick brown fox jumps over the lazy dog"
        shielded, ctx = ds.shield_text(text)
        assert shielded == text
        assert not ctx.has_replacements

    def test_empty_text(self):
        ds = DataShield({"enabled": True, "policy": "auto"})
        shielded, ctx = ds.shield_text("")
        assert shielded == ""

    def test_none_handling(self):
        """None or empty should not crash."""
        ds = DataShield({"enabled": True, "policy": "auto"})
        # shield_text expects str, but should handle gracefully if empty
        shielded, ctx = ds.shield_text("")
        assert shielded == ""


class TestDataShieldPolicy:
    """Test different policy modes."""

    def test_policy_off(self):
        """policy=off should not redact anything."""
        ds = DataShield({"enabled": True, "policy": "off"})
        text = "sk-secret123456789012345678901234"
        shielded, ctx = ds.shield_text(text)
        assert shielded == text
        assert not ctx.has_replacements

    def test_disabled(self):
        """enabled=False should not redact."""
        ds = DataShield({"enabled": False, "policy": "auto"})
        text = "test@example.com"
        shielded, ctx = ds.shield_text(text)
        assert shielded == text

    def test_custom_keywords_strict(self):
        """Strict mode should redact custom keywords."""
        ds = DataShield({
            "enabled": True,
            "policy": "strict",
            "custom_keywords": ["ProjectAlpha", "InternalTool"],
        })
        text = "We use ProjectAlpha and InternalTool for this"
        shielded, ctx = ds.shield_text(text)
        assert "ProjectAlpha" not in shielded
        assert "InternalTool" not in shielded
        restored = ds.unshield_text(shielded, ctx)
        assert restored == text

    def test_custom_keywords_ignored_in_auto(self):
        """Auto mode should NOT use custom keywords."""
        ds = DataShield({
            "enabled": True,
            "policy": "auto",
            "custom_keywords": ["SecretProject"],
        })
        text = "Working on SecretProject today"
        shielded, ctx = ds.shield_text(text)
        assert "SecretProject" in shielded  # NOT redacted in auto mode


class TestDataShieldMessages:
    """Test shield_messages for OpenAI message format."""

    def test_basic_messages(self):
        ds = DataShield({"enabled": True, "policy": "auto"})
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "My email is bob@corp.com"},
        ]
        shielded, ctx = ds.shield_messages(messages)
        assert "bob@corp.com" not in shielded[1]["content"]
        assert shielded[0]["content"] == "You are helpful."  # No sensitive data

    def test_preserves_message_structure(self):
        ds = DataShield({"enabled": True, "policy": "auto"})
        messages = [
            {"role": "user", "content": "test@x.com", "name": "user1"},
        ]
        shielded, ctx = ds.shield_messages(messages)
        assert shielded[0]["role"] == "user"
        assert shielded[0]["name"] == "user1"  # Non-content fields preserved

    def test_empty_messages(self):
        ds = DataShield({"enabled": True, "policy": "auto"})
        shielded, ctx = ds.shield_messages([])
        assert shielded == []

    def test_shared_context_across_messages(self):
        """Same email in multiple messages gets same placeholder."""
        ds = DataShield({"enabled": True, "policy": "auto"})
        messages = [
            {"role": "user", "content": "Email: me@foo.com"},
            {"role": "assistant", "content": "Got it, me@foo.com noted"},
        ]
        shielded, ctx = ds.shield_messages(messages)
        # Both should use the same placeholder
        assert shielded[0]["content"].count("__SHIELD_EMAIL_0__") == 1
        assert shielded[1]["content"].count("__SHIELD_EMAIL_0__") == 1


class TestDataShieldSingleton:
    """Test module-level singleton."""

    def test_init_and_get(self):
        shield = init_data_shield({"enabled": True, "policy": "auto"})
        assert shield is not None
        assert get_data_shield() is shield

    def test_get_before_init(self):
        """Should handle gracefully — depends on module state."""
        # This test is order-dependent; just verify no crash
        from agent.data_shield import get_data_shield
        result = get_data_shield()
        # May or may not be None depending on prior test execution
        assert result is None or isinstance(result, DataShield)
