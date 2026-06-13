---
name: active-message
description: "Proactive companion messaging via cron — sends contextual messages during idle periods"
version: 1.0.0
author: local
license: MIT
platforms: [linux, macos]
prerequisites:
  env_vars: []
metadata:
  hermes:
    tags: [Companion, Cron, Proactive, Messaging, WeChat]
---

# Active Message

Cron-based proactive companion messaging. Sends contextual messages (quotes, news, quizzes, care, music recommendations) during user idle periods based on configurable rules.

## Setup

1. Copy config to `~/.hermes/active-message/config.yaml`
2. Register as cron job: `hermes cron add --name active-message --schedule "every 25m" --script active-message-build-context.py --prompt-file ~/.hermes/active-message/cron_prompt.txt`

## Configuration

See `config.yaml` for all options including active window, idle thresholds, daily limits, and topic weights.

## Files

- `active_message_lib.py` — Core logic: context building, decision engine, state management
- `build_context.py` — Cron pre-script that outputs structured context for the LLM
- `config.yaml` — Configuration template (copy to ~/.hermes/active-message/)
- `cron_prompt.txt` — LLM prompt template for generating messages
- `soul.md` — Character personality definition
- `state.json` — Runtime state (auto-managed)
