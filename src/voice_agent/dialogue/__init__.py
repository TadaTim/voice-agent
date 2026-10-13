"""Dialogue layer: deterministic state machine, slot parsing, guardrails.

No LLM is involved: the agent's behavior is a rule-based state machine so
that grounding and safety properties hold by construction and can be
evaluated exactly."""
