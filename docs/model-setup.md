# Model setup: Claude through the Anthropic API

Phase 0 calls Claude with an Anthropic API key. All Phase 0 data is mock, so no cloud account is needed. Until a key is set, the code uses the stub backend and nothing is blocked.

1. Sign up at [console.anthropic.com](https://console.anthropic.com) and verify your email.
2. Under **Billing**, add a small amount of credit (for example $20).
3. Under **Limits**, set a monthly spend limit (for example $50) so a runaway loop can't surprise you.
4. Optionally create a workspace named `fpso` so its spend shows separately.
5. Under **API keys**, create a key named `fpso-phase0`. Copy it once; the console won't show it again. Never commit it or paste it into a chat.

Then point the code at the API:

```bash
uv pip install -e ".[anthropic]"
export FPSO_MODEL_BACKEND=anthropic
export ANTHROPIC_API_KEY=...        # or keep it in a local .env file, which git ignores
fpso-model-check                    # one tiny call per role; prints the estimated cost
```

For runs in a cloud workspace, store `ANTHROPIC_API_KEY` as a secret in that environment's settings instead of a file.

Model roles and their default models are in `src/fpso/models/__init__.py` (`DEFAULT_ROLE_MODELS`). Every call is written to the action log with token counts and an estimated cost. To move to Amazon Bedrock later, see [aws-setup.md](aws-setup.md).
