# AWS setup for Claude on Bedrock

Phase 0 runs Claude through Amazon Bedrock in one personal AWS account. Until these steps are done the code uses the stub model backend, so nothing is blocked.

1. Turn on MFA for the root user. Create an admin user in IAM Identity Center (or IAM) for day-to-day work, and stop using root.
2. Create an **AWS Budgets** alarm (for example at $50 and $200 a month) **before** enabling any model.
3. Pick one region that offers Claude on Bedrock (default `us-east-1`) and use it for everything.
4. In the Bedrock console, open **Model access** and request access to Claude Haiku, Sonnet and Opus.
5. Create a least-privilege role for the app that allows only `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` on those models. Locally, use short-lived credentials (`aws sso login --profile fpso`). Never commit access keys.
6. Turn on CloudTrail and Bedrock model-invocation logging to an S3 bucket in the same account.

Then point the code at Bedrock:

```bash
export FPSO_MODEL_BACKEND=bedrock
export AWS_REGION=us-east-1
export AWS_PROFILE=fpso
```

Model roles and their default models are in `src/fpso/models/__init__.py` (`DEFAULT_ROLE_MODELS`). Every call is written to the action log with its token counts and an estimated cost.
