How to correctly trace your submission with Langfuse

This project shows how to set up Langfuse tracing for your LangChain agent so that
token usage, costs, and latency are automatically tracked for the challenge.

What this script does:
- Configures a LangChain model via OpenRouter
- Initializes Langfuse with @observe() and CallbackHandler
- Generates a unique session ID in the format {TEAM_NAME}-{ULID}
- Sends 3 traced LLM calls grouped under a single session ID
- Passes session_id via metadata (config={"metadata": {"langfuse_session_id": ...}})
- Flushes traces to Langfuse after the run

How to view your traces:
You can see your tracing details on the Langfuse dashboard in the platform page.
The dashboard is associated with your team. It is not updated in real time,
so there may be a few minutes of delay before the latest traces appear.

Recommended setup:
- Python 3.13 (suggested)
- Langfuse SDK v3 (v4 is not fully supported and may cause unexpected issues)

For the full tutorial and additional examples, refer to
"Resource Management & Toolkit for the Challenge" in the Learn & Train section.
