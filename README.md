# OpenAI Chat Completions vs Responses API

A brief write-up about Chat Completions vs Responses API.

TL;DR
- Responses - use it for new projects especially when interfacing with agents, tool calling, structure outputs, multi-modal prompts and any multi-step workflow.
- Chat Completions - use it for simple, stateless chat or for existing integrations.

Let's look at an example:

## Chat Completions API

App/ Developer owns complete orchestration:
- Build `messages[]` and send to `/chat/completions`
- In case of tool calls, you run it
- Send response back as another message (and repeat)

## Responses API

Provides a single endpoint that:
- Can call tools and iterate internally
- Accepts multi-modal input
- Supports structured outputs (validated JSON schema or function outputs)
- Scales with longer chains

