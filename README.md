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

Let's look at an example:

```python
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

class RepoSummary(BaseModel):
    name: str
    topics: list[str]
    risk_level: str

completion = client.chat.completions.create(
    model="gpt-5-mini",
    messages=[
        {"role": "system", "content": "Extract repo info into the schema."},
        {"role": "user", "content": "Summarize repo: awesome-embeddings. Fields: name, topics[], risk_level."},
    ],
    response_format={
        "type": "json_schema",
        "json_schema": RepoSummary.model_json_schema()
    },
)

parsed = RepoSummary.model_validate_json(completion.choices[0].message.content)
print(parsed.model_dump())
```

`chat.completions` requires you to manually enforce schema i.e. use `response_format="json_schema"` for strict schema-enforced JSON.

## Responses API

Provides a single endpoint that:
- Can call tools and iterate internally
- Accepts multi-modal input
- Supports structured outputs (validated JSON schema or function outputs)
- Scales with longer chains

```python
from openai import OpenAI
from pydantic import BaseModel
client = OpenAI()

class RepoSummary(BaseModel):
    name: str
    topics: list[str]
    risk_level: str

resp = client.responses.parse(
    model="gpt-5-mini",
    input=[
        {"role": "system", "content": "Extract repo info into the schema."},
        {"role": "user", "content": "Summarize repo: awesome-embeddings. Fields: name, topics[], risk_level."}
    ],
    text_format=RepoSummary,  # SDK automatically enforces the schema
)
summary: RepoSummary = resp.output_parsed
print(summary.model_dump())
```

`responses.parse` automatically maps output → Pydantic model.

Let's look at more complex use-case!

## Multi‑turn: tool calling + structured output ("Should I pack an umbrella?")

The user first asks for the weather (the model may call a tool), then asks for a strictly‑typed JSON answer about whether to pack an umbrella.

### Chat Completions API

We orchestrate tool calls and replay the full chat history each turn. JSON mode guarantees valid JSON, but we still enforce our own schema and handle retries.

### Responses API

We let the endpoint manage iterative reasoning. We pass only what’s relevant between steps, and `.parse()` enforces our Pydantic schema, returning a typed object.

Key takeaways:

- Orchestration: CC requires you to build and replay the entire `messages[]` every turn; Responses passes only the minimal context (tool result + follow‑up prompt).
- Schema enforcement: CC JSON mode ensures valid JSON but not your exact schema, so you must validate and handle errors yourself. Responses `.parse()` enforces the Pydantic schema and returns a typed object.
- Error handling: CC flows often need retry logic for malformed JSON or schema drift. Responses reduces drift via strict schema and internal iteration.
- Scalability: CC mixes tools and strict output with more prompt guards and state management. Responses scales to longer, multi‑modal chains without an ever‑growing transcript.

