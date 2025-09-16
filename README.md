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

Let's look at more complex use-case:

