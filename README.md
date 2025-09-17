# OpenAI Chat Completions vs Responses API

A brief write-up about Chat Completions vs Responses API.

TL;DR
- Responses: recommended for new projects, agentic by default, multi‑step flows with optional server‑managed state across turns and evented streaming.
- Chat Completions: ideal for simple, stateless prompts and existing CC integrations; you manage conversation state and validate schemas yourself.

### Quick chooser

| Scenario | Prefer |
| --- | --- |
| Stateless, single‑turn chat or existing CC integration | Chat Completions |
| Multi‑turn with state preserved server‑side | Responses |
| Agent‑like multi‑step orchestration | Responses |
| Connect the model to tools hosted on remote MCP servers | Responses |
| Fine‑grained manual control over history/retries | Chat Completions |
| Longer chains with internal iteration | Responses |

### Minimal side‑by‑side

```python
# Chat Completions (manual orchestration + JSON mode)
resp = client.chat.completions.create(
    model="gpt-5-mini",
    messages=messages,
    response_format={"type": "json_object"}
)
data = json.loads(resp.choices[0].message.content)
item = Item.model_validate(data)

# Responses (no history replay + schema‑enforced parse)
resp = client.responses.parse(
    model="gpt-5-mini",
    input=[{"role": "user", "content": "..."}],
    text_format=Item,
)
item: Item = resp.output_parsed
```



## Chat Completions API

Build and replay `messages[]` every turn, call `/chat/completions`, and validate strict JSON against your schema using `response_format` plus your own parsing.



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

Single endpoint with optional server‑side statefulness and an SDK parse helper for typed outputs. `.parse(..., text_format=...)` returns a typed object (`output_parsed`).

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
    input=[{"role": "user", "content": "Summarize repo: awesome-embeddings."}],
    text_format=RepoSummary,
)
print(resp.output_parsed.model_dump())
```

`responses.parse` automatically maps output → Pydantic model.

A more complex use case

## Multi‑turn: tool calling + structured output ("Should I pack an umbrella?")

The user first asks for the weather (the model may call a tool), then asks for a strictly‑typed JSON answer about whether to pack an umbrella.

### Chat Completions API

We orchestrate tool calls and replay the full chat history each turn. JSON mode guarantees valid JSON, but we still enforce our own schema and handle retries.

```python
import json
from openai import OpenAI
from pydantic import BaseModel, ValidationError
client = OpenAI()

# Chat Completions key points:
# - You must replay the growing chat history each turn.
# - JSON mode guarantees JSON, but you still validate your schema.

def get_weather(city: str) -> dict:
    return {"city": city, "temp_c": 17, "condition": "rain"}

class PackAdvice(BaseModel):
    umbrella: bool
    rationale: str

messages = [{"role": "user", "content": "What's the weather in Paris today?"}]

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather by city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }
}]

# Turn 1: model may call the tool
resp1 = client.chat.completions.create(model="gpt-5-mini", messages=messages, tools=tools, tool_choice="auto")
assistant_msg = resp1.choices[0].message
messages.append(assistant_msg)

if assistant_msg.tool_calls:
    call = assistant_msg.tool_calls[0]
    args = json.loads(call.function.arguments)
    weather = get_weather(**args)
    messages.append({
        "role": "tool",
        "tool_call_id": call.id,
        "name": "get_weather",
        "content": json.dumps(weather)
    })

# Turn 2: ask for JSON and validate it against PackAdvice
messages.append({"role": "user", "content": "Great, should I pack an umbrella? Return JSON only."})
resp2 = client.chat.completions.create(
    model="gpt-5-mini",
    messages=messages,
    response_format={"type": "json_object"}
)

try:
    data = json.loads(resp2.choices[0].message.content)
    advice = PackAdvice.model_validate(data)
    print(advice.model_dump())
except (json.JSONDecodeError, ValidationError) as e:
    print("Invalid JSON for PackAdvice:", e)
```

### Responses API

We let the endpoint manage iterative reasoning. We pass only what’s relevant between steps, and `.parse()` enforces our Pydantic schema, returning a typed object.

```python
import json
from openai import OpenAI
from pydantic import BaseModel
client = OpenAI()

# Responses key points:
# - No chat history replay; send just what's needed.
# - parse(..., text_format=...) enforces your schema.

def get_weather(city: str) -> dict:
    return {"city": city, "temp_c": 17, "condition": "rain"}

class PackAdvice(BaseModel):
    umbrella: bool
    rationale: str

tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "Get current weather by city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
    }
}]

# Turn 1: maybe a tool call
resp1 = client.responses.create(
    model="gpt-5-mini",
    input=[{"role": "user", "content": "What's the weather in Paris today?"}],
    tools=tools,
)

func_calls = [o for o in getattr(resp1, "output", []) if getattr(o, "type", None) in ("function_call", "tool_call")]
if not func_calls:
    print(resp1.output_text)
else:
    fc = func_calls[0]
    fc_args = getattr(fc, "arguments", {})
    args = fc_args if isinstance(fc_args, dict) else json.loads(fc_args)
    weather = get_weather(**args)
    tool_call_id = getattr(fc, "call_id", None) or getattr(fc, "id", None)

    # Turn 2: strict schema output
    resp2 = client.responses.parse(
        model="gpt-5-mini",
        input=[
            {
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "get_weather result (tool_call_id=" + str(tool_call_id) + "): " + json.dumps(weather)
                }]
            },
            {"role": "user", "content": "Great, should I pack an umbrella? Return JSON only."}
        ],
        text_format=PackAdvice,
    )

    advice: PackAdvice = resp2.output_parsed
    print(advice.model_dump())
```

Key takeaways

- Orchestration: CC requires replaying the entire `messages[]` each turn; Responses can keep state on the server and lets you pass only what’s needed.
- Schema handling: Both support schema‑constrained outputs; CC uses JSON schema modes with manual validation; Responses adds an SDK parse helper that returns typed objects.
- MCP: Responses supports remote MCP servers.
- Streaming: CC streams token deltas appended to `message.content`. Responses emits semantic events (e.g. `response.output_text.delta`).
- Storage: Responses are stored by default; Chat Completions are stored not by default. Set `store=False` to disable.
- Error handling: CC flows often need retry logic for malformed JSON or schema drift. Responses 
reduces drift via strict schema and internal iteration.
- Scalability: CC mixes tools and strict output with more prompt guards and state management. 
Responses scales to longer, multi‑modal chains without an ever‑growing transcript.

## Migration: Chat Completions → Responses

- General calls: replace `client.chat.completions.create(...)` with `client.responses.create(...)`.
- Typed outputs: replace JSON mode + manual validation with `client.responses.parse(..., text_format=YourModel)`.
- State: stop replaying the entire `messages[]`; either pass minimal context between turns or use `store=True` and `previous_response_id`.

```python
# Before (Chat Completions)
import json
resp = client.chat.completions.create(
    model="gpt-5-mini",
    messages=messages,
    response_format={"type": "json_object"}
)
data = json.loads(resp.choices[0].message.content)
obj = MyModel.model_validate(data)

# After (Responses)
resp = client.responses.parse(
    model="gpt-5-mini",
    input=[{"role": "user", "content": "..."}],  # No history replay
    text_format=MyModel,
)
obj: MyModel = resp.output_parsed
```

## Streaming

Chat Completions streams token deltas appended to message content, you concatenate these deltas yourself to render text. Responses streams typed, semantic events (for example, `response.output_text.delta`), so you can subscribe directly to text updates, handle clear boundaries between items, and avoid manual diffing of message state.

Chat Completions (token deltas appended to content)

```python
from openai import OpenAI
client = OpenAI()

stream = client.chat.completions.create(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "Explain vector databases in 3 points."}],
    stream=True,
)
for chunk in stream:
    delta = getattr(chunk.choices[0].delta, "content", None) or ""
    print(delta, end="")
```

Short example of streaming with Responses:

```python
from openai import OpenAI
client = OpenAI()

with client.responses.stream(
    model="gpt-5-mini",
    input=[{"role": "user", "content": "Explain vector databases in 3 points."}],
) as stream:
    for event in stream:
        if getattr(event, "type", None) == "response.output_text.delta":
            print(event.delta, end="")
```

## Pitfalls and anti‑patterns

- Chat Completions JSON mode ensures JSON, not your schema. Always validate.
- In Responses, avoid replaying full history; pass minimal context or use `previous_response_id`.
- Prefer `.parse(...)` for strict typing instead of parsing strings manually.

## Cost and latency

- Responses can reduce orchestration complexity and retries on multi‑step chains
- Chat Completions can be faster for single‑step prompts where you already control retries and strictness.