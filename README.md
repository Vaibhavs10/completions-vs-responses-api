# OpenAI Chat Completions vs Responses API

A brief write-up about Chat Completions vs Responses API.

TL;DR
- Responses: recommended for new projects, agentic by default, multi‑step flows with optional server‑managed state across turns and evented streaming.
- Chat Completions: ideal for simple, stateless prompts and existing CC integrations; you manage conversation state and validate schemas yourself. # Note: "and validate schemas yourself." => not really true

### Quick chooser

| Scenario | Prefer |
| --- | --- |
| Stateless, single‑turn chat or existing CC integration | Chat Completions |
| Multi‑turn with state preserved server‑side | Responses |
| Agent‑like multi‑step orchestration | Responses |
| Remote tool execution (MCP server, web search, file search, etc.) | Responses |
| Fine‑grained manual control over history/retries | Chat Completions |
| Longer chains with internal iteration | Responses |

### Minimal side‑by‑side

**NOTE:** not sure example is relevant anymore since almost the same

```python
# Chat Completions
resp = client.chat.completions.create(
    model="gpt-5-mini",
    messages=messages,
    response_format=Item
)
item = resp.choices[0].message.parsed

# Responses
resp = client.responses.parse(
    model="gpt-5-mini",
    input=messages,
    text_format=Item,
)
item: Item = resp.output_parsed
```

## Chat Completions API

Build and replay `messages[]` every turn, call `/chat/completions`, and validate strict JSON against your schema using `response_format` plus your own parsing. # **TODO:** adapt comment?



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
    response_format=RepoSummary,
)

print(completion.choices[0].message.parsed)
```

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
# - JSON mode guarantees JSON, but you still validate your schema. # TODO

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

**TODO:** this snippet feels extremely complicated for something that is supposed to be "easier".
          check https://platform.openai.com/docs/guides/conversation-state?api-mode=responses#passing-context-from-the-previous-response

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

# Turn 1: tool call
resp1 = client.responses.create(
    model="gpt-5-mini",
    input={"role": "user", "content": "What's the weather in Paris today?"},
    tools=tools,
)

# Turn 2: should I take my umbrella?
resp2 = client.responses.create(
    model="gpt-5-mini",
    input={"role": "user", "content": "Great, should I pack an umbrella?"},
    text_format=PackAdvice,
    previous_response_id=resp1.id,
)

advice: PackAdvice = resp2.output_parsed
print(advice)
```

Key takeaways

- Orchestration: CC requires replaying the entire `messages[]` each turn; Responses can keep state on the server and lets you pass only what’s needed. # only when using `previous_response_id` (or conversation) but not by default
- Schema handling: Both support schema‑constrained outputs; CC uses JSON schema modes with manual validation; Responses adds an SDK parse helper that returns typed objects. # TODO: update
- MCP: Responses supports remote MCP servers. # TODO: maybe an MCP example with ResponsesAPI ? remote execution is the key feature in favor of Responses API IMO
- Streaming: CC streams token deltas appended to `message.content`. Responses emits semantic events (e.g. `response.output_text.delta`).
- Storage: Responses are stored by default; Chat Completions are not stored by default. Set `store=False` to disable.
- Error handling: CC flows often need retry logic for malformed JSON or schema drift. Responses 
reduces drift via strict schema and internal iteration. # is this really correct?
- Scalability: CC mixes tools and strict output with more prompt guards and state management. 
Responses scales to longer, multi‑modal chains without an ever‑growing transcript. # **NOTE:** not sure I understand this point

## Migration: Chat Completions → Responses

- General calls: replace `client.chat.completions.create(...)` with `client.responses.create(...)`.
- Typed outputs: replace JSON mode + manual validation with `client.responses.parse(..., text_format=YourModel)`.
- State: stop replaying the entire `messages[]`; simply pass `previous_response_id` to continue a conversation.

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
    # TODO: requires previous_message_id
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
        if event.type == "response.output_text.done": # NOTE: I think events in a stream always have a `.type` which simplifies the logic
            print(event.text) # **NOTE:** the advantage of ResponsesAPI is precisely that you don't need to listen to deltas one by one
```

## Pitfalls and anti‑patterns

**NOTE:** this section feels redundant with what's above

- Chat Completions JSON mode ensures JSON, not your schema. Always validate. # **TODO:** really not sure about that. You can ask for "json_object" (returns a JSON) or "json_schema" (return a JSON following schema)
- In Responses, avoid replaying full history; pass minimal context or use `previous_response_id`.
- Prefer `.parse(...)` for strict typing instead of parsing strings manually.

## Cost and latency

**NOTE:** this section feels a bit artificial

- Responses can reduce orchestration complexity and retries on multi‑step chains
- Chat Completions can be faster for single‑step prompts where you already control retries and strictness.
