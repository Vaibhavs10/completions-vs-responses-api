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

Key takeaways:

- Orchestration: CC requires you to build and replay the entire `messages[]` every turn; Responses passes only the minimal context (tool result + follow‑up prompt).
- Schema enforcement: CC JSON mode ensures valid JSON but not your exact schema, so you must validate and handle errors yourself. Responses `.parse()` enforces the Pydantic schema and returns a typed object.
- Error handling: CC flows often need retry logic for malformed JSON or schema drift. Responses reduces drift via strict schema and internal iteration.
- Scalability: CC mixes tools and strict output with more prompt guards and state management. Responses scales to longer, multi‑modal chains without an ever‑growing transcript.

