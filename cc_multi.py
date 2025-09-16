# pip install openai pydantic
import json
from openai import OpenAI
from pydantic import BaseModel, ValidationError
client = OpenAI()

def get_weather(city: str) -> dict:
    return {"city": city, "temp_c": 17, "condition": "rain"}

class PackAdvice(BaseModel):
    umbrella: bool
    rationale: str

messages = [
    {"role": "user", "content": "What's the weather in Paris today?"}
]

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

# --- Turn 1: model may request the tool ---
resp1 = client.chat.completions.create(
    model="gpt-5-mini",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)
assistant_msg = resp1.choices[0].message
messages.append(assistant_msg)

if assistant_msg.tool_calls:
    call = assistant_msg.tool_calls[0]
    # `arguments` is a JSON string; parse before unpacking
    args = json.loads(call.function.arguments)
    result = get_weather(**args)

    # You must append the tool result and replay the entire history:
    messages.append({
        "role": "tool",
        "tool_call_id": call.id,
        "name": "get_weather",
        "content": str(result)
    })

# --- Turn 2: ask for JSON. JSON mode ensures valid JSON, NOT your schema. ---
messages.append({"role": "user", "content": "Great, should I pack an umbrella? Return JSON only."})

resp2 = client.chat.completions.create(
    model="gpt-5-mini",
    messages=messages,  # ISSUE: must replay all history (grows every turn)
    response_format={"type": "json_object"},  # Guarantees JSON, not schema
)

raw = resp2.choices[0].message.content

# You must still validate the shape yourself (keys/types).
try:
    data = json.loads(raw)
    advice = PackAdvice.model_validate(data)  # may raise if keys/types drift
    print(advice.model_dump())
except (json.JSONDecodeError, ValidationError) as e:
    # ISSUE: You handle recoveries/retries because JSON mode doesn't enforce your schema.
    print("Invalid/Unexpected JSON; need to reprompt or retry:", e)

# - You manage an ever-growing `messages[]` and must remember to replay it each turn.
# - JSON mode won't enforce your exact schema; you must validate + handle errors/retries.
# - Mixing tools + strict output typically requires extra prompting/guards to avoid drift.