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