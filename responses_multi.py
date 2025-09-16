# pip install openai pydantic
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

# Flat tool shape (no nested "function")
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