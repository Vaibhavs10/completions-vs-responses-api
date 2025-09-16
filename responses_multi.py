# pip install openai pydantic
import json
from openai import OpenAI
from pydantic import BaseModel
client = OpenAI()

def get_weather(city: str) -> dict:
    return {"city": city, "temp_c": 17, "condition": "rain"}

class PackAdvice(BaseModel):
    umbrella: bool
    rationale: str

# Tools for Responses = flat shape (no nested "function")
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

# --- Turn 1: user asks; model may call the tool ---
resp1 = client.responses.create(
    model="gpt-5-mini",  # use a Responses-capable model
    input=[{"role": "user", "content": "What's the weather in Paris today?"}],
    tools=tools,
)

# Extract the first function/tool call (if any)
func_calls = [o for o in getattr(resp1, "output", []) if getattr(o, "type", None) in ("function_call", "tool_call")]

if not func_calls:
    # Model answered directly without using the tool
    print(resp1.output_text)  # convenience accessor for assistant text
else:
    fc = func_calls[0]  # has .name, .arguments, .call_id (and sometimes .id)
    # Arguments can be a dict or a JSON string depending on SDK/model
    fc_args = getattr(fc, "arguments", {})
    args = fc_args if isinstance(fc_args, dict) else json.loads(fc_args)
    weather = get_weather(**args)
    # Prefer call_id if present; fallback to id
    tool_call_id = getattr(fc, "call_id", None) or getattr(fc, "id", None)

    # --- Turn 2: require STRICT JSON conforming to PackAdvice ---
    # Using .parse(...) enforces the schema and returns a typed object
    resp2 = client.responses.parse(
        model="gpt-5-mini",
        input=[
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "get_weather result (tool_call_id=" + str(tool_call_id) + "): " + json.dumps(weather)
                    }
                ]
            },
            {"role": "user", "content": "Great, should I pack an umbrella? Return JSON only."}
        ],
        text_format=PackAdvice,  # schema-enforced structured output
    )

    advice: PackAdvice = resp2.output_parsed
    print(advice.model_dump())


# Notes:
# - No ever-growing `messages[]`; we pass just the tool result + the new user turn.
# - `.parse(..., text_format=PackAdvice)` guarantees the exact schema (no manual JSON parsing/validation).
# - Easy to scale to more tools, images, or other content parts without reshaping everything as chat messages.