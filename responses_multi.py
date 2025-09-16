# pip install openai pydantic
from openai import OpenAI
from pydantic import BaseModel
client = OpenAI()

# --- App tool (your backend) ---
def get_weather(city: str) -> dict:
    # Pretend this hits your weather service
    return {"city": city, "temp_c": 17, "condition": "rain"}

# --- JSON contract for the second turn ---
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

# --- Turn 1: user asks weather; model may call the tool ---
resp1 = client.responses.create(
    model="gpt-4o-mini",
    input=[{"role": "user", "content": "What's the weather in Paris today?"}],
    tools=tools,
)

# Extract tool call (if any), run it
tool_calls = [o for o in resp1.output if getattr(o, "type", None) == "tool_call"]
if tool_calls:
    tc = tool_calls[0].tool_call
    result = get_weather(**tc.arguments)

    # --- Turn 2: follow-up requires STRICT JSON per our schema
    # NOTE: Using .parse() enforces schema and returns a typed object.
    resp2 = client.responses.parse(
        model="gpt-4o-2024-08-06",
        input=[
            # Only stitch in what's relevant, not the entire transcript.
            {"role": "tool", "name": "get_weather", "tool_call_id": tc.id, "content": str(result)},
            {"role": "user", "content": "Great, should I pack an umbrella? Return JSON only."}
        ],
        text_format=PackAdvice,   # <- strict schema enforcement
        # (Under the hood, this uses response_format=json_schema+strict)
    )

    advice: PackAdvice = resp2.output_parsed
    print(advice.model_dump())   # {'umbrella': True/False, 'rationale': '...'}
else:
    print(resp1.output_text)

# - No ever-growing `messages[]` replay. We pass only the tool result + the new user turn.
# - .parse() guarantees a schema-valid JSON object (no brittle post-processing).
# - Easy to scale to more tools, images, or other content parts without reshaping everything as chat messages.