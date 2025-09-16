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
