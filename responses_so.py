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