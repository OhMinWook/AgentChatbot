from pydantic import BaseModel, Field

class SummarizeOptions(BaseModel):
    format: str = Field(default="bullets", pattern="^(bullets|text)$")
    max_bullets: int = Field(default=8, ge=1, le=30)
    max_tokens: int | None = Field(default=None, ge=16, le=4096)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)

class SummarizeRequest(BaseModel):
    text: str

class SummarizeResponse(BaseModel):
    summary: str
    model: str
    prompt_version: str