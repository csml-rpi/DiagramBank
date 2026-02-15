from dataclasses import dataclass

@dataclass
class Config:
    model_provider: str = "bedrock"
    model_version: str = "arn:aws:bedrock:us-west-2:991404956194:application-inference-profile/f6tueltt82a2"
    temperature: float = 0.6

    # model_provider: str = "gemini"
    # model_version: str = "gemini-2.5-pro"
    # temperature: float = 0.0

    # model_provider: str = "openai"
    # model_version: str = "gpt-5-mini"
    # temperature: float = 0.0

    generation_model: str = "gemini-3-pro-image-preview"
    