from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    API_KEY: str

    class Config:
        env_file = ".env"


settings = Settings()

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False
)


def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Verify the API Key sent in the request header.
    """

    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API Key."
        )

    return api_key