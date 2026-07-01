from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    project_name: str = "OCTera"
    api_v1_prefix: str = "/api/v1"

    database_url: str = "sqlite:///./octera.db"

    secret_key: str = "change-me-in-production"
    access_token_expire_minutes: int = 60 * 24

    storage_dir: str = "storage/studies"

    cors_origins: list[str] = ["http://localhost:3000"]


settings = Settings()
