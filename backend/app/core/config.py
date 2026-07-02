from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    project_name: str = "OCTera"
    app_version: str = "0.1.0"
    api_v1_prefix: str = "/api/v1"

    environment: str = "development"

    database_url: str = "sqlite:///./octera.db"

    secret_key: str = "change-me-in-production"
    access_token_expire_minutes: int = 60 * 24

    storage_dir: str = "storage/studies"
    max_upload_size_mb: int = 20

    cors_origins: list[str] = ["http://localhost:3000"]

    login_rate_limit_attempts: int = 5
    login_rate_limit_window_seconds: int = 60


settings = Settings()

INSECURE_DEFAULT_SECRET_KEY = "change-me-in-production"


def assert_production_config_is_safe() -> None:
    if settings.environment == "production" and settings.secret_key == INSECURE_DEFAULT_SECRET_KEY:
        raise RuntimeError(
            "SECRET_KEY is still the insecure default. Set a real SECRET_KEY before running with "
            "ENVIRONMENT=production."
        )
