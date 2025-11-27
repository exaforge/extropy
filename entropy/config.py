"""Configuration management for Entropy."""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI
    openai_api_key: str = ""

    # LM Studio (Phase 3)
    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_model: str = "llama-3.2-3b"

    # Database
    db_path: str = "./storage/entropy.db"

    # Defaults
    default_population_size: int = 1000

    @property
    def db_path_resolved(self) -> Path:
        """Resolve database path."""
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def cache_dir(self) -> Path:
        """Cache directory for research results."""
        path = Path("./data/cache")
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

