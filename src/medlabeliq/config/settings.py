from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for Phase 1 scripts."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = Path(__file__).resolve().parents[3]

    dailymed_base_url: str = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
    dailymed_download_base_url: str = "https://dailymed.nlm.nih.gov/dailymed"

    http_timeout_seconds: int = 60
    http_user_agent: str = "MedLabelIQ/0.1 (research project)"

    postgres_db: str = "medlabeliq"
    postgres_user: str = "medlabeliq"
    postgres_password: str = "medlabeliq"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "medlabeliq_chunks"
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"

    reranker_model_name: str = "BAAI/bge-reranker-base"
    reranker_candidate_pool: int = 30

    llm_api_key: str = ""
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.1
    llm_max_output_tokens: int = 1200
    llm_seed: int = 700
    answer_top_k: int = 5

    answer_candidate_pool: int = 15
    evidence_max_per_section: int = 1

    answer_verifier_enabled: bool = True

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def smoke_set_path(self) -> Path:
        return self.project_root / "data" / "seeds" / "smoke_set.yaml"

    @property
    def interim_dir(self) -> Path:
        return self.project_root / "data" / "interim"

    @property
    def raw_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def raw_spl_dir(self) -> Path:
        return self.raw_dir / "spl"

    @property
    def raw_history_dir(self) -> Path:
        return self.raw_dir / "history"

    @property
    def label_candidates_csv_path(self) -> Path:
        return self.interim_dir / "label_candidates.csv"

    @property
    def label_candidates_json_path(self) -> Path:
        return self.interim_dir / "label_candidates.json"

    @property
    def label_history_csv_path(self) -> Path:
        return self.interim_dir / "label_history.csv"

    @property
    def download_manifest_csv_path(self) -> Path:
        return self.interim_dir / "download_manifest.csv"

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


settings = Settings()