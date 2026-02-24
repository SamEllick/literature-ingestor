from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LM Studio
    lms_base_url: str = "http://localhost:1234/v1"
    lms_api_key: str = "lm-studio"
    lms_chat_model: str = "local-model"
    lms_embed_model: str = "local-model"
    lms_embed_dim: int = 768

    # ChromaDB (embedded, no server required)
    chroma_path: str = "./data/chroma"
    chroma_collection: str = "literature"

    # Marker PDF parser device ("cpu" or "cuda")
    marker_device: str = "cpu"

    # Storage
    metadata_db_path: str = "./data/metadata.db"
    papers_dir: str = "./data/papers"


settings = Settings()
