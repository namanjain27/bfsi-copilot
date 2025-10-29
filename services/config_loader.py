import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from .logger_setup import setup_logger

logger = setup_logger()

class ConfigLoader:
    """
    Configuration loader for EchoPilot application.
    Loads and manages configuration from YAML files with fallback to defaults.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader with optional custom config path

        Args:
            config_path: Optional path to config file. Defaults to config/base.yaml
        """
        if config_path is None:
            # Default to config/base.yaml relative to project root
            # Go up from services/ directory to project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "base.yaml"
            logger.debug(f"Config loader: project_root={project_root}, config_path={config_path}")

        self.config_path = Path(config_path)
        self._config_data = None
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file with error handling"""
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file not found at {self.config_path}. Using defaults.")
                self._config_data = self._get_default_config()
                return

            with open(self.config_path, 'r', encoding='utf-8') as file:
                self._config_data = yaml.safe_load(file)
                logger.info(f"Configuration loaded from {self.config_path}")

        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}. Using defaults.")
            self._config_data = self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}. Using defaults.")
            self._config_data = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file loading fails"""
        return {
            "model": {
                "name": "gemini-2.5-flash",
                "provider": "google_genai"
            },
            "retrieval": {
                "k": 5,
                "search_type": "similarity",
                "threshold": 0.2
            },
            "rag_scoring": {
                "weights": {
                    "semantic": 0.4,
                    "keyword": 0.3,
                    "quality": 0.2,
                    "recency": 0.1
                },
                "tfidf": {
                    "max_features": 1000,
                    "ngram_range": [1, 2],
                    "stop_words": "english"
                },
                "default_threshold": 0.3
            },
            "document_processing": {
                "chunking": {
                    "chunk_size": 1000,
                    "chunk_overlap": 200
                },
                "supported_extensions": [".pdf", ".docx", ".txt", ".md"]
            },
            "chat": {
                "max_retrieval_results": 8,
                "summary": {
                    "max_length_chars": 200
                }
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation

        Args:
            key_path: Dot-separated path to config value (e.g., 'retrieval.k')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        if self._config_data is None:
            return default

        keys = key_path.split('.')
        value = self._config_data

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            logger.warning(f"Config key '{key_path}' not found. Using default: {default}")
            return default

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section

        Args:
            section: Section name (e.g., 'retrieval', 'rag_scoring')

        Returns:
            Configuration section dictionary or empty dict if not found
        """
        return self.get(section, {})

    def reload(self) -> None:
        """Reload configuration from file"""
        self._load_config()
        logger.info("Configuration reloaded")

    def validate_scoring_weights(self) -> bool:
        """
        Validate that RAG scoring weights sum to 1.0

        Returns:
            True if weights are valid, False otherwise
        """
        weights = self.get_section('rag_scoring').get('weights', {})
        total = sum(weights.values())

        if abs(total - 1.0) > 0.01:
            logger.warning(f"RAG scoring weights sum to {total}, not 1.0. Consider normalizing.")
            return False
        return True

# Global config instance
_config_instance = None

def get_config() -> ConfigLoader:
    """
    Get global configuration instance (singleton pattern)

    Returns:
        ConfigLoader instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader()
    return _config_instance

def reload_config() -> None:
    """Reload global configuration"""
    global _config_instance
    if _config_instance is not None:
        _config_instance.reload()