import logging
from typing import Any

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Central place for registering and fetching ML models."""

    def __init__(self) -> None:
        self._registry: dict[str, type] = {}
        self._instances: dict[str, Any] = {}

    def register(self, name: str, model_cls: type) -> None:
        if name in self._registry:
            logger.warning(f"Overwriting existing model registration: {name}")
        self._registry[name] = model_cls

    def get(self, name: str) -> Any:
        if name in self._instances:
            return self._instances[name]

        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(f"Model '{name}' not found in registry. Available: {available}")

        # lazy init
        logger.info(f"Instantiating model: {name}")
        cls = self._registry[name]
        instance = cls()
        instance.load()
        self._instances[name] = instance
        return instance

    def list_models(self) -> list[str]:
        return list(self._registry.keys())

    def is_loaded(self, name: str) -> bool:
        return name in self._instances


# global registry — import this, not the class
registry = ModelRegistry()
