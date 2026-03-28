from abc import ABC, abstractmethod


class BaseMemory(ABC):
    @abstractmethod
    def save(self, user_id: str, messages: list[dict]) -> None:
        """Save messages to memory after each turn."""
        ...

    @abstractmethod
    def retrieve(self, user_id: str, query: str) -> str:
        """Retrieve relevant memory as a string to inject into system prompt."""
        ...


class DummyMemory(BaseMemory):
    """No-op memory for testing the agent skeleton."""

    def save(self, user_id: str, messages: list[dict]) -> None:
        pass

    def retrieve(self, user_id: str, query: str) -> str:
        return ""
