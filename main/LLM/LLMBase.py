from abc import ABC, abstractmethod
from ..output_models import CypherQueryList

class LLMBase(ABC):

    @abstractmethod
    def translate(self, request: str) -> str:
        pass

    @abstractmethod
    def generate_kg_query(self, content: str) -> CypherQueryList:
        ...

    