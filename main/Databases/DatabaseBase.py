from abc import ABC, abstractmethod
from .config_defs import MainConfig, GDBTag

class DatabaseBase(ABC):
    def __init__(self, config: MainConfig):
        self.config = config
    
    @abstractmethod
    def execute_query(self, execute_query: str):
        pass
    
    @abstractmethod
    def disconnect(self):
        pass