from .config_defs import GDBTag, MainConfig
from .DatabaseBase import DatabaseBase

from typing import List, Dict, Any

class GraphDatabase(DatabaseBase):
    def __init__(self, config: MainConfig, database: DatabaseBase):
        super().__init__(config)
        self.database = database

    @staticmethod
    def new_instance_from_config(config: MainConfig) -> "GraphDatabase": 
        from .Neo4j import Neo4jDatabase

        match config.gdb.tag:
            case GDBTag.NEO4J:
                return GraphDatabase(config, Neo4jDatabase(config))
            case _:
                raise ValueError("Invalid LLM tag")

    def execute_query(self, query: str):
        return self.database.execute_query(query)

    def list_nodes_and_properties(self) -> List[Dict[str, Any]]:
        return self.database.list_nodes_and_properties()
  
    def disconnect(self):
        self.database.disconnect()

    def flush_all(self):
        self.database.flush_all()