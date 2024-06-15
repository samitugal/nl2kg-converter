from .config_defs import GDBTag, MainConfig
from .DatabaseBase import DatabaseBase

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
  
    def disconnect(self):
        self.database.disconnect()