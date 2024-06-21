import warnings

from .config_defs import MainConfig
from .DatabaseBase import DatabaseBase

from neo4j import GraphDatabase
from typing import Any, List, Dict
from dotenv import load_dotenv

from typing import TypeAlias

warnings.filterwarnings("ignore", category=DeprecationWarning)

QueryOutputDict: TypeAlias = List[Dict[str, Any]]

class Neo4jDatabase(DatabaseBase):
    instance = None

    def __new__(cls, *args, **kwargs):
        if not cls.instance:
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self, config: MainConfig):
        if not hasattr(self, 'initialized'):  # Avoid re-initializing the singleton
            super().__init__(config)
            load_dotenv()
            AUTH = (config.neo4j.user, config.neo4j.password)
            with GraphDatabase.driver(config.neo4j.uri, auth=AUTH) as driver:
                self.driver = driver
                driver.verify_connectivity()
            self.initialized = True

    def execute_query(self, query: str):
        """Executes Query for Neo4j"""
        with self.driver.session() as session:
            session.write_transaction(self._execute_query, query)

    @staticmethod
    def _execute_query(tx, query: str):
        result = tx.run(query)
        return result

    def flush_all(self):
        with self.driver.session() as session:
            session.write_transaction(self._flush_all)

    @staticmethod
    def _flush_all(tx):
        tx.run("MATCH (n) DETACH DELETE n")

    def list_nodes_and_properties(self) -> QueryOutputDict:
        nodes_list = []
        with self.driver.session() as session:
            result = session.run("MATCH (n) RETURN n, labels(n) AS labels, keys(n) AS keys, properties(n) AS properties")
            for record in result:
                node = {
                    "id": record["n"].element_id,
                    "labels": record["labels"],
                    "keys": record["keys"],
                    "properties": record["properties"]
                }
                nodes_list.append(node)
        self.disconnect()
        return nodes_list

    def list_n_degree_nodes(self, node_id: str, degree_count: int) -> QueryOutputDict:
        nodes_list = []
        with self.driver.session() as session:
            result = session.run(f"""
            MATCH (n)-[*{degree_count}]-(m)
            WHERE id(n) = {node_id}
            RETURN m
            """)
            
            for record in result:
                node = record["m"]
                node_data = {
                    "id": node.element_id,
                    "labels": list(node.labels),
                    "properties": dict(node)
                }
                nodes_list.append(node_data)
        
        self.disconnect()
        return nodes_list

    def disconnect(self) -> None:
        if self.driver is not None:
            self.driver.close()

if __name__ == "__main__":
    config = MainConfig.from_file("configs/GraphDatabase/neo4j.yaml")
    db = Neo4jDatabase(config)
    db.disconnect()
