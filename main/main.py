from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from .Databases.config_defs import MainConfig
from .LLM.config_defs import LLMMainConfig
from .LLM.Pipeline import Pipeline
from .ContentProvider.content_provider import ContentProvider
from .Databases.GraphDatabase import GraphDatabase

class MainRuntimeVars(BaseSettings):
    GRAPHDATABASE_CONNECTION_PATH: str
    LLM_CONFIG_PATH: str

    class Config:
        env_file = ".env"
        extra = "allow"

app = FastAPI()
envvars = MainRuntimeVars()

load_dotenv()

database_config: MainConfig = MainConfig.from_file(envvars.GRAPHDATABASE_CONNECTION_PATH)
print(f"Using config from {envvars.GRAPHDATABASE_CONNECTION_PATH}")
llm_config: LLMMainConfig = LLMMainConfig.from_file(envvars.LLM_CONFIG_PATH)
print(f"Using config from {envvars.LLM_CONFIG_PATH}")

content = ContentProvider()
llm = Pipeline.new_instance_from_config(config=llm_config)
database = GraphDatabase.new_instance_from_config(config=database_config)

def generate_knowledge_graph():
    response = llm.generate_kg_query(content.all_contexts)
    database = GraphDatabase.new_instance_from_config(config=database_config)
    database.flush_all()
    for query in response.queries:
        print(query)
        database.execute_query(query= query)
    
    return response

def answer_questions():
    DEGREE = 1

    question = content.all_qas[0]["question"]
    answers = content.all_qas[0]["answers"]

    target_node_id = llm.detect_target_node(content= content.all_contexts, graphdb_nodes = database.list_nodes_and_properties()).node_id.split(':')[-1]
    related_nodes = database.list_n_degree_nodes(node_id = target_node_id, degree_count = DEGREE)

    res = llm.QA_Model(question= question, related_nodes= related_nodes)
    while not res.success:
        related_nodes = database.list_n_degree_nodes(node_id = target_node_id, degree_count = DEGREE)
        res = llm.QA_Model(question= question, related_nodes= related_nodes)
    
    return res

answer_questions()