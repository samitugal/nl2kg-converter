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
#response = llm.generate_kg_query(content.all_contexts)
database = GraphDatabase.new_instance_from_config(config=database_config)
target_node = llm.detect_target_node(content= content.all_qas[0]["question"], graphdb_nodes= database.list_nodes_and_properties())

print(content.all_qas[0]["question"])
print("")
print(database.list_nodes_and_properties())
print("")
print(target_node)
database.disconnect()