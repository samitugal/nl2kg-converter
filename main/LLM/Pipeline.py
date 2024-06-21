from .config_defs import LLMMainConfig, LLMTag
from .LLMBase import LLMBase
from ..output_models import CypherQueryList, TranslateModelOutput

class Pipeline:
    def __init__(self, config: LLMMainConfig, llm: LLMBase):
        self.config = config
        self.llm = llm

    @staticmethod
    def new_instance_from_config(config: LLMMainConfig) -> "Pipeline": 
        from .Bedrock import Bedrock
        from .OpenAI import OpenAI

        match config.llm.llm_tag:
            case LLMTag.BEDROCK:
                return Pipeline(config, Bedrock(config))
            case LLMTag.OPENAI:
                return Pipeline(config, OpenAI(config))
            case _:
                raise ValueError("Invalid LLM tag")

    def generate_kg_query(self, content: str) -> CypherQueryList:
        return self.llm.generate_kg_query(content)

    def translate(self, content: str) -> TranslateModelOutput:
        return self.llm.translate(content)

    def detect_target_node(self, content: str, graphdb_nodes: list) -> str:
        return self.llm.detect_target_node(content, graphdb_nodes)
