from .config_defs import LLMMainConfig, LLMTag
from .LLMBase import LLMBase
from ..output_models import *

from typing import List, Dict, Any

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

    def detect_target_node(self, content: str, graphdb_nodes: list) -> NodeDetectionModelOutput:
        return self.llm.detect_target_node(content, graphdb_nodes)

    def QA_Model(self, question: str, related_nodes: List[Dict[str, Any]]) -> QAModelOutput:
        return self.llm.QA_Model(question, related_nodes)

    def validate_answer(self, answers: list[str], model_answer: str) -> ValidationModelOutput:
        return self.llm.validate_answer(answers, model_answer)
