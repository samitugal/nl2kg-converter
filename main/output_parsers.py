from .output_models import TranslateModelOutput, CypherQueryList, NodeDetectionModelOutput
from langchain.output_parsers import PydanticOutputParser

class TranslateModelOutputParser:
    def __init__(self):
        self.translate_parser = PydanticOutputParser(pydantic_object= TranslateModelOutput)

class QueryGenerationOutputParser:
    def __init__(self):
        self.query_generator_parser = PydanticOutputParser(pydantic_object= CypherQueryList)

class NodeDetectionOutputParser: 
    def __init__(self):
        self.node_detection_parser = PydanticOutputParser(pydantic_object = NodeDetectionModelOutput)