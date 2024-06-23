from .output_models import TranslateModelOutput, CypherQueryList, NodeDetectionModelOutput, QAModelOutput, ValidationModelOutput
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

class QAModelOutputParser:
    def __init__(self):
        self.qa_model = PydanticOutputParser(pydantic_object = QAModelOutput)

class ValidationModelOutputParser:
    def __init__(self):
        self.validate_model = PydanticOutputParser(pydantic_object = ValidationModelOutput)