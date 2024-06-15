import html
import re
import json

from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.globals import set_debug
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError
from typing import TypeVar

from ..output_models import CypherQueryList, TranslateModelOutput
from .config_defs import LLMMainConfig, LLMTag
from .LLMBase import LLMBase
from ..output_parsers import TranslateModelOutputParser, QueryGenerationOutputParser

U = TypeVar("U", bound=BaseModel)

class OpenAI(LLMBase):
    def __init__(self, config: LLMMainConfig):
        if(config.llm.llm_tag != LLMTag.OPENAI):
            raise ValueError("OpenAIPipeline can only be used with OpenAI")
        if config.openai is None:
            raise ValueError("OpenAIPipeline requires a OpenAIConfig")
        
        load_dotenv()
        self.config = config
        
        if config.openai.json_mode:
            model_kwargs = {"response_format": {"type": "json_object"}}
        else:
            model_kwargs = {}
        self.client = ChatOpenAI(model=config.openai.model_name, temperature=config.llm.temperature, model_kwargs=model_kwargs)


    def translate(self, content: str) -> TranslateModelOutput:
        """
            Translation Step For Content.
        """

        translate_template = """
        <InstructionStructure>
            <PrimaryTask>
                Translate the following text to English I will provide you in <Content> tag. 
            </PrimaryTask>
            <Content>
                Content: {content}
            </Content>
            <Output>
                <<OUTPUT (must include ```json at the start of the response)>>
                <<OUTPUT (must end with ```)>>
            </Output>
            <FormatInstructions>
                {format_instructions}
            </FormatInstructions>
        </InstructionStructure>
        """
        
        output_parser = TranslateModelOutputParser().translate_parser
        translate_template = PromptTemplate(input_variables=["content"], 
                                            template=translate_template, 
                                            partial_variables={"format_instructions": output_parser.get_format_instructions() })
        chain = translate_template | self.client | output_parser
        response: TranslateModelOutputParser = chain.invoke(input={"content": content})

        return response 

    def generate_kg_query(self, content: str) -> CypherQueryList:
        """
            Knowledge Graph Generation Step
        """

        knowledge_graph_template = """
        <InstructionStructure>
            <PrimaryTask>
                You are an ontology specialist. Your primary goal is to analyze the provided text and identify objects, their attributes, 
                and the relationships between these objects. You must output this information in the specified format and also generate 
                Cypher queries to create these objects, attributes, and relationships in a graph database.
            </PrimaryTask>
            <Content>
                Content: {content}
            </Content>
            <Output>
                <<OUTPUT (must include ```json at the start of the response)>>
                <<OUTPUT (must end with ```)>>
            </Output>
            <FormatInstructions>
                {format_instructions}
            </FormatInstructions>
        </InstructionStructure>
        """

        output_parser = QueryGenerationOutputParser().query_generator_parser
        knowledge_graph_template = PromptTemplate(input_variables=["content"], 
                                            template=knowledge_graph_template, 
                                            partial_variables={"format_instructions": output_parser.get_format_instructions() })
        chain = knowledge_graph_template | self.client | output_parser
        response: QueryGenerationOutputParser = chain.invoke(input={"content": content})

        return response



    