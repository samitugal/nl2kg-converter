import boto3
from typing import TypeVar
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
from langchain_community.llms import Bedrock
from langchain_aws import ChatBedrock
from langchain.prompts.prompt import PromptTemplate

from ..output_models import CypherQueryList, TranslateModelOutput
from .config_defs import LLMMainConfig, LLMTag
from .LLMBase import LLMBase
from ..output_parsers import TranslateModelOutputParser, QueryGenerationOutputParser

U = TypeVar("U", bound=BaseModel)

class Bedrock(LLMBase):
    def __init__(self, config: LLMMainConfig):
        if config.llm.llm_tag != LLMTag.BEDROCK:
            raise ValueError("BedrockPipeline can only be used with Bedrock")
        if config.bedrock is None:
            raise ValueError("BedrockPipeline requires a BedrockConfig")

        self.config = config
        
        load_dotenv()

        bedrock = boto3.client(service_name='bedrock-runtime', region_name=config.bedrock.region_name)
        self.client = ChatBedrock(model_id=config.bedrock.model_id, client=bedrock, model_kwargs={"temperature": config.llm.temperature})


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
                Obey the output schema, Do not change field names.
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
            <Notes>
                <Note>
                    This genereated queries will executed in Neo4j graph database so you must be careful about syntax and punctuation.
                    <Example>
                        "CREATE (stadium:Stadium name: 'Levi's Stadium', location: 'Santa Clara, California')"
                        <Description>
                            This example is wrong because when you seperate Levi's there is a syntax error occured.
                            Be careful about punctuation. 
                        </Description>
                    </Example>
                </Note>
                <Note>,
                    While you generating about relations between entities. Use entity properties and type to generate precise relations.
                    Do not skip it with like CREATE (broncos)-[:DEFEATED]->(panthers). Generate queries like
                    MATCH (broncos:Team [name: 'Denver Broncos']),(panthers:Teams [name: 'Carolina Panthers']) CREATE (broncos)-[:DEFEATED]->(panthers)
                </Note>
            </Notes>
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
        print(response)
        return response

    