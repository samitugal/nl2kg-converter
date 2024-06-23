import string
from typing import TypeVar, List, Dict, Any
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate

from .LLMAbstractBase import LLMAbstractBase
from ..output_models import *
from ..output_parsers import *

U = TypeVar("U", bound=BaseModel)

class LLMBase(LLMAbstractBase):
    def __init__(self, config):
        self.config = config
        load_dotenv()

    def _create_chain(self, template: str, input_variables: List[str], partial_variables: Dict[str, Any], parser):
        prompt_template = PromptTemplate(
            input_variables=input_variables, 
            template=template, 
            partial_variables=partial_variables
        )
        return prompt_template | self.client | parser

    def translate(self, content: str) -> TranslateModelOutput:
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
        chain = self._create_chain(translate_template, ["content"], {"format_instructions": output_parser.get_format_instructions()}, output_parser)
        response: TranslateModelOutputParser = chain.invoke(input={"content": content})
        return response 

    def generate_kg_query(self, content: str) -> CypherQueryList:
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
                        "CREATE (childrensMemorialHealthInstitute:Hospital [name: 'Children's Memorial Health Institute', recognition: 'highest-reference hospital in all of Poland'])"
                        <Description>
                            This example is wrong because when you seperate Levi's there is a syntax error occured.
                            Or 'Children's Memorial Health Institute' is not a valid entity name because there are three single quotes.
                            It causes error in syntax. Be careful about punctuation. 
                        </Description>
                    </Example>
                </Note>
                <Note>,
                    While you generating about relations between entities. Use entity properties and type to generate precise relations.
                    Do not skip it with like CREATE (broncos)-[:DEFEATED]->(panthers). Generate queries like
                    MATCH (broncos:Team [name: 'Denver Broncos']),(panthers:Teams [name: 'Carolina Panthers']) CREATE (broncos)-[:DEFEATED]->(panthers)
                    Make relation generation as precise as possible. Use all properties and types to generate relation.
                    Because If any missunderstanding in queries cost us millions of unnecessary relations.
                </Note>
                <Note>
                    You can generate new entities for classify the content.
                    <Example>
                        If the content is related about singers, you can generate an entity named 'Singers' and generate relatons with it.
                    </Example>
                </Note>
                <Note>
                    Do not use multiple generation in one query. Generate this queries one by one.
                </Note>
                <Note>
                    Be sure property value definitions must be between double quotes.
                </Note>
                <Note>
                    Generate at least one relation for each entity.
                </Note>
                <Note>
                    Detail the properties of the nodes as much as possible.
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
        chain = self._create_chain(knowledge_graph_template, ["content"], {"format_instructions": output_parser.get_format_instructions()}, output_parser)
        response: QueryGenerationOutputParser = chain.invoke(input={"content": self.translate(content).translated_content})
        return self._clean_response(response)

    def _clean_response(self, response_list: CypherQueryList):
        translator = str.maketrans('', '', string.punctuation)
        for query in response_list.queries:
            query = query.translate(translator)
        return response_list

    def detect_target_node(self, content: str, graphdb_nodes: List[Dict[str, Any]]) -> NodeDetectionModelOutput:
        target_node_template = """
        <InstructionStructure>
            <PrimaryTask>
                The main goal is to identify the node related to the question asked by the user within the <Content> tag. 
                For this, you can find a dictionary containing node information within the <Context> tag. 
                Return the id information of the related node as the answer.
            </PrimaryTask>
            <Content>
                Content: {content}
            </Content>
            <Context>
                {context}
            </Context>
            <Output>
                <<OUTPUT (must include ```json at the start of the response)>>
                <<OUTPUT (must end with ```)>>
            </Output>
            <FormatInstructions>
                {format_instructions}
            </FormatInstructions>
        </InstructionStructure>
        """
        output_parser = NodeDetectionOutputParser().node_detection_parser
        chain = self._create_chain(target_node_template, ["content", "context"], {"format_instructions": output_parser.get_format_instructions()}, output_parser)
        response: NodeDetectionOutputParser = chain.invoke(input={"content": content, "context": graphdb_nodes})
        return response

    def QA_Model(self, question: str, related_nodes: List[Dict[str, Any]]) -> QAModelOutput:
        qa_model_template = """
        <InstructionStructure>
            <PrimaryTask>
                The main goal is to provide the answer to the question asked by the user within the <Content> tag. 
                For this, you can find other nodes related to the user's question within the <Context> tag. 
                If the provided node information is not sufficient, return false for the related boolean field in the output. 
                If you are not sure about the answer, there is no need to provide an answer. Just return the mandatory field in the output.
            </PrimaryTask>
            <Content>
                Content: {content}
            </Content>
            <Context>
                {context}
            </Context>
            <Output>
                <<OUTPUT (must include ```json at the start of the response)>>
                <<OUTPUT (must end with ```)>>
            </Output>
            <FormatInstructions>
                {format_instructions}
            </FormatInstructions>
        </InstructionStructure>
        """
        output_parser = QAModelOutputParser().qa_model
        chain = self._create_chain(qa_model_template, ["content", "context"], {"format_instructions": output_parser.get_format_instructions()}, output_parser)
        response: QAModelOutputParser = chain.invoke(input={"content": question, "context": related_nodes})
        return response

    def validate_answer(self, answers: list[str], model_answer: str) -> ValidationModelOutput:
        validation_model_template = """
        <InstructionStructure>
            <PrimaryTask>
                The main objective is to compare the output of an LLM model with the expected results. You can find the result from the model within the <ModelResponse> tag 
                and the list of expected results within the <ExpectedResponses> tag. Check if the semantic accuracy is achieved. Produce the output in the format I provided.
            </PrimaryTask>
            <ModelResponse>
                ModelResponse: {ModelResponse}
            </ModelResponse>
            <ExpectedResponses>
                {ExpectedResponses}
            </ExpectedResponses>
            <Output>
                <<OUTPUT (must include ```json at the start of the response)>>
                <<OUTPUT (must end with ```)>>
            </Output>
            <FormatInstructions>
                {format_instructions}
            </FormatInstructions>
        </InstructionStructure>
        """
        output_parser = ValidationModelOutputParser().validate_model
        chain = self._create_chain(validation_model_template, ["ModelResponse", "ExpectedResponses"], {"format_instructions": output_parser.get_format_instructions()}, output_parser)
        response: ValidationModelOutputParser = chain.invoke(input={"ModelResponse": model_answer, "ExpectedResponses": answers})
        return response