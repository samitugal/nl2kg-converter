from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

class CypherQueryList(BaseModel):
    queries: list[str] = Field(default_factory=list)

class TranslateModelOutput(BaseModel):
    translated_content: str = Field(description= "translation of content")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "translated_content": self.translated_content,
        }

class NodeDetectionModelOutput(BaseModel):
    node_id: str = Field(description= "Node id information which related with question")

class QAModelOutput(BaseModel):
    success: bool = Field(description="Indicates if the model can provide an answer")
    answer: Optional[str] = Field(None, description="The model's answer to the question, if available")

class ValidationModelOutput(BaseModel):
    result: bool = Field("Is model response is  true or false indicator")

@dataclass
class Answer:
    answer_start: int
    text: str

@dataclass
class QuestionAnswer:
    answers: List[Answer]
    question: str
    id: str

@dataclass
class Paragraph:
    context: str
    qas: List[QuestionAnswer]

@dataclass
class Article:
    title: str
    paragraphs: List[Paragraph]

