import boto3
from langchain_community.llms import Bedrock as ChatBedrock
from .LLMBase import LLMBase
from .config_defs import LLMTag, LLMMainConfig

class Bedrock(LLMBase):
    def __init__(self, config):
        super().__init__(config)
        if config.llm.llm_tag != LLMTag.BEDROCK:
            raise ValueError("BedrockPipeline can only be used with Bedrock")
        if config.bedrock is None:
            raise ValueError("BedrockPipeline requires a BedrockConfig")

        bedrock = boto3.client(service_name='bedrock-runtime', region_name=config.bedrock.region_name)
        self.client = ChatBedrock(model_id=config.bedrock.model_id, client=bedrock, model_kwargs={"temperature": config.llm.temperature})
