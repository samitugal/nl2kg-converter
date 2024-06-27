import warnings
import boto3

from langchain_community.chat_models import BedrockChat
from .LLMBase import LLMBase
from .config_defs import LLMTag, LLMMainConfig
from .callbacks import LLMCallbacks

from main.output_models import TranslateModelOutput

warnings.filterwarnings("ignore", category=DeprecationWarning, module='langchain')

class Bedrock(LLMBase):
    def __init__(self, config):
        super().__init__(config)
        if config.llm.llm_tag != LLMTag.BEDROCK:
            raise ValueError("BedrockPipeline can only be used with Bedrock")
        if config.bedrock is None:
            raise ValueError("BedrockPipeline requires a BedrockConfig")

        bedrock = boto3.client(service_name='bedrock-runtime', region_name=config.bedrock.region_name)
        self.client = BedrockChat(model_id=config.bedrock.model_id, client=bedrock, model_kwargs={"temperature": config.llm.temperature}, callbacks = [LLMCallbacks()])
        self.translator = boto3.client(service_name='translate', region_name=config.bedrock.region_name, use_ssl=True)

    def translate(self, content: str) -> TranslateModelOutput:
        try:
            response = self.translator.translate_text(Text=content, SourceLanguageCode="auto", TargetLanguageCode="en")
            return TranslateModelOutput(translated_content=response['TranslatedText'])
        except Exception as e:
            return super().translate(content)
