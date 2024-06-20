import warnings

from langchain_openai import ChatOpenAI
from .LLMBase import LLMBase
from .config_defs import LLMTag, LLMMainConfig
from .callbacks import LLMCallbacks

warnings.filterwarnings("ignore", category=DeprecationWarning, module='langchain')

class OpenAI(LLMBase):
    def __init__(self, config):
        super().__init__(config)
        if config.llm.llm_tag != LLMTag.OPENAI:
            raise ValueError("OpenAIPipeline can only be used with OpenAI")
        if config.openai is None:
            raise ValueError("OpenAIPipeline requires a OpenAIConfig")
        
        if config.openai.json_mode:
            model_kwargs = {"response_format": {"type": "json_object"}}
        else:
            model_kwargs = {}
        self.client = ChatOpenAI(model=config.openai.model_name, temperature=config.llm.temperature, model_kwargs=model_kwargs, callbacks = [LLMCallbacks()])
