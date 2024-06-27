import time

from typing import Any, Dict, List
from langchain_core.callbacks import BaseCallbackHandler
from langchain.schema import LLMResult

class LLMCallbacks(BaseCallbackHandler):
    def __init__(self):
        super(LLMCallbacks, self).__init__()
        self.starttime = None
        self.duration : int = 0
        self.model_name: str = None

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        """Run when LLM starts running."""
        model_info = None
        
        if 'kwargs' in serialized:
            kwargs = serialized['kwargs']
            if 'model_kwargs' in kwargs:
                model_kwargs = kwargs['model_kwargs']
                if 'messages' in model_kwargs:
                    messages = model_kwargs['messages']
                    if isinstance(messages, list) and len(messages) > 0:
                        first_message = messages[0]
                        if isinstance(first_message, dict) and 'model_info' in first_message:
                            model_info = first_message['model_info']
        
        self.model_name = model_info if model_info else serialized['name']
        self.starttime = time.time()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        self.duration = time.time() - self.starttime
        print(f"{self.model_name.capitalize()} Duration => {self.duration} seconds")

