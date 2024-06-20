import time

from typing import Any, Dict, List
from langchain_core.callbacks import BaseCallbackHandler
from langchain.schema import LLMResult

class LLMCallbacks(BaseCallbackHandler):
    def __init__(self):
        super(LLMCallbacks, self).__init__()
        self.starttime = None
        self.duration : int = 0

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""

        self.starttime = time.time()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        self.duration = time.time() - self.starttime
        print(f"LLM Model Duration => {self.duration} seconds")

