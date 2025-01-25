import os
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from openai import OpenAI


class localLlama(LLM):

    model_name = "synogate/llama3"
    client: Any = None
    first_call = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize the client. We're using the OpenAI API here but redirecting to a local server.
        self.client = OpenAI(base_url="http://s.synogate.com:8080/v1", api_key="sk-xxx")

    @property
    def _llm_type(self) -> str:
        return "custom local"
    
    def _call(
        self,
        prompt: str,
        stop=["\n"],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        """
        Make an API call to the model on the server using the specified prompt and return the response.
        """
        prompt, self.first_call = prompt_format(prompt, self.first_call)

        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop
        )

        result = response.model_extra["content"]
        return result