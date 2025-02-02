import os
from components.helpers import prompt_format
from typing import Any, List, Mapping, Optional, ClassVar, Iterator
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from openai import OpenAI


class localLlama(LLM):

    model_name: ClassVar[str] = "inet/llama3"
    client: Any = None
    first_call: bool = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize the client. We're using the OpenAI API here but redirecting to a local server.
        self.client = OpenAI(base_url="http://127.0.0.1:8080", api_key="sk-xxx")

    @property
    def _llm_type(self) -> str:
        return "custom local"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        """
        Make an API call to the model on the server using the specified prompt and return the response.
        """
        prompt, self.first_call = prompt_format(prompt, self.first_call)

        response = self.client.completions.create(
            model="llama3",
            prompt=prompt,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop
        )

        # print("response:", response)
        result = response.model_extra["content"]
        return result

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """
        Stream the LLM on the given prompt.
        """

        prompt, self.first_call = prompt_format(prompt, self.first_call)

        response = self.client.completions.create(
            model="llama3",
            prompt=prompt,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop,
            stream=True
        )

        for chunk in response:
            token = chunk.content
            chunk = GenerationChunk(text=token)
            
            yield chunk