import os
from components.helpers import prompt_format
from typing import Any, List, Mapping, Optional, ClassVar, Iterator
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from openai import OpenAI
from config_manager import get_config


class localLlama(LLM):

    model_name: ClassVar[str] = "inet/llama3"
    client: Any = None
    first_call: bool = True
    system_prompt: str = "You are a helpful AI assistant."
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        config = get_config()
        url = config.sys.url

        # Initialize the client. We're using the OpenAI API here but redirecting to a local server.
        self.client = OpenAI(base_url=url, api_key="sk-xxx")
        self.system_prompt = config.sys.system_prompt
        print(f'[DEBUG] system_prompt: {self.system_prompt}')

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
        response = self.client.chat.completions.create(
            model="llama3",
            messages=[
                {
                    "role": "system", 
                    "content": self.system_prompt
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=1000,
            temperature=0.7,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop,
        )

        result = response.choices[0].message.content

        return result

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
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