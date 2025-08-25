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
        #self.client = OpenAI(base_url="http://127.0.0.1:8080", api_key="sk-xxx")
        self.client = OpenAI(base_url="http://localhost:1337", api_key="sk-xxx")
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
            stop=stop,
            #cache_skip=True,
        )

        #print response in red
        #print("\033[91m" + str(response) + "\033[0m")

        # Check different response structures and extract content
        if hasattr(response, 'model_extra') and 'content' in response.model_extra:
            result = response.model_extra["content"]
        elif hasattr(response, 'choices') and len(response.choices) > 0:
            if hasattr(response.choices[0], 'text'):
                result = response.choices[0].text
            elif hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                result = response.choices[0].get('text', '')
            else:
                result = response.choices[0].get('text', '')
        else:
            result = str(response)


        # print("response:", response)
        # result = response.model_extra["content"]
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