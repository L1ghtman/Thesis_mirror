import re 
import sys
import json
from pathlib import Path
from typing import Optional
from deepeval.models import DeepEvalBaseLLM
from openai import OpenAI
import torch
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)

class LocalEvalLlama(DeepEvalBaseLLM):
    """
    Wrapper for using local LLM model with DeepEval.
    """

    def __init__(self):
        self.client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-xxx")
        self.model_name = "Local LLM" # TODO: implement options for selecting eval model
        self.STATEMENTS_GRAMMAR = open("statements.gbnf").read()

    def load_model(self):
        return self.client
    
    def _clean_json_response(self, response: str) -> str:
        """Clean response of any LLM generated artifacts"""
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)

        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json_match.group(0).strip()

        return response.strip()

    def generate(self, prompt: str) -> str:
        client = self.load_model()
        needs_json = "json" in prompt.lower() or '"' in prompt or "{" in prompt

        if needs_json:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a precise evaluator. "
                        "You MUST output a valid JSON object with exactly one key named 'statements'. "
                        "The value of 'statements' must be a list of strings. "
                        "Do not use objects or other keys. "
                        "Example: {\"statements\": [\"statement 1\", \"statement 2\"]}"
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        else:
            messages = [
                {
                    "role": "user", 
                    "content": prompt
                }
            ]

        try:
            response = client.chat.completions.create(
                model="localmodel",
                messages=messages,
                max_tokens=2000,
                temperature=0.1,
                #response_format={"type": "json_object"},
                top_p=0.95,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                #extra_body={"grammar": self.STATEMENTS_GRAMMAR}
            )

            result = response.choices[0].message.content

            if needs_json:
                result = self._clean_json_response(result)
            
            return result
        
        except Exception as e:
            raise RuntimeError(f"Error generatign respones: {e}")
    
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name
    
class LocalEvalTransformer(DeepEvalBaseLLM):

    #model_name = "Qwen/Qwen2.5-72B-Instruct"

    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-7B-Instruct"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, max_new_tokens=2000, do_sample=False)

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: Optional[BaseModel] = None):
        if schema:
            parser = JsonSchemaParser(schema.model_json_schema())
            prefix_fn = build_transformers_prefix_allowed_tokens_fn(self.tokenizer, parser)
            output = self.pipe(prompt, prefix_allowed_tokens_fn=prefix_fn, return_full_text=False)
        else:
            print("[DEBUG] no schema")
            output = self.pipe(prompt, return_full_text=False)
        
        result = output[0]["generated_text"]
        return schema(**json.loads(result)) if schema else result

    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model_name

    
def get_local_llama():
    return LocalEvalLlama()

def get_local_transformer():
    return LocalEvalTransformer()