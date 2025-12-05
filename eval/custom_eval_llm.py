import os
os.environ["HF_HOME"] = "/home/users/l/luca_lichterman/huggingface_cache"
os.environ["HF_HUB_CACHE"] = "/home/users/l/luca_lichterman/huggingface_cache"
os.environ["DEEPEVAL_HOME"] = "/tmp/.deepeval"

import json
import torch
from pydantic import BaseModel
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

class LocalEvalLlama(DeepEvalBaseLLM):
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, max_new_tokens=2000, do_sample=False)

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: Optional[BaseModel] = None):
        if schema:
            parser = JsonSchemaParser(schema.model_json_schema())
            prefix_fn = build_transformers_prefix_allowed_tokens_fn(self.tokenizer, parser)
            output = self.pipe(prompt, prefix_allowed_tokens_fn=prefix_fn, return_full_text=False)
        else:
            output = self.pipe(prompt, return_full_text=False)
        
        result = output[0]["generated_text"]
        return schema(**json.loads(result)) if schema else result

    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model_name

# Test with deepeval
print("Loading model...")
eval_llm = LocalEvalLlama()

print("Running AnswerRelevancyMetric...")
test_case = LLMTestCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris."
)

metric = AnswerRelevancyMetric(threshold=0.5, model=eval_llm)
metric.measure(test_case)

print(f"Score: {metric.score}")
print(f"Passed: {metric.is_successful()}")
print(f"Reason: {metric.reason}")