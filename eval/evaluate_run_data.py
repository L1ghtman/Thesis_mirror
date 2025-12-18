import json
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import pandas as pd
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
import utils
import argparse
from custom_llm import get_local_llama, get_local_transformer
import os
import warnings
import time

os.environ['DEEPEVAL_TELEMETRY_OPT_OUT'] = 'YES'
warnings.filterwarnings('ignore')

def run_metric(metric, test_case, metric_name):
    try:
        start_time = time.time()
        metric.measure(test_case)
        end_time = time.time()
        score = metric.score
        success = metric.is_successful()
        reason = metric.reason
        print(f"\n{metric_name}")
        print(f"    Score: {score:.2f}")
        print(f"    Passed: {'✓' if success else '✗'}")
        print(f"    Reason: {reason[:150]}...")
        data = {
            "score": score,
            "success": success,
            "reason": reason,
            "time" : end_time - start_time
        }
        return True, data
    except Exception as e:
        error_msg = str(e)
        print(f"\n{metric_name}: ✗ Error")
        if "timed out" in error_msg.lower():
            print(f"  Issue: Request timed out")
        elif "invalid JSON" in error_msg.lower():
            print(f"  Issue: Model output was not valid JSON")
            print(f"  Tip: The local model may need better prompting for structured output")
        elif "'statements'" in error_msg:
            print(f"  Issue: Model output missing required JSON key 'statements'")
        elif "context" in error_msg.lower():
            print(f"  Issue: {error_msg}")
            print(f"  Tip: This metric requires 'context' or 'retrieval_context' in test case")
        else:
            print(f"  Issue: {error_msg[:200]}")
        data = {
            "error": error_msg,
            "time" : end_time - start_time
        }
        return False, data

def run_relevancy_metric(model, run_data):
    eval_data = []
    metric = AnswerRelevancyMetric(
        model=model,
        threshold=0.8,
        async_mode=False
    )
    for item in run_data:
        test_case = LLMTestCase(
            input=item.query,
            actual_output=item.response
        )
        success, data = run_metric(metric, test_case, 'AnswerRelevancy')
        if success:
            print("\n✓ Evaluation successful!")
        else:
            print("\n✗ Evaluation failed - see tips above")
        eval_data.append(data)
    return eval_data

def save_data(eval_data, run_id):
    with open(f"eval_logs/{run_id}_eval.json", 'w') as file:
        json.dump(eval_data, file, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, required=True, help='Path to the file containing the run data to be analyzsed.')
    args = parser.parse_args()

    # cache_run_003.json
    file = args.run 
    run_id = file.split('_')[3].split('.')[0]
    print(f"[DEBUG] run_id: {run_id}")
    run_data = utils.ingest_data(file) 
    try:
        model = get_local_transformer()
        #model = get_local_llama()
        #model = get_robust_local_llama()
        eval_data = run_relevancy_metric(model, run_data)
        save_data(eval_data, run_id)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__=="__main__":
    main()