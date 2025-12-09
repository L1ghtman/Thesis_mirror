import os
import sys
import time
import logging
import traceback
import argparse
from components import custom_llm, llm_logger
from components.dataset_manager import DatasetManager
from config_manager import load_config, get_config

parser = argparse.ArgumentParser(description="Run LLM benchmarks")
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()
config = load_config(args.config)

from components.helpers import info_print, debug_print, get_info_level, process_request

def main():
    try:
        config = get_config()
        INFO, DEBUG = get_info_level(config)

        #if config.sys['hpc']:
        #    job_id = os.environ.get('SLURM_JOB_ID', 'local')
        #    CACHE_DIR = f"llm_sim_runs/{CACHE_DIR}_{job_id}"

        #os.makedirs(CACHE_DIR, exist_ok=True)

        dataset_manager = DatasetManager()
        dataset_name    = config.experiment['dataset_name']
        load_from_file  = config.experiment['load_from_file']
        sample_size     = config.experiment['sample_size']

        if load_from_file:
            dataset_name = dataset_manager.load(
                dataset="file",
                file_path=f"dataset_cache/{dataset_name}.json",
                dataset_name=dataset_name
            )

        dataset_name = dataset_manager.load(
            dataset     = dataset_name,
            max_samples = sample_size
        )

        debug_print(dataset_name, DEBUG)
        dataset_manager.set_active_dataset(dataset_name)
        questions = dataset_manager.get_questions(dataset_name=dataset_name)

        selected_questions = []
        if config.experiment['partial_questions']:
            min = config.experiment['range_min']
            max = config.experiment['range_max']
            for q in questions[min:max]:
                selected_questions.append(q)
        else:
            for q in questions:
                selected_questions.append(q)

        llm = custom_llm.localLlama()
        Logger = llm_logger.LLMLogger()

        for i, question in enumerate(selected_questions, start=1):
            info_print(f'Processing question {i}/{len(selected_questions)}: {question}', INFO)
            start_time = time.time()
            answer = llm._call(question)
            end_time = time.time()
            response_time = end_time - start_time
            Logger.log_request(
                query=question,
                response=answer,
                response_time=response_time,
            )
            info_print(f"Question: {question}", INFO)
            info_print("Response time: {:.2f}".format(response_time), INFO)
            info_print(f"Answer: {answer}\n", INFO)
            info_print("\033[94m" + "-----------------------------------------------------------" + "\033[0m\n", INFO)

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        logging.error("Full traceback:")
        logging.error(traceback.format_exc())

if __name__=="__main__":
    main()