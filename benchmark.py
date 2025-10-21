import os
import sys
import logging
import argparse
import warnings
import traceback
import multiprocessing
from config_manager import load_config
from gptcache.core import Cache
from gptcache.processor.pre import get_prompt
from gptcache.adapter.langchain_models import LangChainLLMs
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.manager.eviction import EvictionBase
from gptcache.similarity_evaluation import SbertCrossencoderEvaluation
from components.dataset_manager import DatasetManager
from components.cache_utils import embedding_func, system_cleanup, lsh_temperature_func
from components import custom_llm, new_cache_logger, cache_analyzer, lsh_based_estimator
from components.helpers import info_print, debug_print, get_info_level, process_request

os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore", message="The method `BaseLLM.__call__` was deprecated")

def main():
    try:
        parser = argparse.ArgumentParser(description="Run GPTCache benchmark experiments")
        parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file')
        args = parser.parse_args()

        #config = Config.from_yaml(args.config)
        config = load_config(args.config)

        INFO, DEBUG = get_info_level(config)

        print(f'[TEST] INFO={INFO}, DEBUG={DEBUG}')

        logging.basicConfig(
            level=config.logging['level'],
            format=config.logging['format'],
            stream=sys.stdout
        )

        CACHE_DIR = config.cache['CACHE_DIR']
        os.makedirs(CACHE_DIR, exist_ok=True)

        vector_store_params = {
            "dimension": config.vector_store['dimension'],
            "index_type": config.vector_store['index_type'],
            "metric_type": config.vector_store['metric_type'],
            "index_path": os.path.join(CACHE_DIR, "faiss.index"),
        }

        cache_base = CacheBase("sqlite", sql_url=f"sqlite:///{os.path.join(CACHE_DIR, 'cache.db')}")
        vector_base = VectorBase("faiss", **vector_store_params)

        max_cache_size = config.experiment['max_cache_size']
        cache_strategy = config.experiment['cache_strategy']
        
        data_manager = get_data_manager(cache_base, vector_base, max_size=max_cache_size, name=cache_strategy)

        dataset_manager = DatasetManager()
        dataset = config.experiment['dataset']
        dataset_manager.load_from_file(
            file_path=f"dataset_cache/{dataset}.json",
            dataset_name = dataset
        )
        dataset_manager.set_active_dataset(dataset)
        questions = dataset_manager.get_questions(dataset_name=dataset)

        selected_questions = []
        if config.experiment['partial_questions']:
            min = config.experiment['range_min']
            max = config.experiment['range_max']
            for q in questions[min:max]:
                selected_questions.append(q['question'])
        else:
            for q in questions:
                selected_questions.append(q['question'])

        CacheLogger = new_cache_logger.CacheLogger()
        LSHCache = lsh_based_estimator.LSHCache()
        llm = custom_llm.localLlama()
        cached_llm = LangChainLLMs(llm=llm)
        semantic_cache = Cache()

        semantic_cache.init(
            embedding_func=embedding_func,
            data_manager=data_manager,
            similarity_evaluation=SbertCrossencoderEvaluation(),
            pre_embedding_func=get_prompt,
            lsh_cache=LSHCache,
            temperature_func=lsh_temperature_func,
        )

        use_cache = config.experiment['use_cache']

        try:
            for i, question in enumerate(selected_questions, start=1):
                info_print(f'Processing question {i}/{len(selected_questions)}: {question}', INFO)
                process_request(
                    question,
                    cached_llm,
                    semantic_cache,
                    CacheLogger,
                    use_cache=use_cache,
                )
            CacheLogger.close()
            report_path = cache_analyzer.generate_latest_run_report(log_dir="cache_logs")
            print(f'Performance report saved to: {report_path}')
        finally:
            system_cleanup(semantic_cache, vector_base, data_manager)

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        logging.error("Full traceback:")
        logging.error(traceback.format_exc())

if __name__=="__main__":
    multiprocessing.set_start_method('spawn')
    main()