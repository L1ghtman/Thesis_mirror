from gptcache.similarity_evaluation import SbertCrossencoderEvaluation
from components.custom_sim_eval import CustomSimilarityEvaluation
from components import custom_sbert

evaluation = custom_sbert.SbertCrossencoderEvaluation()
#evaluation = SbertCrossencoderEvaluation()
custom_eval = CustomSimilarityEvaluation()
score = evaluation.evaluation(
    {
        'question': 'What is the color of sky?'
    },
    {
        'question': 'hello'
    }
)



print(score)