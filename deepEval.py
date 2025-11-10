from deepeval.metrics import AnswerRelevancyMetric,ContextualPrecisionMetric
from deepeval.evaluate import DisplayConfig
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from deepeval import evaluate
import os
import asyncio

load_dotenv()

def extract_test_results(data):

    extracted = []

    try:

        for test in data:
            test_info = {
                "test_name": test.name,
                "success": test.success,
                "input": test.input,
                "expected_output": test.expected_output,
                "actual_output": test.actual_output,
                "retrieval_context": test.retrieval_context,
                "metrics": []
            }

            for metric in test.metrics_data:
                metric_info = {
                    "metric_name": metric.name,
                    "score": metric.score,
                    "threshold": metric.threshold,
                    "success": metric.success,
                    "reason": metric.reason,
                    "evaluation_model": metric.evaluation_model
                }
                test_info["metrics"].append(metric_info)

            extracted.append(test_info)

    except Exception as e:

        print(f"Exception: {e}")

    return extracted

def execute( testCases: list[LLMTestCase] ):

    model = GeminiModel("gemini-2.5-flash",api_key=os.getenv('LLM_API_KEY'))

    answer_relevancy = AnswerRelevancyMetric(model=model,threshold=0.8)
    contextual_precision = ContextualPrecisionMetric(model=model,threshold=0.8)

    # test_case_example = LLMTestCase(
    #     input="Explain the difference between supervised and unsupervised learning with examples.",
    #     actual_output="""Supervised learning uses labeled data to train models that can predict outcomes — for instance, predicting exam scores based on study hours.
    # Unsupervised learning, on the other hand, works without labeled outputs, finding hidden patterns, like grouping news articles by topic using clustering.""",
    #     retrieval_context=["""Supervised learning uses labeled datasets to train models, meaning the input data is paired with the correct output. Common algorithms include linear regression and decision trees.  
    # Unsupervised learning works with unlabeled data, aiming to find hidden patterns or structures. Examples include clustering (like K-means) and dimensionality reduction (like PCA).
    # """],
    #     expected_output="""Supervised learning involves training a model on labeled data, where the correct outputs are known. For example, predicting house prices using historical data.
    # Unsupervised learning deals with unlabeled data to find patterns or groupings, such as using K-means clustering to group customers by purchasing behavior."""
    # )

    results = evaluate(test_cases=testCases, metrics=[answer_relevancy, contextual_precision],display_config=DisplayConfig(verbose_mode=False,print_results=False,show_indicator=False))

    return results

def deepEvalTest(testCases):

    results = execute(testCases)

    parsedResults = extract_test_results(data=results.test_results)

    # for pr in parsedResults:

    #     print(f"Test Result for {pr['text_name']}:\n",pr['metrics'],"\n")

    return parsedResults



