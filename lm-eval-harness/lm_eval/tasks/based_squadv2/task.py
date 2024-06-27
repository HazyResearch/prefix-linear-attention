from typing import List
import re
import numpy as np
from lm_eval.api.task import ConfigurableTask
from lm_eval.api.instance import Instance
import datasets


def _squad_metric(predictions, references):
    # squad_metric = load("squad_v2")
    squad_metric = datasets.load_metric("squad_v2")
    return squad_metric.compute(predictions=predictions, references=references)


def _squad_agg(key, items):
    predictions, references = zip(*items)
    return _squad_metric(predictions=predictions, references=references).get(key, 0)


class BasedSQUADCompletion(ConfigurableTask):
    VERSION = 0
    DATASET_PATH = "hazyresearch/based-squad"
    DATASET_NAME = "default"

    def __init__(self):
        super().__init__(config={'metadata': {'version': self.VERSION}})

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        out = doc['text'].strip()
        return out

    def doc_to_target(self, doc):
        return doc["value"]
        
    def construct_requests(self, doc, ctx, **kwargs):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """

        return [
            Instance(
                request_type="generate_until",
                doc=doc,
                arguments=(ctx, {
                    "until": ["\n"], 
                    "max_gen_toks": 48
                }),
                idx=0,
                **kwargs,
            ),
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=(ctx, " " + "unanswerable"),
                idx=0,
                **kwargs,
            ),
        ]

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        continuation, (logprob_unanswerable, _) = results

        from math import exp
        no_answer_probability = exp(logprob_unanswerable)
        predictions = {
            "id": doc["new_id"],
            "prediction_text": continuation,
            "no_answer_probability": no_answer_probability,
        }

        references = {
            "id": doc["new_id"],
            "answers": {'text':[doc["value"]], 'answer_start':[]}
        }

        out = {
            "exact": (
                predictions,
                references,
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "f1": (
                predictions,
                references,
            ),  # The F-score of predicted tokens versus the gold answer
            "contains": contains_score(continuation, [doc["value"]])
        }
        return out

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        from functools import partial
        return {
            "exact": partial(
                _squad_agg, "exact"
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "f1": partial(
                _squad_agg, "f1"
            ),  # The F-score of predicted tokens versus the gold answer
            "contains": np.mean,
        }


    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            "exact": True,  # Exact match (the normalized answer exactly match the gold answer)
            "f1": True,  # The F-score of predicted tokens versus the gold answer
            "contains": True,  # Exact match (the normalized answer exactly match the gold answer
        }
    

def contains_score(prediction: str, labels: List[str]):
    return max(
        int(bool(re.search(re.compile(re.escape(label), re.IGNORECASE), prediction)))
        for label in labels
    )


class BasedSQUADCompletionTwice(BasedSQUADCompletion):

    def doc_to_text(self, doc):
        context = doc["context"]
        context = context.split("Question:")[0]
        question = doc['question']
        doc_text = doc['text'].strip()

        out = (
            question + " " +
            context + "\n" + 
            doc_text 
        )
        return out

