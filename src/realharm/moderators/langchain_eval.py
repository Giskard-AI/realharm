"""Langchain Evaluation"""
from langchain.evaluation.criteria import CriteriaEvalChain
from langchain_openai import ChatOpenAI
from realharm.models import ModerationOutput
from realharm.moderators.base import BaseModerator

class LangchainEvalModerator(BaseModerator):
    def __init__(self):
        self.criteria = ["misogyny", "harmfulness", "maliciousness", "controversiality", "criminality", "insensitivity"]

    def check(self, conversation: list[dict[str, str]], is_sample_safe: bool, **kwargs):
        llm = ChatOpenAI(temperature=0)
        evaluators ={crit: CriteriaEvalChain.from_llm(llm=llm, criteria=crit) for crit in self.criteria}
        for idx, message in enumerate(conversation):
            # Langchain Eval always need the previous message for context 
            # to evaluate the current message, we put an initial 
            if is_sample_safe and message["role"] == "user":
                continue 
            if idx == 0:
                eval_results = {crit: evaluators[crit].evaluate_strings(prediction=message["content"], input="") for crit in self.criteria}
            else:
                prev_message = conversation[idx - 1]["content"]
                eval_results = {crit: evaluators[crit].evaluate_strings(prediction=message["content"], input=prev_message) for crit in self.criteria}

            categories = [crit for crit in self.criteria if eval_results[crit]["value"] == "Y"]
            if len(categories) > 0:
                return ModerationOutput(safe=False, categories=categories)
        return ModerationOutput(safe=True, categories=[])
    
