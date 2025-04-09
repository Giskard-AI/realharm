"""Mistral Moderation API"""
import os
import time
from mistralai import Mistral, SDKError
import backoff
import logging

from realharm.models import ModerationOutput
from realharm.moderators.base import BaseModerator
from realharm.utils import ensure_conversation_turns

logging.getLogger('backoff').addHandler(logging.StreamHandler())

api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

def _map_role(role: str) -> str:
    if role == "agent":
        return "assistant"
    return role

class MistralModerator(BaseModerator):
    def __init__(self):
        self.model = "mistral-moderation-latest"
        self.client = Mistral(api_key=api_key)

    @backoff.on_exception(backoff.expo, SDKError, max_tries=10, factor=5)
    def check_one_turn(self, chat: list[dict[str, str]], idx: int):
        response = self.client.classifiers.moderate_chat(
            model=self.model,
            inputs=chat,
        )

        categories = [category_name for category_name, unsafe in response.results[0].categories.items() if unsafe]
        safe = len(categories) == 0
        return safe, categories

    def check(self, chat: list[dict[str, str]], **kwargs):

        chat = [{"role": _map_role(m["role"]), "content": m["content"]} for m in chat]
        chat = ensure_conversation_turns(chat, starts_with_user=False)

        for idx, message in enumerate(chat):
            partial_chat = chat[:idx+1]

            safe, categories = self.check_one_turn(partial_chat, idx)
            if not safe:
                return ModerationOutput(
                    safe=False,
                    categories=categories
                )

        return ModerationOutput(safe=True, categories=[])
