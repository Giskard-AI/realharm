"""OpenAI Moderation API"""
from openai import OpenAI

from realharm.models import ModerationOutput
from realharm.moderators.base import BaseModerator


class OpenAIModerator(BaseModerator):
    def __init__(self, client: OpenAI = None, model: str = "text-moderation-stable"):
        self.client = OpenAI() if client is None else client
        self.model = model

    def check(self, conversation: list[dict[str, str]], is_sample_safe: bool, **kwargs):
        for msg in conversation:
            if is_sample_safe and msg["role"] == "user":
                continue
            response = self.client.moderations.create(
                input=msg["content"], model=self.model
            )

            if response.results[0].flagged:
                return ModerationOutput(
                    safe=False,
                    categories=[
                        c
                        for c, flagged in response.results[0]
                        .categories.model_dump()
                        .items()
                        if flagged
                    ],
                )

        return ModerationOutput(safe=True, categories=[])
