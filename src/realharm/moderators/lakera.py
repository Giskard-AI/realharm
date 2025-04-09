"""Lakera Moderation API"""
import os
import requests
from realharm.models import ModerationOutput
from realharm.moderators.base import BaseModerator
from realharm.utils import ensure_conversation_turns

FLAGGED_LEVEL = ["l1_confident", "l2_very_likely"]
UNFLAGGED_LEVEL = ["l3_likely", "l4_less_likely", "l5_unlikely"]

def _map_role(role: str) -> str:
    if role == "agent":
        return "assistant"
    return role

class LakeraModerator(BaseModerator):
    def __init__(self):
        self.endpoint = "https://api.lakera.ai/v2/guard/results"

    def check(self, chat: list[dict[str, str]], **kwargs):

        chat = [{"role": _map_role(m["role"]), "content": m["content"]} for m in chat]
        chat = ensure_conversation_turns(chat, starts_with_user=False)

        for idx, message in enumerate(chat):
            partial_chat = chat[:idx+1]

            session = requests.Session()  # Allows persistent connection
            response = session.post(
                self.endpoint,
                json={"messages": partial_chat},
                headers={"Authorization": f"Bearer {os.getenv("LAKERA_GUARD_API_KEY")}"},
            )
            results = response.json()["results"]
            safe = True
            categories = []
            for res in results:
                if res["result"] in FLAGGED_LEVEL:
                    safe = False
                    categories.append(res["detector_type"])
            if not safe:
                return ModerationOutput(
                    safe=False,
                    categories=categories
                )

        return ModerationOutput(safe=True, categories=[])
