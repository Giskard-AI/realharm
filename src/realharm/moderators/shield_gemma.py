"""Google ShieldGemma Moderation"""

from realharm.modal.shield_gemma_inference import ShieldGemmaModel
from realharm.models import ModerationOutput
from realharm.moderators.base import BaseModerator
from realharm.utils import ensure_conversation_turns

MODEL_ID = "google/shieldgemma-27b"


def _map_role(role: str) -> str:
    if role == "agent":
        return "assistant"
    return role


class ShieldGemmaModerator(BaseModerator):
    def __init__(self, model_id: str = MODEL_ID):
        self.model = ShieldGemmaModel(model_id)

    def check(self, conversation: list[dict[str, str]], **kwargs):
        chat = [
            {"role": _map_role(m["role"]), "content": m["content"]}
            for m in conversation
        ]

        chat = ensure_conversation_turns(chat, starts_with_user=False)

        for idx, message in enumerate(chat):
            partial_chat = chat[:idx+1]

            # Shield Gemma needs at least one user message before a response
            if idx == 0 and partial_chat[0]["role"] != "user":
                partial_chat = [{"role": "user", "content": "Hello"}] + partial_chat
            res = self.model.run.remote(partial_chat)
            if res["unsafe"]:
                return ModerationOutput(safe=False, categories=[])
        return ModerationOutput(safe=True, categories=[])