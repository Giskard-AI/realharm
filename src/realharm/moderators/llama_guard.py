"""Meta Llama Guard Moderation"""

from realharm.modal.llama_guard_inference import LlamaGuardModel
from realharm.modal.prompt_guard_inference import PromptGuardModel
from realharm.models import ModerationOutput
from realharm.moderators.base import BaseModerator
from realharm.utils import ensure_conversation_turns

MODEL_ID = "meta-llama/Llama-Guard-3-8B"


def _map_role(role: str) -> str:
    if role == "agent":
        return "assistant"
    return role


class LlamaGuardModerator(BaseModerator):
    def __init__(self, model_id: str = MODEL_ID):
        self.model = LlamaGuardModel(model_id)
        self.prompt_model = PromptGuardModel()
    def check(self, conversation: list[dict[str, str]], is_sample_safe: bool, **kwargs):
        chat = [
            {"role": _map_role(m["role"]), "content": m["content"]}
            for m in conversation
        ]

        print("Length of chat before ensuring turns:", len(chat))
        chat = ensure_conversation_turns(chat, starts_with_user=True)
        print("Length of chat after ensuring turns:", len(chat))

        for idx, message in enumerate(chat):
            if not is_sample_safe and message["role"] == "user":
                # PromptGuard returns true if there is a jailbreak in the message
                result = self.prompt_model.run.remote(message["content"])
                if result:
                    return ModerationOutput(safe=False, categories=["JAILBREAK"])
                        
            # LlamaGuard always wants the conversation to start with
            # a user message, we force this by adding a user greeting
            # if the conversation does not start with a user message.
            partial_chat = chat[:idx+1]
            if partial_chat[0]["role"] != "user":
                partial_chat = [{"role": "user", "content": "Hello"}] + partial_chat

            res = self.model.run.remote(partial_chat)

            # Parse response
            label, *extras = res.strip().split("\n")
            try:
                categories = extras[0].split(",")
            except IndexError:
                categories = []
            
            if label.startswith("unsafe"):
                return ModerationOutput(safe=False, categories=categories)

        return ModerationOutput(safe=True, categories=[])
