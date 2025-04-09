"""ProtectAI LLM Guard"""

from realharm.modal.llm_guard_inference import LLMGuardScanner
from realharm.models import ModerationOutput
from realharm.moderators.base import BaseModerator


class LLMGuardModerator(BaseModerator):
    def __init__(self):
        self.scanner = LLMGuardScanner()

    def check(self, conversation: list[dict[str, str]], is_sample_safe: bool, **kwargs) -> ModerationOutput:
        for i, msg in enumerate(conversation):
            if is_sample_safe and msg["role"] == "user":
                continue
            
            if msg["role"] == "user":
                _, is_valid, _ = self.scanner.scan_prompt.remote(
                    msg["content"], fail_fast=False
                )

            else:
                try:
                    prev_msg = conversation[i - 1]["content"]
                except IndexError:
                    prev_msg = ""

                _, is_valid, _ = self.scanner.scan_output.remote(
                    prev_msg,
                    msg["content"],
                    fail_fast=False,
                )

            invalid_categories = [k for k, v in is_valid.items() if not v]

            if invalid_categories:
                return ModerationOutput(safe=False, categories=invalid_categories)

        return ModerationOutput(safe=True, categories=[])
