import modal

from realharm.modal.app import HF_MODEL_DIR, app, inference_image_with_source, volume


@app.cls(
    volumes={HF_MODEL_DIR: volume},
    gpu="l4",
    image=inference_image_with_source,
)
class LLMGuardScanner:
    def __init__(self):
        self.input_scanners = []
        self.output_scanners = []

    @modal.enter()
    def enter(self):
        from llm_guard.input_scanners import (
            Gibberish as InGibberish,
        )
        from llm_guard.input_scanners import (
            InvisibleText,
            PromptInjection,
            Secrets,
            TokenLimit,
        )
        from llm_guard.input_scanners import (
            Toxicity as InToxicity,
        )
        from llm_guard.output_scanners import (
            Bias,
            FactualConsistency,
            Gibberish,
            MaliciousURLs,
            NoRefusal,
            Relevance,
            Sensitive,
            Toxicity,
        )

        self.input_scanners = [
            InvisibleText(),
            Secrets(),
            PromptInjection(),
            TokenLimit(),
            InToxicity(),
            InGibberish(),
        ]

        self.output_scanners = [
            Relevance(),
            Sensitive(),
            Toxicity(),
            Bias(),
            MaliciousURLs(),
            Gibberish(),
        ]

    @modal.method()
    def scan_prompt(
        self, prompt: str, fail_fast: bool = True
    ) -> tuple[str, dict[str, bool], dict[str, float]]:
        from llm_guard import scan_prompt

        return scan_prompt(self.input_scanners, prompt, fail_fast=fail_fast)

    @modal.method()
    def scan_output(
        self, prev_msg: str, msg: str, fail_fast: bool = True
    ) -> tuple[str, dict[str, bool], dict[str, float]]:
        from llm_guard import scan_output

        return scan_output(self.output_scanners, prev_msg, msg, fail_fast=fail_fast)
