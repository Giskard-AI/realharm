import modal
import os

from realharm.modal.app import app, volume, inference_image_with_source, HF_MODEL_DIR

MODAL_HF_SECRET = os.getenv("MODAL_HF_SECRET")

@app.cls(
    volumes={HF_MODEL_DIR: volume},
    gpu="l4",
    image=inference_image_with_source,
    secrets=[modal.Secret.from_name(MODAL_HF_SECRET)],
)
class LlamaGuardModel:
    def __init__(self, repo_id: str = "meta-llama/Llama-Guard-3-8B"):
        self.repo_id = repo_id
        self.model = None
        self.tokenizer = None

    # @modal.build()
    # def build(self):
    #     from huggingface_hub import snapshot_download

    #     snapshot_download(self.repo_id)

    @modal.enter()
    def enter(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = AutoModelForCausalLM.from_pretrained(
            self.repo_id, use_cache=True, device_map="auto", torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo_id)
        self.model.eval()

    @modal.method()
    def run(self, chat: list[dict[str, str]]):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized!")

        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(
            "cuda"
        )
        output = self.model.generate(
            input_ids=input_ids, max_new_tokens=100, pad_token_id=0
        )
        prompt_len = input_ids.shape[-1]

        return self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
