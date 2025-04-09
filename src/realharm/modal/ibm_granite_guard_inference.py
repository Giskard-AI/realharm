import modal
import os
from realharm.modal.app import app, volume, inference_image_with_source, HF_MODEL_DIR

MODAL_HF_SECRET = os.getenv("MODAL_HF_SECRET")

@app.cls(
    volumes={HF_MODEL_DIR: volume},
    gpu="L4",
    image=inference_image_with_source,
    secrets=[modal.Secret.from_name(MODAL_HF_SECRET)],
)
class GraniteGuardModel:
    def __init__(self, repo_id: str = "ibm-granite/granite-guardian-3.1-8b"):
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
        import math
        import torch

        # Code from https://huggingface.co/ibm-granite/granite-guardian-3.1-8b

        safe_token = "No"
        unsafe_token = "Yes"
        nlogprobs = 20

        def parse_output(output, input_len):
            label, prob_of_risk = None, None

            if nlogprobs > 0:

                list_index_logprobs_i = [torch.topk(token_i, k=nlogprobs, largest=True, sorted=True)
                                        for token_i in list(output.scores)[:-1]]
                if list_index_logprobs_i is not None:
                    prob = get_probabilities(list_index_logprobs_i)
                    prob_of_risk = prob[1]

            res = self.tokenizer.decode(output.sequences[:,input_len:][0],skip_special_tokens=True).strip()
            if unsafe_token.lower() == res.lower():
                label = unsafe_token
            elif safe_token.lower() == res.lower():
                label = safe_token
            else:
                label = "Failed"

            return label, prob_of_risk.item()

        def get_probabilities(logprobs):
            safe_token_prob = 1e-50
            unsafe_token_prob = 1e-50
            for gen_token_i in logprobs:
                for logprob, index in zip(gen_token_i.values.tolist()[0], gen_token_i.indices.tolist()[0]):
                    decoded_token = self.tokenizer.convert_ids_to_tokens(index)
                    if decoded_token.strip().lower() == safe_token.lower():
                        safe_token_prob += math.exp(logprob)
                    if decoded_token.strip().lower() == unsafe_token.lower():
                        unsafe_token_prob += math.exp(logprob)

            probabilities = torch.softmax(
                torch.tensor([math.log(safe_token_prob), math.log(unsafe_token_prob)]), dim=0
            )

            return probabilities

        # Please note that the default risk definition is of `harm`. If a config is not specified, this behavior will be applied.
        guardian_config = {"risk_name": "harm"}
        input_ids = self.tokenizer.apply_chat_template(
            chat, guardian_config = guardian_config, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        input_len = input_ids.shape[1]

        self.model.eval()

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=20,
                return_dict_in_generate=True,
                output_scores=True,
            )

        label, prob_of_risk = parse_output(output, input_len)

        print(f"# risk detected? : {label}") # Yes
        print(f"# probability of risk: {prob_of_risk:.3f}") # 0.995

        return {"unsafe": label == "Yes", "score": prob_of_risk}

        