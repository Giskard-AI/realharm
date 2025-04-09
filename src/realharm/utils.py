from pathlib import Path

import pandas as pd
import yaml


def load_yaml_file(file_path: str):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def save_yaml_file(file_path: str, data: dict):
    with open(file_path, "w") as file:
        yaml.dump(data, file)


def read_samples(datapath):
    samples = []
    for file_path in Path(datapath).glob("**/rh_*.yml"):
        with file_path.open("r") as f:
            sample = yaml.safe_load(f)

            samples.append(
                {
                    "sample_id": f"{file_path.parent.stem}_{file_path.stem}",
                    **sample,
                }
            )
    return pd.DataFrame(samples).sort_values(by="sample_id", ascending=False)


def ensure_conversation_turns(chat: list[dict[str, str]], starts_with_user: bool = True) -> list[dict[str, str]]:
    if starts_with_user:
        chat = [{"role": "user", "content": "Hello"}] + chat

    conversation_with_turns = []
    is_last_assistant_msg = True

    for idx, message in enumerate(chat):
        if message["role"] == "user" and is_last_assistant_msg:
            conversation_with_turns.append({"role": "user", "content": message["content"]})
            is_last_assistant_msg = False
        elif message["role"] == "user" and not is_last_assistant_msg:
            conversation_with_turns[-1]["content"] += message["content"]
        elif message["role"] == "assistant" and not is_last_assistant_msg:
            conversation_with_turns.append({"role": "assistant", "content": message["content"]})
            is_last_assistant_msg = True
        else:
            conversation_with_turns.append(message)

    return conversation_with_turns
