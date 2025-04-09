from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import argparse

ROOT_PATH = Path(__file__).parent.parent
load_dotenv(ROOT_PATH / ".env")

from realharm.modal.app import app
from realharm.moderators import MODERATOR_REPOSITORY
from realharm.utils import read_samples


DATA_PATH = ROOT_PATH / "data" 

samples = read_samples(DATA_PATH)


@app.local_entrypoint()
def main(*arglist):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--moderator_names", nargs="+", help="The moderator names to run the benchmark for", default=["OpenAIModerator"]
    )
    args = parser.parse_args(args=arglist)
    moderator_names = args.moderator_names

    if len(moderator_names) == 0:
        raise ValueError("No moderator names provided")
    

    if moderator_names[0] == "all":
        moderator_list = [moderator() for name, moderator in MODERATOR_REPOSITORY.items()]
    else:
        moderator_list = [MODERATOR_REPOSITORY[name]() for name in moderator_names]

    for moderator in moderator_list:
        results = []

        for sample in samples.itertuples():
            print("Processing", sample.sample_id)

            result = moderator.check(sample.conversation, is_sample_safe=sample.label == "safe")

            print("label = ", result.label, "categories = ", result.categories)
            print()

            results.append(
                {
                    "sample_id": sample.sample_id,
                    "moderator": moderator.__class__.__name__,
                    "moderation_label": result.label,
                    "moderation_categories": result.categories,
                }
            )

        pd.DataFrame(results).to_json(
            ROOT_PATH / f"benchmark/benchmark_{moderator.__class__.__name__}.jsonl",
            orient="records",
            lines=True,
        )


if __name__ == "__main__":
    main()

