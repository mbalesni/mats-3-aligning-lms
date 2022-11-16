from __future__ import annotations
import json
import argparse
import sys
from typing import Union, cast
import logging
import pandas as pd
from pathlib import Path
from tqdm.autonotebook import tqdm

from src.openai_bb import get_task, evaluate_on_task

def main():
    args = parse_args(sys.argv[1:])
    project_dir = Path(__file__).resolve().parent.parent
    base_data_dir = Path(project_dir, "data")
    # writing to google drive if we are in a colab notebook
    base_results_dir = Path(project_dir, "results")

    # if a results directory is supplied, use that as the experiment dir
    # otherwise, generate one from the dataset name and current time
    if args.exp_dir is not None:
        write_dir = Path(base_results_dir, args.exp_dir)
    else:
        write_dir = Path(".")
    write_dir.mkdir(parents=True, exist_ok=True)

    # we have to set up the logging AFTER deciding on a dir to write to
    log_path = Path(write_dir, "log.log")
    arg_log_path = Path(write_dir, "args.log")
    with arg_log_path.open("w") as f:
        json.dump(args.__dict__, f, indent=2)

    logging.info(f"Logging set up with args\n{args}")
    logging.info(f"Saving to results to {write_dir}")

    # put a copy of the data in the experiment dir for reference

    # device = "cuda:0" if args.use_gpu and torch.cuda.is_available() else "cpu"
    model_names = args.models
    task = get_task(args.dataset_name)
    all_results = []
    for model_name in tqdm(model_names):
        all_results.append(evaluate_on_task(task, model_name))

    # final step to add all results to a jsonl
    for results, model_name in zip(all_results, model_names):
        results_path = Path(write_dir, model_name + ".json")
        with results_path.open("w") as f:
            json.dump(results, f)


def set_up_logging(log_path: Path, logging_level: str):
    logging_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARN,
        "error": logging.ERROR,
    }
    
    logging.basicConfig(
        level=logging_levels[logging_level],
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    # suppress debug warnings from the Requests library
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Run model sizes and get losses for texts in a file"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="The name of the dataset to use",
        required=False,
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        help="The name of the experiment to resume or write to",
        required=False,
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="The specific models to use",
        default=["ada", "babbage", "curie"],
        choices=[
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl",
            "gpt-neo-125M",
            "gpt-neo-1.3B",
            "gpt-neo-2.7B",
            "gpt-j-6B",
            "ada",
            "babbage",
            "curie",
            "davinci",
            "text-ada-001",
            "text-babbage-001",
            "text-curie-001",
            "text-davinci-001",
            "text-davinci-002",
            "opt-125m",
            "opt-350m",
            "opt-1.3b",
            "opt-2.7b",
            "opt-6.7b",
            "opt-13b",
        ],
        required=True,
    )
    parser.add_argument(
        "--logging-level",
        type=str,
        help="The level of logging to print",
        default="info",
        choices=[
            "debug",
            "info",
            "warn",
            "error",
        ],
    )
    args = parser.parse_args(args)
    return args

if __name__ == "__main__":
    main()