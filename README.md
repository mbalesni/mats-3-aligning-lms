# mats-3-aligning-lms
A common repo of the MATS 3.0 stream on Aligning Language Models


## Install

> Note: I couldn't get `tensorflow-text`, a dependency of BIG-Bench, to build on an Apple Silicon Mac. It's possible but takes a bit more time, so I suggest not using an Apple Silicon Mac for this project if you can. If you come up with a way to make it work, please share! This [issue](https://github.com/tensorflow/text/pull/756) is a good place to start.

1. Install via pip:

```
pip install -e .
```

2. Create an `.env` file with your OpenAI API key to use it in scripts (see `.env.example` for inspiration).

## Example Usage
Below is an example of running three gpt3 models on the `emoji-movie` and `temporal-sequences` tasks, storing the results in JSON files in `results/test_run/emoji_movie` etc.
```
python src/main.py --exp-dir test_run --task-names temporal_sequences emoji_movie --models ada babbage curie
```