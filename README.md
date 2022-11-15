# mats-3-aligning-lms
A common repo of the MATS 3.0 stream on Aligning Language Models


## Install

> Note: I couldn't get `tensorflow-text`, a dependency of BIG-Bench, to build on an Apple Silicon Mac. It's possible but takes a bit more time, so I suggest not using an Apple Silicon Mac for this project if you can. If you come up with a way to make it work, please share! This [issue](https://github.com/tensorflow/text/pull/756) is a good place to start.

1. Install python dependencies:

```bash
pip install git+https://github.com/google/BIG-bench.git
pip install openai python-dotenv scipy numpy
```

2. Create an `.env` file with your OpenAI API key to use it in scripts (see `.env.example` for inspiration).
