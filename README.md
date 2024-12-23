# Attention Entropy Patterns

_UPDATE_: This research also did not show promising results, but outputted interesting insights about entropy patterns. At least I learned a lot in the process!

This research project aims to identify the patterns of entropy and varentropy in attention weights, and their relation to model factuality, reasoning performance and hallucination rates.

This study is heavily inspired by [Entropix](https://github.com/xjdr-alt/entropix) project, which uses a special token sampling strategy based on entropy and varentropy of token probability distributions, which has shown promising results on reasoning.

This study aims to extrapolate this aproach deeper, into attention layers.

## Methodology

The study evaluates LLaMA 3.2 of sizes 1B, 3B, 11B (Vision), 90B (Vision)

The study uses a graph with X as token (step in autoregressive generation) and Y as average entropy / varentropy across layers across heads per specific token (step).

### Factual vs Creative

Study will ask GPT-4o to come up with 100 different prompts which question about dry and factural information, and 100 different prompts asking LLM do generate something creative.

Graph will be made for each model, displaying average entropy / varentropy trends for factual and creative prompts.

### Truthfulness and Hallucination

The study will evaluate models on [SimpleQA](https://openai.com/index/introducing-simpleqa/) benchmark by OpenAI, record attention weights each run, and later study will compare entropy and varentropy patterns across non-hallucinated responses (Correct, Not Attempted) and hallucinated (Incorrect).

### Reasoning

The study will evaluate models on [GSM8K](https://huggingface.co/datasets/openai/gsm8k) dataset. Later, the models' wrong answers will be provided to GPT-4o to act as a critic, given problem, suggested solution and right solution, and critic will attempt to highlight the "moment where things went wrong" in model's CoT output, leading to wrong answer.

After that, entropy and varentropy patterns will be compared between corrent and incorrect answers, and also will be compared between wrong answer's "ordinary" tokens and "moment where things went wrong", to understand whether LLMs may have different entropy and varentropy patterns at "wrong" tokens compared to the rest of the answer.

Later, this may provide an opportunity to develop an instrument which would assign each token "confidence" score, helping to combat hallucination in LLMs.
