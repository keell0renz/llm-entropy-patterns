def creative_prompt(N: int = 50):
    CREATIVE_PROMPT = f"""
    You are an interviewer of novel Large Language Models.

    Your job is to come up with {N} prompts which ask a target LLM to generate a response.
    It may be a question, or a prompt to say something. 

    Your goal is to ask such questions and prompts that a target LLM would produce a creative and unique response,
    which is not just memorized phrases during pre-training. Your goal is to make model think, imagine, come up, and "hallucinate".

    You must ensure that model provides a consistent and extensive response of 200 words, with allowed deviation of +- 10 words per answer.

    However -- you NEED NOT to state that instruction directly, it will be encoded in LLMs system prompt. This requirement is mainly for you
    to come up with proper prompts which provide opportunity for model to give sufficiently long answer.

    Warning: You must NOT enclose prompts in quotes, it will break the dataset.

    Examples:

    "Come up with a crazy history about a humanoid panda, mandelbrot set and Cthulhu cooperating to create an empire to worship Aztec god of sun."

    "Describe your traits of personality and what kind of 'totem fruit' you would assign to them? Be original"

    "How do I solve a mathematical problem with with an IPhone, matchlock musket, my grandad's whiskey and cola from friend Jeremiah? Use all the items provided!"

    "If you were a hog powered by thermonuclear testicles in your armpits, describe your usual day and 'what's in my bag today giiiirls??? :)'."

    "Create a story involving gnomes, Jurgen Shmidthuber, illegal marshmallows, corrupt DEA agents and russian hackers. Create an intriguing and complex plot."
    
    """.strip()

    return CREATIVE_PROMPT


def factual_prompt(N: int = 50):
    FACTUAL_PROMPT = """
    You are an interviewer of novel Large Language Models.

    Your job is to come up with {N} prompts which ask a target LLM to generate a response.
    It may be a question, or a prompt to say something. 

    Your goal is to ask such questions and prompts that a target LLM would produce a factual and formal, explanatory, truthful response,
    which uses information memorized during pre-training. Your goal is to make model just output / explain knowledge and facts received during intentional
    overfitting on pre-training dataset. Model must not be super creative, original, just an "AI version of Google or Wolfram Alpha".

    You must ensure that model provides a consistent and extensive response of 200 words, with allowed deviation of +- 10 words per answer.

    However -- you NEED NOT to state that instruction directly, it will be encoded in LLMs system prompt. This requirement is mainly for you
    to come up with proper prompts which provide opportunity for model to give sufficiently long answer.

    Warning: You must NOT enclose prompts in quotes, it will break the dataset.

    Examples:

    "Explain the process of mitosis."

    "How does water cycle in the nature?"

    "Who is Abraham Lincoln?"

    "Output begginning part of U.S constitution."

    "History of U.S summarized from foundation to modernity."

    """.strip()

    return FACTUAL_PROMPT
