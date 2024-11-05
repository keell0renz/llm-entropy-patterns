import instructor
from pydantic import BaseModel
from openai import OpenAI
from typing import List


class Prompts(BaseModel):
    factual: List[str]
    creative: List[str]
