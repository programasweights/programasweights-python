"""
Example: Use ProgramAsWeights with LangChain.

Replace LLM calls in your LangChain pipeline with local neural programs
for deterministic, fast, and free execution.

Usage:
    pip install programasweights langchain

"""
from langchain.tools import tool
import programasweights as paw


# === Option 1: As a LangChain Tool ===

@tool
def extract_emails(text: str) -> str:
    """Extract email addresses from text and return as JSON list."""
    fn = paw.function("programasweights/email-extractor")
    return fn(text)


@tool
def classify_sentiment(text: str) -> str:
    """Classify the sentiment of text as positive, negative, or neutral."""
    fn = paw.function("programasweights/sentiment-classifier")
    return fn(text)


# Use in an agent:
# from langchain.agents import initialize_agent, AgentType
# from langchain.llms import OpenAI
#
# tools = [extract_emails, classify_sentiment]
# agent = initialize_agent(tools, OpenAI(), agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
# agent.run("Extract emails from: Contact alice@test.com and bob@example.org")


# === Option 2: As a custom LLM (replace API calls entirely) ===

from langchain.llms.base import LLM
from typing import Optional, List


class PawLLM(LLM):
    """Use a .paw program as a LangChain LLM."""
    
    program_id: str = ""
    _fn: any = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, program_id: str, **kwargs):
        super().__init__(program_id=program_id, **kwargs)
        self._fn = paw.function(program_id)
    
    @property
    def _llm_type(self) -> str:
        return "paw"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return self._fn(prompt)


# Usage:
# llm = PawLLM(program_id="yuntian-deng/email-extractor")
# result = llm("Contact alice@test.com")
# print(result)  # ["alice@test.com"]
#
# # In a chain:
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
#
# prompt = PromptTemplate(input_variables=["text"], template="{text}")
# chain = LLMChain(llm=llm, prompt=prompt)
# chain.run("Contact alice@test.com or bob@example.org")


# === Option 3: Replace evaluation step in RAG pipeline ===

def evaluate_answer_with_paw(prediction: str, target: str) -> bool:
    """Use PAW for answer equivalence checking in evaluation."""
    equiv_checker = paw.function("programasweights/answer-equiv")
    result = equiv_checker(f"Prediction: {prediction}\nTarget: {target}")
    return "equivalent" in result.lower()


# In your RAG evaluation:
# for pred, target in zip(predictions, ground_truth):
#     correct = evaluate_answer_with_paw(pred, target)
#     # No API calls, no cost, deterministic!


if __name__ == "__main__":
    print("LangChain + ProgramAsWeights Integration Examples")
    print()
    print("Option 1: Use as LangChain Tool (@tool decorator)")
    print("Option 2: Use as custom LLM (PawLLM class)")
    print("Option 3: Use for evaluation in RAG pipelines")
    print()
    print("See code comments for usage details.")
