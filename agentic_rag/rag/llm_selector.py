"""
llm_selector.py
Utility to select an LLM (OpenAI or Groq) in LangChain format.
"""

import os
from langchain_openai import OpenAI as LangOpenAI
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()


def get_llm(provider: str = "openai", temperature: float = 0.4, max_tokens: int = 500):
    """
    Return an initialized LLM instance based on provider.
    Supported providers: 'openai', 'groq'
    """

    provider = provider.lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY in environment.")
        return LangOpenAI(temperature=temperature, max_tokens=max_tokens)

    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Missing GROQ_API_KEY in environment.")
        return ChatGroq(
            temperature=temperature,
            model_name="llama-3.1-8b-instant",
            max_tokens=max_tokens
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'groq'.")


# ----------------------------------------------------------------------
# 🧪 Example usage
# ----------------------------------------------------------------------
# if __name__ == "__main__":
#     llm = get_llm(provider="groq")
#     response = llm.invoke("What is Milvus and why is it used in RAG pipelines?")
#     print("Response:\n", response.content)
