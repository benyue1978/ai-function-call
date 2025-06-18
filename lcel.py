"""
lcel.py
-------
A minimal example of using LangChain Expression Language (LCEL)
to build a simple chain: user input -> tool call -> LLM summary.

▶ Requirements
pip install langchain openai 'langchain-core>=0.1.0'

▶ Usage
export OPENAI_API_KEY=sk-xxx
python lcel.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

load_dotenv()

# Example tool: simple FAQ search
def search_faq(keyword: str) -> str:
    """Search a hardcoded FAQ dictionary."""
    print("!!! search_faq called !!!")
    _FAQ = {
        "Tesla": "Tesla is an American electric vehicle and clean energy company.",
        "Model 3": "Model 3 is a compact electric sedan with up to 602 km range (CLTC).",
    }
    for k, v in _FAQ.items():
        if k.lower() in keyword.lower():
            return v
    return "Sorry, no relevant information found."

# Wrap the tool as a RunnableLambda
tool_chain = RunnableLambda(search_faq)

# Define a prompt for the LLM to summarize the tool result
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Summarize the following information for the user."),
    ("human", "{tool_result}")
])

# Initialize the LLM
temperature = 0.0
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)

# Build the LCEL chain: user input -> tool -> prompt -> LLM
lcel_chain = (
    {"tool_result": tool_chain} | prompt | llm
)

if __name__ == "__main__":
    user_question = "What is Tesla?"
    lcel_answer = lcel_chain.invoke(user_question)
    print("Q:", user_question)
    print("A:", lcel_answer.content.strip()) 