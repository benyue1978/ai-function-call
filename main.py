import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import AgentType
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

# 加载.env文件中的环境变量
load_dotenv()

# 请确保设置了OPENAI_API_KEY和SERPAPI_API_KEY环境变量

def answer_question(question: str) -> str:
    """
    Use LangChain with SerpAPI to answer a question, always reply in the language of the question.
    """
    system_prompt = "You are an intelligent assistant. Always answer the user's question in the same language as the question."

    llm = OpenAI(temperature=0)
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events or the world"
        )
    ]
    # 构建带有system prompt的agent
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, prompt=prompt
    )
    return agent.invoke({"input": question})["output"]

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py <your question>")
        exit(1)
    question = " ".join(sys.argv[1:])
    answer = answer_question(question)
    print(f"Answer: {answer}") 