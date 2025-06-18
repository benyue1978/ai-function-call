import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent, AgentType
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
    tools = load_tools(["serpapi", "llm-math"], llm=llm)
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