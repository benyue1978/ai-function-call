"""
mini_agent.py
-------------
A minimal Agent + Tool framework with LangChain as the final LLM driver.

▶ 安装依赖
pip install langchain openai  # 选用其它 LLM 供应商时替换 openai

▶ 运行
export OPENAI_API_KEY=sk-xxx    # 如果你用的是 OpenAI
python mini_agent.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage

load_dotenv()

# ────────────────────────────────
# 1. Tool 定义
# ────────────────────────────────
@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[str], str]


# 示例数据源
_FAQ = {
    "特斯拉": "Tesla 是一家美国电动车与能源公司，使命是加速世界向可持续能源转变。",
    "Model 3": "Model 3 是一款紧凑型纯电动轿车，最高续航 602 km（CLTC）。",
}


def search_faq(keyword: str) -> str:
    print("!!! search_faq called !!!")
    def normalize(s):
        return s.strip().replace(" ", "").replace("　", "").lower()
    keyword_norm = normalize(keyword)
    for k in _FAQ:
        k_norm = normalize(k)
        if k_norm in keyword_norm:
            print(f"[Tool命中] {k} in {keyword}")
            return _FAQ[k]
    print("[Tool未命中] 返回兜底答案")
    return "抱歉，未找到相关信息。"


TOOLS: List[Tool] = [
    Tool(name="SearchFAQ", description="查找公司 FAQ 数据", func=search_faq),
]

TOOL_NAMES = ", ".join(t.name for t in TOOLS)

# ────────────────────────────────
# 2. Agent 框架
# ────────────────────────────────
@dataclass
class AgentAction:
    tool: str
    tool_input: str
    log: str


@dataclass
class AgentFinish:
    result: str
    log: str


class MiniAgent:
    """
    一个"单步"Agent：让 LLM 决定
    ① 调哪个 Tool，
    ② 或者直接给最终答案。
    """

    SYSTEM_PROMPT = (
        "你是一个可以调用工具的智能助手。\n"
        f"可用工具列表：{TOOL_NAMES}\n"
        "当需要调用工具时，请输出：\n"
        "Action: <tool_name>\n"
        "Action Input: <input>\n"
        "Action Input 必须只用用户问题中的核心关键词，不要用完整问题，不要翻译或改写。\n"
        "你必须优先调用工具，只有在工具无结果时才可以直接输出Final Answer。\n"
        "当你有了结果时，输出：\n"
        "Final Answer: <answer>\n"
        "不要输出其他格式。"
    )

    def __init__(self, llm):
        self.llm = llm
        self.intermediate_steps: List[Tuple[AgentAction, str]] = []

    # -------- 核心循环 -------- #
    def run(self, question: str) -> str:
        # 1) 让 LLM 思考下一步
        scratchpad = self._construct_scratchpad()
        user_prompt = f"{scratchpad}\nQuestion: {question}"
        resp = self._llm_call(user_prompt)
        print("[LLM输出]", resp)
        parsed = self._parse(resp)

        # 2) 如果是工具调用
        if isinstance(parsed, AgentAction):
            obs = self._exec_tool(parsed)
            self.intermediate_steps.append((parsed, obs))
            # 仅做一次工具调用，再让 LLM 总结
            scratchpad = self._construct_scratchpad()
            final_resp = self._llm_call(scratchpad)
            print("[LLM输出]", final_resp)
            final = self._parse(final_resp)
            assert isinstance(final, AgentFinish)
            return final.result

        # 3) 如果直接给出答案
        assert isinstance(parsed, AgentFinish)
        return parsed.result

    # ────────── 私有辅助 ──────────
    def _construct_scratchpad(self) -> str:
        if not self.intermediate_steps:
            return ""
        parts = []
        for action, obs in self.intermediate_steps:
            parts.append(action.log)
            parts.append(f"Observation: {obs}")
        return "\n".join(parts) + "\n"

    def _llm_call(self, message: str) -> str:
        msgs = [AIMessage(content=self.SYSTEM_PROMPT), HumanMessage(content=message)]
        resp = self.llm.invoke(msgs)
        return resp.content.strip()

    def _parse(self, text: str) -> AgentAction | AgentFinish:
        if text.startswith("Action:"):
            lines = text.split("\n")
            tool = ""
            tool_input = ""
            for line in lines:
                if line.startswith("Action:"):
                    tool = line.split(":", 1)[1].strip()
                elif line.startswith("Action Input:"):
                    tool_input = line.split(":", 1)[1].strip()
            return AgentAction(tool, tool_input, log=text)
        elif text.startswith("Final Answer:"):
            answer = text.split(":", 1)[1].strip()
            return AgentFinish(result=answer, log=text)
        else:
            return AgentFinish(result=text, log=text)

    def _normalize(self, text: str) -> str:
        return text.strip().lower().replace(" ", "")

    def _exec_tool(self, action: AgentAction) -> str:
        for t in TOOLS:
            if t.name == action.tool:
                result = t.func(action.tool_input)
                print(f"[Tool最终返回] {repr(result)}")
                return result
        return f"工具 {action.tool} 未找到"


# ────────────────────────────────
# 3. 示例运行
# ────────────────────────────────
if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    agent = MiniAgent(llm)

    question = "特斯拉是哪个公司？"
    answer = agent.run(question)
    print("Q:", question)
    print("A:", answer)

    question2 = "Model 3 的续航多少？"
    answer2 = agent.run(question2)
    print("Q:", question2)
    print("A:", answer2)