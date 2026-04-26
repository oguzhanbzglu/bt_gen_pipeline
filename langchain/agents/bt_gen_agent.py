import re
import sys
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import operator

PACKAGE_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PACKAGE_ROOT.parent))
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT / "langchain" / "tools"))

from generate_and_execute_tool import generate_and_execute

CHAT_MODEL = "qwen2.5:14b"
ALL_TOOLS = [generate_and_execute]

_llm = ChatOllama(model=CHAT_MODEL, temperature=0)
_llm_with_tools = _llm.bind_tools(ALL_TOOLS)
_tool_node = ToolNode(ALL_TOOLS)


def _extract_yaml_path(messages: list) -> str:
    for msg in messages:
        if isinstance(msg, HumanMessage):
            match = re.search(r'(/[^\s]+\.ya?ml)', msg.content)
            if match:
                return match.group(1)
    return ""


def _build_system_prompt(yaml_path: str) -> SystemMessage:
    p = yaml_path or "<not yet known>"
    return SystemMessage(content=(
        f"You are a robotic BT execution assistant.\n\n"
        f"YAML file for this session: {p}\n\n"
        f"Call generate_and_execute(yaml_path='{p}') to generate the BT and run it on the robot.\n"
        f"The tool handles XML generation, execution, collision recovery, and retries automatically.\n"
        f"Do not call any other tools. Report the result when done."
    ))


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    finished: bool
    yaml_path: str


def llm_node(state: AgentState) -> AgentState:
    messages = list(state["messages"])
    yaml_path = state.get("yaml_path", "")

    if not yaml_path:
        yaml_path = _extract_yaml_path(messages)

    if messages and isinstance(messages[0], SystemMessage):
        messages[0] = _build_system_prompt(yaml_path)
    else:
        messages = [_build_system_prompt(yaml_path)] + messages

    response = _llm_with_tools.invoke(messages)
    return {
        "messages": [response],
        "finished": state.get("finished", False),
        "yaml_path": yaml_path,
    }


def tools_node(state: AgentState) -> AgentState:
    result = _tool_node.invoke(state)
    finished = state.get("finished", False)

    for msg in result.get("messages", []):
        if getattr(msg, "name", "") == "generate_and_execute":
            content = str(getattr(msg, "content", ""))
            if '"success": true' in content or "'success': True" in content:
                finished = True
                print("[Agent] Task completed successfully.")

    return {
        "messages": result.get("messages", []),
        "finished": finished,
        "yaml_path": state.get("yaml_path", ""),
    }


def router(state: AgentState) -> Literal["tools", "end"]:
    if state.get("finished", False):
        return "end"
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "end"


def create_agent():
    builder = StateGraph(AgentState)
    builder.add_node("llm", llm_node)
    builder.add_node("tools", tools_node)
    builder.add_edge(START, "llm")
    builder.add_edge("tools", "llm")
    builder.add_conditional_edges("llm", router, {"tools": "tools", "end": END})
    return builder.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    graph = create_agent()
    state: AgentState = {"messages": [], "finished": False, "yaml_path": ""}

    print(f"BT Gen Agent (model: {CHAT_MODEL})")
    print('Type an instruction or "quit".\n')

    while True:
        user_input = input("> ")
        if user_input.lower() == "quit":
            break

        state["messages"].append(HumanMessage(content=user_input))
        state = graph.invoke(state, config=config)

        last = state["messages"][-1]
        print(getattr(last, "content", str(last)))
