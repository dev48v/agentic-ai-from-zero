"""Framework adapters.

Each adapter is imported explicitly and imports its framework lazily, so `agentfuse` stays
dependency-free no matter which of these exist:

    agentfuse.adapters.openai_tools     any OpenAI-compatible endpoint (NIM, Groq, OpenAI…)
    agentfuse.adapters.langgraph_guard  a drop-in wrapper for LangGraph's ToolNode
"""

__all__ = ["openai_tools", "langgraph_guard"]
