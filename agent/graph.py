from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, ToolMessage
from typing import TypedDict, Annotated, List
from uuid import UUID
import json
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from langgraph.graph.message import add_messages

from app.database import get_db, User
from core.session_manager import SessionManager
from agent.prompts import AgentState, get_system_prompt
from agent.tools import AVAILABLE_TOOLS
from agent.llm_client import OpenAILLMClient
from app.config import settings

logger = logging.getLogger(__name__)

llm_client = OpenAILLMClient(api_key=settings.OPENAI_API_KEY)

# State definition for LangGraph
class AgentGraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: str
    db: AsyncSession

# Custom Tool Node
class CustomToolNode(ToolNode):
    async def ainvoke(self, state: AgentGraphState, *args, **kwargs) -> AgentGraphState:
        user_id = state['user_id']
        db = state['db']
        
        tool_messages = []
        for tool_call in state['messages'][-1].tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            # Inject user_id and db into tool args
            tool_args['user_id'] = user_id
            tool_args['db'] = db
            
            # Find the tool to execute
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if not tool:
                raise ValueError(f"Tool '{tool_name}' not found.")

            try:
                # Execute the tool with the modified arguments
                result = await tool.ainvoke(tool_args)
                tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call['id']))
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                tool_messages.append(ToolMessage(content=f"Error: {e}", tool_call_id=tool_call['id']))

        return {"messages": tool_messages}


# Graph Nodes
async def agent_node(state: AgentGraphState) -> AgentGraphState:
    """Main agent processing node"""
    user_id = UUID(state['user_id'])
    db = state['db']
    session_context = await SessionManager.get_session_context_for_agent(db, user_id)

    system_prompt = get_system_prompt(
        state=session_context["agent_state"],
        product_context=session_context["product_context"],
        design_url=session_context["design_urls"].get("design_url") if session_context["design_urls"] else None,
        upscaled_url=session_context["design_urls"].get("upscaled_url") if session_context["design_urls"] else None,
        orders_history=session_context["orders_history"]
    )

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    ai_message = await llm_client.create_completion(
        messages=messages,
        tools=[t.schema() for t in AVAILABLE_TOOLS],
        temperature=0.3,
        max_tokens=1000
    )

    return {"messages": [ai_message]}

async def update_state_node(state: AgentGraphState) -> AgentGraphState:
    """Update database state based on tool calls"""
    user_id = UUID(state['user_id'])
    db = state['db']
    
    # Check if there are tool messages (responses from tools)
    if len(state['messages']) >= 2:
        # Get the last AI message that triggered the tool
        for msg in reversed(state['messages'][:-1]):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call['name']
                    
                    # Get corresponding tool response
                    last_message = state['messages'][-1]
                    if isinstance(last_message, ToolMessage):
                        try:
                            tool_output = json.loads(last_message.content)
                            
                            if tool_name == 'create_design':
                                await SessionManager.update_design_urls(
                                    db, user_id, 
                                    design_url=tool_output.get('design_url')
                                )
                                await SessionManager.update_session_state(
                                    db, user_id, 
                                    AgentState.DESIGN_CREATED.value
                                )
                            elif tool_name == 'upscaling':
                                await SessionManager.update_design_urls(
                                    db, user_id, 
                                    upscaled_url=tool_output.get('upscaled_url')
                                )
                            elif tool_name == 'place_order':
                                # Don't clear session here - let the order endpoint handle it
                                pass
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse tool output: {last_message.content}")
                break
    
    return state

# Router function
def should_continue(state: AgentGraphState) -> str:
    return "tools" if isinstance(state["messages"][-1], AIMessage) and state["messages"][-1].tool_calls else END

# Build the graph
workflow = StateGraph(AgentGraphState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", CustomToolNode(AVAILABLE_TOOLS))
workflow.add_node("update_state", update_state_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "update_state")
workflow.add_edge("update_state", "agent")

app_graph = workflow.compile()