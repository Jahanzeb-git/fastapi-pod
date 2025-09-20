import json
import logging
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

logger = logging.getLogger(__name__)

from app.config import settings

class OpenAILLMClient:
    """Async wrapper for OpenAI GPT-5 API"""
    
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = settings.OPENAI_MODEL
    
    async def create_completion(
        self,
        messages: List[BaseMessage],
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000
    ) -> AIMessage:
        """Create async completion with GPT-5"""
        
        # Convert LangChain messages to OpenAI format
        openai_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                openai_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                openai_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                openai_messages.append({"role": "assistant", "content": msg.content})
        
        try:
            # Prepare the request parameters
            request_params = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Add tools if provided
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"
            
            # Make async call to OpenAI
            response = await self.client.chat.completions.create(**request_params)
            
            # Convert response to AIMessage
            message = response.choices[0].message
            
            # Handle tool calls if present
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_calls = []
                for tc in message.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "args": json.loads(tc.function.arguments)
                    })
                return AIMessage(content=message.content or "", tool_calls=tool_calls)
            else:
                return AIMessage(content=message.content)
                
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise