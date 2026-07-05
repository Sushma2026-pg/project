import os
import requests
from typing import List, Dict, Any
from lib.messages import BaseMessage, UserMessage, AIMessage, TokenUsage
from lib.tooling import Tool

class LLM:
    def __init__(self, model="tavily-search", temperature=0.7, api_key: str | None = None):
        self.model = model
        self.temperature = temperature
        # Allow explicit api_key or fallback to environment
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables")

        # Tavily API endpoint
        self.base_url = "https://api.tavily.com/search"

        # Tools registry (optional)
        self.tools: Dict[str, Tool] = {}


    def register_tool(self, tool: Tool):
        """Register a tool so the agent can call it."""
        self.tools[tool.name] = tool

    def _convert_input(self, input: Any) -> List[BaseMessage]:
        """Normalize input into a list of BaseMessage objects."""
        if isinstance(input, str):
            return [UserMessage(content=input)]
        elif isinstance(input, BaseMessage):
            return [input]
        elif isinstance(input, list) and all(isinstance(m, BaseMessage) for m in input):
            return input
        else:
            raise ValueError(f"Invalid input type {type(input)}.")

    def invoke(self, input: str | BaseMessage | List[BaseMessage]) -> AIMessage:
        """Send a query to Tavily API and wrap the response as an AIMessage."""
        messages = self._convert_input(input)
        # For simplicity, just use the last user message as the query
        query = messages[-1].content

        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "query": query,
            "search_depth": "advanced",
            "include_answer": True,
            "include_raw_content": False,
            "include_images": False
        }

        response = requests.post(self.base_url, json=payload, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(f"Tavily API error {response.status_code}: {response.text}")

        data = response.json()
        answer = data.get("answer", "")
        results = data.get("results", [])

        return AIMessage(
            content=answer,
            tool_calls=None,
            token_usage=TokenUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0
            )
        )
