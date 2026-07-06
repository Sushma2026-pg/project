import os
import requests
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from dotenv import load_dotenv
from lib.messages import (
    TokenUsage,
    AIMessage,
    BaseMessage,
    UserMessage,
)
from lib.tooling import Tool

# Load environment variables from .env
load_dotenv()


class LLM:
    def __init__(
        self,
        model: str = "tavily-search",
        temperature: float = 0.0,
        tools: Optional[List[Tool]] = None,
        api_key: Optional[str] = None
    ):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables")

        # Tavily API endpoint
        self.base_url = "https://api.tavily.com/search"
        self.tools: Dict[str, Tool] = {
            tool.name: tool for tool in (tools or [])
        }

    def register_tool(self, tool: Tool):
        self.tools[tool.name] = tool

    def _convert_input(self, input: Any) -> List[BaseMessage]:
        if isinstance(input, str):
            return [UserMessage(content=input)]
        elif isinstance(input, BaseMessage):
            return [input]
        elif isinstance(input, list) and all(isinstance(m, BaseMessage) for m in input):
            return input
        else:
            raise ValueError(f"Invalid input type {type(input)}.")

    def invoke(self, input: str | BaseMessage | List[BaseMessage]) -> AIMessage:
        messages = self._convert_input(input)
        query = messages[0].content  # Tavily only needs the query string

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

        return AIMessage(
            content=answer,
            tool_calls=[],
            token_usage=None
        )


# Example usage
if __name__ == "__main__":
    llm = LLM()  # will automatically pick up TAVILY_API_KEY from .env
    result = llm.invoke("When was Pokémon Gold and Silver released?")
    print("Answer:", result.content)
