from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import phi.api
from dotenv import load_dotenv
import os
import phi
from phi.playground import Playground, serve_playground_app
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()
phi.pi = os.getenv("PHI_API_KEY")

executor = ThreadPoolExecutor()

# Web Search Agent
web_search_agent = Agent(
    name="Web_Agent",
    role="Find stock-related news and recent updates from the internet.",
    model=Groq(id="llama-3.1-8b-instant"),
    tools=[DuckDuckGo()],
    instructions=[
        "Always include sources.",
        "Prefer official reports over general news.",
        "Summarize findings concisely."
    ],
    show_tool_calls=True,
    markdown=True,
)

# Finance Analysis Agent
finance_agent = Agent(
    name="Finance_Agent",
    model=Groq(id="llama-3.1-8b-instant"),
    role="Analyze stock performance, trends, fundamental metrics, and SEC Filings.",
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True
        )
    ],
    instructions=[
        "Use bullet points for clarity.",
        "Include key financial ratios.",
        "Cross-check financial data before summarizing."
    ],
    show_tool_calls=True,
    markdown=True,
)

# Multi-Agent Playground
app = Playground(agents=[finance_agent, web_search_agent]).get_app()

async def main():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, serve_playground_app, "playground:app")

if __name__ == "__main__":
    asyncio.run(main())

