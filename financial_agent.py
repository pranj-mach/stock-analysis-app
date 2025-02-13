from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

# Web Search Agent
web_search_agent = Agent(
    name="Web_Agent",
    role="Find stock-related information from the internet, including recent news and earnings reports.",
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
    role="Analyze stock performance, trends, and fundamental metrics.",
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True,
        ),
    ],
    instructions=[
        "Use bullet points for clarity.",
        "Include key financial ratios.",
        "Cross-check financial data before summarizing."
    ],
    show_tool_calls=True,
    markdown=True,
)

# Multi-Agent Orchestrator
multi_ai_agent = Agent(
    name="Orchestrator",
    team=[web_search_agent, finance_agent],
    model=Groq(id="llama-3.1-8b-instant"),
    instructions=[
        "First, ask Web_Agent for recent stock news and sentiment.",
        "Then, pass relevant findings to Finance_Agent for deeper analysis.",
        "Ensure responses are structured and include sources.",
        "Use bullet points for clarity."
    ],
    show_tool_calls=True,
    markdown=True,
)

# Run the Multi-Agent System
multi_ai_agent.print_response("Summarize analyst recommendations for NVDA", stream=True)
