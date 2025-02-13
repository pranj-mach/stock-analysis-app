import streamlit as st
import os
import requests
import re  # ✅ Used for extracting stock prices
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import phi
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.playground import Playground, serve_playground_app

# ✅ Load environment variables
load_dotenv()
phi.pi = os.getenv("PHI_API_KEY")

# ✅ ThreadPoolExecutor for async execution
executor = ThreadPoolExecutor()

# ✅ Function to fetch live USD to INR conversion rate
def get_usd_to_inr():
    try:
        url = "https://api.exchangerate-api.com/v4/latest/USD"  
        response = requests.get(url).json()
        return response["rates"].get("INR", 87.4)  # ✅ Default to 83 INR if API fails
    except Exception as e:
        st.error(f"⚠️ Error fetching exchange rate: {e}")
        return 87.4  # ✅ Fallback exchange rate

# ✅ Function to format stock symbols (e.g., INFY -> INFY.NS)
def format_stock_symbol(symbol):
    if not symbol.endswith(".NS") and not symbol.endswith(".BO"):
        return symbol + ".NS"  # ✅ Default to NSE if no suffix
    return symbol

# ✅ Finance Analysis Agent (Yahoo Finance)

finance_agent = Agent(
    name="Finance_Agent",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
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

# ✅ Web Search Agent (DuckDuckGo)
web_search_agent = Agent(
    name="Web_Agent",
    role="Find stock-related news and recent updates from the internet.",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[DuckDuckGo()],
    instructions=[
        "Always include sources.",
        "Prefer official reports over general news.",
        "Summarize findings concisely."
    ],
    show_tool_calls=True,
    markdown=True,
)

# ✅ Multi-Agent Playground
app = Playground(agents=[finance_agent, web_search_agent]).get_app()

# ✅ Function to get stock analysis
def get_stock_analysis(symbol):
    response = finance_agent.run(symbol)
    return response.content

# ✅ Function to fetch latest news using Web Search Agent
def get_latest_news(query):
    response = web_search_agent.run(query)
    return response.content

# ✅ Function to extract and convert stock price to INR (if needed)
def extract_stock_price_and_convert(result, exchange_rate, stock_symbol=""):
    stock_data = {}
    lines = result.split("\n")

    for line in lines:
        line = line.strip()
        if "Current Price" in line or "Stock Price" in line:
            try:
                # ✅ Extract numeric value using regex
                price_match = re.search(r"([\d,]+\.\d+)", line)
                if price_match:
                    price_value = float(price_match.group(1).replace(",", ""))  # ✅ Remove commas if any

                    # ✅ Detect stock exchange from symbol (NSE/BSE = INR, NYSE/NASDAQ = USD)
                    if stock_symbol and (stock_symbol.endswith(".NS") or stock_symbol.endswith(".BO")):
                        # ✅ Indian stock, no conversion
                        stock_data["Stock Price (INR)"] = f"₹{price_value}"
                    else:
                        # ✅ US stock (or unknown), convert to INR
                        price_inr = round(price_value * exchange_rate, 2)
                        stock_data["Stock Price (INR)"] = f"₹{price_inr}"
                        stock_data["Stock Price (USD)"] = f"${price_value}"
                else:
                    raise ValueError("Price not found in expected format")
            except Exception as e:
                st.error(f"⚠️ Error extracting stock price: {e}")
                stock_data["Stock Price"] = "N/A"

    return stock_data

# ✅ Streamlit UI
st.title("📊 AI-Powered India Stock & News Dashboard ")

# ✅ Sidebar for User Selection
st.sidebar.header("Select an Option")
query_type = st.sidebar.radio("What do you want?", ["Stock Analysis", "Latest News"])

# ✅ User Input Fields
if query_type == "Stock Analysis":
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., INFY, TCS)")
elif query_type == "Latest News":
    search_query = st.text_input("Enter a topic (e.g., TataMotors stock news)")

# ✅ Async Function to Run Agents
async def run_agent(agent, input_text):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, agent.run, input_text)

# ✅ Process User Request
if st.button("Analyze"):
    if query_type == "Stock Analysis" and stock_symbol:
        formatted_symbol = format_stock_symbol(stock_symbol)  # ✅ Ensure correct symbol format
        st.write(f"🔍 Fetching data for: **{formatted_symbol}**")

        with st.spinner("Fetching data..."):
            raw_result = get_stock_analysis(formatted_symbol)

            if not raw_result or "error" in raw_result.lower():
                st.error("⚠️ Could not fetch data. Please check the stock symbol.")
            else:
                exchange_rate = get_usd_to_inr()  # ✅ Fetch live exchange rate
                stock_prices = extract_stock_price_and_convert(raw_result, exchange_rate, formatted_symbol)  # ✅ FIXED

                # ✅ Display Stock Prices
                if stock_prices and "Stock Price (INR)" in stock_prices:
                    st.subheader("💰 Stock Price")
                    for key, value in stock_prices.items():
                        st.write(f"- **{key}**: {value}")  # ✅ Proper bullet points
                else:
                    st.error("⚠️ Stock price not available. Try a different stock symbol.")

                # ✅ Display Stock Analysis in Bullet Points
                st.subheader("📊 Stock Analysis Summary")
                st.markdown(raw_result)  # ✅ Ensure markdown rendering for bullet points

    elif query_type == "Latest News" and search_query:
        st.write(f"🔍 Searching news for: **{search_query}**")
        with st.spinner("Fetching news..."):
            result = get_latest_news(search_query)
            st.markdown(result)  # ✅ Display AI response
    else:
        st.error("⚠️ Please enter a valid input.")

# ✅ Async Function to Run Playground (Like Your Original Code)
async def main():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, serve_playground_app, "playground:app")

if __name__ == "__main__":
    asyncio.run(main())