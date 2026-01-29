from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os
import openai

from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# -------------------------------
# Take stock name from terminal
# -------------------------------
stock_name = input("Enter the stock/company name: ").strip()

## Web Search Agent
web_search_agent = Agent(
    name="WebSearchAgent",
    role="Search the web for information using DuckDuckGo and analyze financial data using yFinance.",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions="""Always include sources.""",
    show_tool_calls=True,
    markdown=True,
)

## FInancial Analysis Agent
financial_agent = Agent(
    name="Financial AI Analysis Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True,
                         stock_fundamentals=True, company_news=True)],
    instructions="""Use tables to display the data""",
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent = Agent(
    team=[web_search_agent, financial_agent],
    # model=Groq(model="llama-3.3-70b-versatile"),
    instructions="Always include sources. Use tables to display the data." ,
    show_tool_calls=True,
    markdown=True,
)

##multi_ai_agent.print_response("Summarize analyst recommendations
## for Microsoft and find recent news that could impact their stock prices.",
##stream=True)

# -------------------------------
# Dynamic prompt using user input
# -------------------------------
prompt = (
    "Summarize analyst recommendations for {} and find recent news that could impact their stock prices."
).format(stock_name)

multi_ai_agent.print_response(prompt, stream=True)