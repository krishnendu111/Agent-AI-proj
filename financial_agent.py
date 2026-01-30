import os
import openai
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv

# Load environment variables (Make sure GROQ_API_KEY is in your .env)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

##os.environ['HF_TOKEN'] = ''
###login(token=os.environ['HF_TOKEN'])

# We combine tools into one Multi-Purpose Agent to avoid "transfer_task" errors in Groq
financial_ai_agent = Agent(
    name="Financial AI Agent",
    ###model=Groq(id="meta-llama/Llama-3.3-70B-Instruct"),  # Changed to a model optimized for tool calling
    tools=[
        DuckDuckGo(),
        YFinanceTools(
            stock_price=True, 
            analyst_recommendations=True, 
            stock_fundamentals=True, 
            company_news=True
        ),
    ],
    instructions=[
        "Use DuckDuckGo to search for the latest news and market sentiment.",
        "Use YFinanceTools to get stock prices, analyst recommendations, and fundamentals.",
        "Always include sources and links for news items.",
        "Use tables to display financial data and comparisons.",
        "If you cannot find specific data, state it clearly rather than guessing.",
        "Format your final response in clear Markdown."
    ],
    show_tool_calls=True,
    markdown=True,
)

def start_terminal_app():
    print("====================================================")
    print("📊 Financial AI Agent (Powered by Groq & Phidata)")
    print("Type 'exit' or 'quit' to stop the program.")
    print("====================================================\n")

    while True:
        # Prompt user for company name
        company = input("Enter the stock/company name: ").strip()

        # Exit condition
        if company.lower() in ['exit', 'quit']:
            print("Shutting down... Goodbye!")
            break

        if not company:
            continue

        print(f"▰▰▰▰▰▰▱ Thinking about {company}...")
        
        prompt = (
            f"Summarize analyst recommendations for {company} and find "
            f"recent news from the last 30 days that could impact their stock prices."
        )

        try:
            # Execute the agent
            financial_ai_agent.print_response(prompt, stream=True)
            
        except Exception as e:
            # Catch Groq-specific tool errors
            if "Failed to call a function" in str(e):
                print("\n[System Note]: Groq had trouble with the tool call. Trying one more time with a simpler request...")
                financial_ai_agent.print_response(f"Show me the stock price and latest news for {company}", stream=True)
            else:
                print(f"\n[Error]: {e}")

if __name__ == "__main__":
    start_terminal_app()