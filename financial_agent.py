# import os
# import openai
# from flask import Flask, render_template, request, jsonify
# from phi.agent import Agent
# from phi.model.groq import Groq
# from phi.tools.yfinance import YFinanceTools
# from phi.tools.duckduckgo import DuckDuckGo
# from dotenv import load_dotenv
# import markdown2
# import yfinance as yf

# load_dotenv()

# app = Flask(__name__)

# Load environment variables (Make sure GROQ_API_KEY is in your .env)

import os
from flask import Flask, render_template, request, jsonify
from phi.agent import Agent
from phi.model.openai import OpenAIChat # Changed to OpenAI
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import markdown2
import time

load_dotenv()

app = Flask(__name__)

# Ensure OPENAI_API_KEY is in your .env
openai_key = os.getenv("OPENAI_API_KEY")

# openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the Agent with OpenAI
financial_ai_agent = Agent(
    name="Financial AI Agent",
    model=OpenAIChat(id="gpt-4o"),
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
    markdown=True,
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_stock', methods=['POST'])
def get_stock():
    company = request.form.get('company')
    if not company:
        return jsonify({"error": "Please enter a stock name"})

    prompt = (f"Provide a financial summary and news for {company}. "
              f"Use tables for data.")

    start_time = time.time()
    
    try:
        # Use stream=False to ensure we get a RunResponse object, not a generator
        run_response = financial_ai_agent.run(prompt, stream=False)
        
        # Verify we actually got content back
        if run_response and hasattr(run_response, 'content'):
            content = run_response.content
        else:
            # Fallback if the response structure is unexpected
            content = str(run_response)

        # Convert to HTML
        html_output = markdown2.markdown(content, extras=["tables", "fenced-code-blocks"])
        return jsonify({"output": html_output})

    except Exception as e:
        print(f"Detailed Error: {e}") # This helps you debug in the terminal
        return jsonify({"error": f"Agent Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)