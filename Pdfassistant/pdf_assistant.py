import os
import uuid

from dotenv import load_dotenv
import typer
from typing import Optional, List

# Core agentic AI imports (names follow those spoken/shown in the video)
from phi.assistant import Assistant  # or: from phi import assistant as Assistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase

from phi.vectordb.pgvector import PgVector2
from dotenv import load_dotenv

load_dotenv()


# Configure your Groq / Grok key (the video uses GROQ_API_KEY instead of OPENAI_API_KEY)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Please set GROQ_API_KEY in your environment.")

# DB URL copied from the Docker command in the video (adjust user/pass/host/port if needed)
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector2(collection="recipes", db_url=db_url),
)

# Load the knowledge base into the vector DB and configure assistant storage
knowledge_base.load()

storage=PgAssistantStorage(
        table_name="pdf_assistant",
        db_url=db_url,
    )

def pdf_assistant(new: bool = False, user: str = "user"):
    run_id: Optional[str] = None
    """
    CLI entrypoint for the PDF assistant.

    - Connects to PGVector via DB_URL
    - Uses the PDFURLKnowledgeBase for retrieval
    - Persists runs and chat history via PGAssistantStorage
    """
    # Create the assistant instance

    if not new:
        existing_run_ids: List[str] = storage.get_all_run_ids(user)
        if len(existing_run_ids) > 0:
            run_id = existing_run_ids[0]  # Continue the latest run



    run_id = str(uuid.uuid4())
    assistant = Assistant(
        run_id=run_id,                # first run; will be filled after start
        user_id=user,
        knowledge_base=knowledge_base,
        storage=PgAssistantStorage(
            table_name="pdf_assistant",
            db_url=db_url,
        ),
        show_tools_in_core_call=True,
        search_knowledge=True,
        read_chat_history=True,
    )

    if run_id is None:
        assistant.run_id = run_id
        print(f"Starting a new run {run_id}\n")        
    else:
        print(f"Continuing run {run_id}\n")

    # Start the run, get a run_id from the assistant
    # run = assistant.start()
    # assistant.run_id = run.id

    # Simple CLI loop using typer + rich/markdown rendering (as in the video)
    assistant.cli_app(markdown=True, stream=True)


app = typer.Typer()


@app.command()
def run(
    new: bool = typer.Option(
        False, "--new", help="Start a new run instead of continuing previous one."
    ),
    user: str = typer.Option("user", "--user", help="User identifier."),
):
    pdf_assistant(new=new, user=user)


if __name__ == "__main__":
    app()
