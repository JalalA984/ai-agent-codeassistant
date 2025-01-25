# Imports
from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from prompts import context
from dotenv import load_dotenv
import logging
from retrying import retry
from code_reader import code_reader

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Retry logic for handling intermittent timeout errors
@retry(stop_max_attempt_number=3, wait_fixed=2000)  # Retry up to 3 times, wait 2 seconds between attempts
def query_with_retry(agent, prompt):
    return agent.query(prompt)

# Configure the LLMs with a higher timeout
llm = Ollama(model="mistral", request_timeout=60.0)  # Increased timeout
llm_code = Ollama(model="codellama", request_timeout=60.0)

# Document parser and loader
parser = LlamaParse(result_type="markdown")
file_extractor = {".pdf": parser}

# Load and verify documents
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()
logging.info(f"Loaded {len(documents)} documents.")

# Create embeddings and vector store index
embedding_model = resolve_embed_model("local:BAAI/bge-m3")
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embedding_model)
query_engine = vector_index.as_query_engine(llm=llm)

# Define tools for the agent
tools = {
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="api_documentation",
            description="Provides documentation about code for an API. Use this to read API documentation."
        ),
    ),
    code_reader,
}

# Create the ReAct agent
agent = ReActAgent.from_tools(tools, llm=llm_code, verbose=True, context=context)

# Main interaction loop
while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    try:
        result = query_with_retry(agent, prompt)  # Use retry logic
        print(result)
    except Exception as e:
        logging.error(f"Error processing the prompt: {e}")
        print("An error occurred. Please try again.")
