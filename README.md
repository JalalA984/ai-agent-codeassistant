# Coding Assistant AI-Agent

## Overview

This Coding Assistant AI-Agent is a powerful local AI tool designed to assist developers in their coding tasks. It leverages advanced language models and vector indexing to provide intelligent code generation, documentation lookup, and contextual assistance.

## Features

- Local AI processing using Ollama with Mistral model
- Document parsing and indexing with LlamaIndex framework
- Code generation with CodeLlama model
- API documentation lookup
- Intelligent code reading and analysis
- Interactive command-line interface

## Tech Stack

- Python 3.11.9
- Ollama (for running Mistral locally)
- LlamaIndex (for data loading and processing)
- LlamaCloud and LlamaParse (for enhanced parsing capabilities)
- BAAI/bge-m3 (for embeddings)
- CodeLlama (for code-specific language processing)

## Prerequisites

- Python 3.11.9
- Visual Studio C/C++ (for CMake tools required by some dependencies)
- Ollama installed and configured

## Installation

1. Clone the repository:
   ```
   git clone git@github.com:JalalA984/ai-agent-codeassistant.git

   cd ai-agent-codeassistant
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables in a `.env` file.

## Usage

Run the main script to start the AI agent:

```
python main.py
```

Enter your prompts when prompted. The agent will process your request, generate code, and save the output to the `output` directory.

## Key Components

- `Ollama`: Runs the Mistral and CodeLlama models locally
- `LlamaParse`: Parses input documents into a structured format
- `VectorStoreIndex`: Indexes and enables efficient querying of document content
- `ReActAgent`: Manages the interaction between different tools and models
- `QueryPipeline`: Processes user input through a series of transformations

## File Structure

- `main.py`: The main script that runs the AI agent
- `code_reader.py`: Contains the code reading and analysis functionality
- `prompts.py`: Stores prompt templates used by the agent
- `data/`: Directory for storing input documents
- `output/`: Directory where generated code files are saved
