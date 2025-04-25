# The project

This repository contains the LaTeX and Python sources for my final year dissertation project.

My project is a University Artificially Intelligent Chatbot.
This is the culmination of my extensive research into Large Language Models with a particular focus on Retrieval-Augmented Generation,
which was conducted alongside my other final year modules.

The project primarily uses LangChain, LangGraph and Streamlit at its core. LangChain is used as a wrapper for LLM interactions and PDF processing, 
with LangGraph dictating the chatbot's conditional functionality of deciding whether to use RAG for an answer. Streamlit acts as the chatbot's 
frontend, providing a simple, sleek and intuitive web-based UI.

The version in this repository is a PROTOTYPE, created for the purposes of the CMP6200 module. As explained in the dissertation itself (Chapter 6), 
with more time and funding this project could have been expanded upon further, with the most notable improvement being a domain to host the chatbot 
on. In this version, the chatbot runs locally.

# Running locally
**Running the chatbot locally requires you to have a system environment variable set titled `OPENAI_API_KEY` with a valid OpenAI API key stored in it. If you do not have one, please let me know before assuming the code doesn't work.**

This project has multiple dependencies before it can be run. These are reflected in the project's requirements.txt file to allow for the rapid 
creation of your own virtual environment with the necessary packages. Specifically, these dependencies are:
- Python 3.10
- FAISS-CPU >= 1.10.0
- Langchain >= 0.3.17
- langchain-community >= 0.3.16
- langchain-openai >= 0.3.3
- langgraph >= 0.2.70
- openai >= 1.61.1
- pdfminer-six >= 20240706
- pypdf >= 5.2.0
- streamlit >= 1.42.2
- deepeval >= 2.6.4 **(Optional, only needed for the unrelated Evaluation script)**

It is worth noting that the package manager I used was UV rather than other conventional managers like Pip or Conda. I checked compatibility with 
Pip and confirmed that the project will build successfully using it, though it took around 5 minutes to install every package, unlike UV which 
does so within seconds. I have not tested with Conda. Therefore, if issues should arise if you use one of these managers, I would recommend using UV 
as I have. As such, both a `requirements.txt` (for Pip usage) and a `pyproject.toml` (for UV usage) are present.

To install with UV, simply run `uv sync` from within the same directory as the pyproject.toml file.

To install with Pip, I would first recommend creating a virtual environment with `python -m venv .venv`, activating the environment, and then 
running `pip install -r requirements.txt` from within the same directory as that file.