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

It is worth noting that the package manager used was UV rather than other conventional managers like Pip or Conda. 
Therefore, if issues should arise if you use one of these managers, I would recommend using UV as I have. As such, both a requirements.txt and a 
pyproject.toml are present.