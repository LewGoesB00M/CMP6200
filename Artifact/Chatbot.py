# Used to fetch system environment variables.
import os

# Initialises the LLM.
from langchain.chat_models import init_chat_model

# The vector DB used for this prototype. It's worked well this far,
# so I'll likely use it in future versions, too.
from langchain_community.vectorstores import FAISS

# Allows for inputs to be sent to the LLM as the 
# system (specialised prompt that defines LLM behaviour).
from langchain_core.messages import SystemMessage

# Allows for "tools" to be created. The LLM can use these tools
# to execute defined code, such as retrieving data from the FAISS DB.
from langchain_core.tools import tool

# Used to embed queries to the FAISS DB.
from langchain_openai import OpenAIEmbeddings

# LangGraph key functionality. Declares a clear and consistent 
# structure for the chatbot, and also provides functionality for 
# conditional actions. This is described later in the code.
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# Sets the directory of the FAISS DB that's being loaded from.
    # Options (all begin with "VectorStores/"):
        #   FAISS: Chunk size 1000, Overlap 200, PyPDFLoader with default args. Poor results (40%)
        #   FAISS-SmallChunks: Chunk size 500, Overlap 100, PyPDFLoader with default args. Worst results. (20%)
        #   FAISS-BigChunks: Chunk size 1500, Overlap 300, PyPDFLoader with default args. Poor results (40%)
        #   FAISS-HugeChunks: Chunk size 2000, Overlap 500, PyPDFLoader with default args. Best results (80%), will be used for demo.
dbPath = "VectorStores/FAISS-HugeChunks"


# Sets up the embedding model with the API key.
embedder = OpenAIEmbeddings(
    model = "text-embedding-3-small",
    openai_api_key = os.environ["OPENAI_API_KEY"]
)

# Load the vector database.
db = FAISS.load_local(folder_path = dbPath,
                      embeddings = embedder,
                      allow_dangerous_deserialization=True)

# The FAISS DB is also stored as a serialized .pkl file. It's possible
# for these files to contain malicious code that would be executed on deserialization,
# hence "allow_dangerous_deserialization". However, I generated the files myself and know
# that they aren't malicious.


# Initialise the LLM.
# LangChain automatically interprets the LLM in question to be OpenAI's gpt-4o-mini simply by
# specifying its name as a string argument.
llm = init_chat_model("gpt-4o-mini", temperature = 0,
                      openai_api_key = os.environ["OPENAI_API_KEY"])


@tool(response_format = "content")
def retrieve(query):
    # This docstring is used as the context for the LLM, letting it know what the tool does.         
    """Retrieves the 3 most relevant context chunks for a given query.
    
    Args:
        query: The user's question, optimized for a semantic search."""
    
    retrievedChunks = db.similarity_search(query, k = 3)
    
    # Each retrieved chunk is seperated by two newlines.
    content = "\n\n".join(
        (f"{chunk.page_content}")
        for chunk in retrievedChunks
    )
    
    return content

# Creating a node for the list of tools that the LLM can access. 
# At the moment, it's only one tool, the retriever, but other tools can 
# easily be added later by appending them to this list.
tools = ToolNode([retrieve])

def query_or_respond(state: MessagesState):
    # Creates the retrieval agent by giving the LLM access to the retrieval tool.
    retrievalAgent = llm.bind_tools([retrieve])
    
    # The LLM decides on its own if it needs retrieval based on the existing conversation.
    response = retrievalAgent.invoke(state["messages"])
    
    # If it wants to use a tool, it will return a blank message with the metadata requesting a tool call.
    # Otherwise, it will return a generic message without any context, which occurs if the information is
    # already known or the query is simply too general ("Hello", for example).
    return {"messages": [response]}



# The final step of the process, generating a message based on the info gathered
# from the retrieval tool.
def generate(state: MessagesState):
    # Retrieves the most recent tool call from the tools node.
    recentToolMsgs = []
    
    # The MessagesState stores the most recent messages at the bottom, as it's an append-only list.
    # This means that it'll need to be reversed for the most recent messages.
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recentToolMsgs.append(message)
        else:
            # If it's a normal message, stop.
            # This is because the context from earlier messages doesn't need
            # repeating again, as it would enormously increase token usage and therefore cost.
            break

    # Saves the context from the retriever tool.
    docsContent = "\n\n".join(doc.content for doc in recentToolMsgs)
    
    # This is the LLM's system prompt, which decides how the LLM behaves.
    systemPrompt = f"""
       You are a friendly assistant to help new students get acclimated to Birmingham City University.
       If you don't know the answer, say that you don't know. 
       When referring to context, be specific and quote the context. 
       Use five sentences maximum and keep the answer concise. 
       Use the following pieces of retrieved context to answer the question.
       \n\n
       Context: {docsContent} 
    """ 
    
    # The list of messages in the conversation.
    # Only adds messages that AREN'T tool calls, as tool calls are blank messages.
    conversation = [
        message for message in state["messages"] # Every message in the conversation
        if message.type in ("human", "system") # If it's human input or the system prompt
        or (message.type == "ai" and not message.tool_calls) # Or from the LLM and isn't a tool call.
    ]
    
    # It's important to note that the conversation isn't a list of strings; it's a list 
    # of LangChain AI/HumanMessages. This means that when the LLM is invoked, it's being 
    # invoked with the entire conversation of messages, not just one prompt containing them all
    # as a big block of text.
    
    # The LLM is given its system prompt (containing current retrieved context if there is any)
    # alongside all other messages in the conversation.
    history = [SystemMessage(systemPrompt)] + conversation
    

    # Get the LLM's response to the prompt and return the response.
    response = llm.invoke(history)
    return {"messages": [response]}


# Initialise the graph.
graph = StateGraph(MessagesState)

# Add all the nodes.
graph.add_node(query_or_respond)
graph.add_node(tools)
graph.add_node(generate)

# The graph starts with choosing whether to query the DB or directly respond.
graph.set_entry_point("query_or_respond")

# query_or_respond has conditions:
# If the user's query needs RAG context, the retrieve tool will be called.
# If it does not, the LLM will generate a response by itself.

# "What will my grade be reduced by if I submit 3 days late?" invokes the retrieval tool. 
# "Hello!" should not.
graph.add_conditional_edges(
    "query_or_respond", tools_condition, 
    # tools_condition is True if the Agent wants to use a tool.
    # If retrieval is not needed, skip to the end, which generates a general non-BCU related answer.
    # If retrieval is needed, call the retrieval tool.
    {END: END, 
     "tools": "tools"},
)

# An edge is also needed between the tool call and response generation
# to ensure the response has the RAG context.
graph.add_edge("tools", "generate")

# After the response is generated, the graph is done.
graph.add_edge("generate", END)

# Compile the graph so Streamlit can use it.
graph = graph.compile()

