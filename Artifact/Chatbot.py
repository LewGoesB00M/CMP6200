# Used to get the OpenAI API key from the system environment variables.
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
        #   FAISS: Chunk size 1000, Overlap 200, PyPDFLoader with default args.
        #   FAISS-SmallChunks: Chunk size 500, Overlap 100, PyPDFLoader with default args.
        #   FAISS-Unstructured: Chunk size 1000, Overlap 200, UnstructuredPDFLoader with default args.
        #   FAISS-BigChunks: Chunk size 1500, Overlap 300, PyPDFLoader with default args.
        #   FAISS-HugeChunks: Chunk size 2000, Overlap 500, PyPDFLoader with default args.
FAISS_PATH = "VectorStores/FAISS-HugeChunks"


# Sets up the embedding model with the API key.
embedder = OpenAIEmbeddings(
    model = "text-embedding-3-small",
    
    # Get the OpenAI API key from environment variables so that 
    # it's not visible in this code on GitHub. OpenAI would revoke the key if it leaked.
    api_key = os.environ["OPENAI_API_KEY"]
)

# Load the vector database.
db = FAISS.load_local(folder_path = FAISS_PATH,
                      embeddings = embedder,
                      allow_dangerous_deserialization=True)

# The FAISS DB is also stored as a serialized .pkl file. It's possible
# for these files to contain malicious code that would be executed on deserialization,
# hence "allow_dangerous_deserialization". However, I generated the files myself and know
# that they aren't malicious.


# Initialise the model.
# Automatically uses OPENAI_API_KEY from environment vars, so not necessary to specify here.
llm = init_chat_model("gpt-4o-mini", temperature = 0.2)

@tool(response_format = "content")
def retrieve(query):
    # This docstring is used as the context for the LLM.         
    """Retrieves the 3 most relevant context chunks for the user's query."""
    
    # Other tested queries that didn't work as well:
        # """Performs a semantic search for the most relevant chunks to the user's query."""
        
        # """Generate a semantic search query and then retrieves the 3 most relevant
        # context chunks for the query. You do not need to specify 'BCU'."""
    
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
    # Allows the LLM to use the retrieve tool that was created earlier.
    retrievalAgent = llm.bind_tools([retrieve])
    
    # The LLM decides on its own if it needs retrieval.
    response = retrievalAgent.invoke(state["messages"])
    
    # A key element of using a MessagesState is that that will append
    # the response to the conversation history rather than overwriting.
    return {"messages": [response]}



# The final step of the process, generating a message based on the info gathered
# from the retrieval tool.
def generate(state: MessagesState):
    # To massively reduce token consumption (and therefore cost),
    # the most recent RAG context from tool calls is added to the 
    # prompt to stop the LLM searching the entire conversation history 
    # for something that was JUST said. 
    recentToolMsgs = []
    
    # To get the most recent ones, the list needs to be reversed
    # so that the most recent come first instead of last.
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recentToolMsgs.append(message)
        else:
            # If it's a normal message, stop.
            break
    
    # Put the tool messages in their original order in case 
    # the sequence of the retrieved context mattered. 
    toolMsgs = recentToolMsgs[::-1]

    # Saves the context from the recent tool messages.
    docsContent = "\n\n".join(doc.content for doc in toolMsgs)
    
    # This is the LLM's system prompt, which decides how the LLM behaves.
    systemPrompt = (
        "You are an assistant to help new students get acclimated to Birmingham City "
        "University. If you don't know the answer, say that you "
        "don't know. When referring to context, be specific and quote the context. "
        "You must never say 'the context', and should instead act like a friendly human. "
        "Use five sentences maximum and keep the answer concise. "
        "Use the following pieces of retrieved context to answer "
        "the question."
        "\n\n"
        f"Context: {docsContent}" # RAG context is attached here
    )
    
    # The list of messages in the conversation.
    # Only adds messages that AREN'T tool calls, as the information
    # from the tool calls is added to the system prompt as seen above.
    conversation = [
        message for message in state["messages"] # Every message in the conversation
        if message.type in ("human", "system")# If it's human input or the system prompt
        or (message.type == "ai" and not message.tool_calls)# Or AI and not a tool call.
    ]
    
    # The final prompt consists of the system prompt and the conversation.
    # This does mean that as a conversation continues, token cost will greatly increase.
    prompt = [SystemMessage(systemPrompt)] + conversation
    

    # Get the LLM's response to the prompt and return the response.
    response = llm.invoke(prompt)
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