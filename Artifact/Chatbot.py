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
# structure for the chatbot. LangGraph is described in detail in 
# its own notebook section.
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# Get the OpenAI API key from environment variables so that 
# it's not visible in this code on GitHub. OpenAI would revoke the key if it leaked.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Sets the directory of the FAISS DB that's being loaded from.
    # Options:
    #   FAISS-PyPDF: Chunk size 1000, Overlap 200, PyPDFLoader with default args.
    #   FAISS-SmallChunks: Chunk size 500, Overlap 100, PyPDFLoader with default args.
FAISS_PATH = "FAISS-SmallChunks"


# Sets up the embedding model with the API key.
embedder = OpenAIEmbeddings(
    model = "text-embedding-3-small",
    api_key = os.environ["OPENAI_API_KEY"]
)

# Load the vector database.
db = FAISS.load_local(folder_path = FAISS_PATH,
                      embeddings = embedder,
                      allow_dangerous_deserialization=True) 


# Initialise the model.
llm = init_chat_model("gpt-4o-mini", temperature = 0.2)


@tool(response_format = "content_and_artifact")
def retrieve(query):
    # The docstring below is actually REQUIRED by LangGraph, and this 
    # won't run without it.
    """Retrieves the 3 most relevant chunks for the user's query."""
    retrievedDocs = db.similarity_search(query, k = 3)
    
    # The chunk's content and the document it came from (e.g "Attendance.pdf")
    content = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrievedDocs
    )
    return content, retrievedDocs


# Initialise an empty graph. Nodes and edges are added later.
graph_builder = StateGraph(MessagesState)

def query_or_respond(state: MessagesState):
    # Allows the LLM to use the retrieve tool that was created earlier.
    ragLLM = llm.bind_tools([retrieve])
    
    # The LLM decides on its own if it needs retrieval.
    response = ragLLM.invoke(state["messages"])
    
    # A key element of using a MessagesState is that that will append
    # the response to the conversation history rather than overwriting.
    return {"messages": [response]}


# Creating a node for the list of tools that the LLM can access. 
# At the moment, it's only one tool, the retriever, but other tools can 
# easily be added later by appending them to this list.
tools = ToolNode([retrieve])

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
        # This is formatted like this to follow the Ruff linter's line length rule.
        "You are an assistant to help new students get acclimated to Birmingham City "
        "University. If you don't know the answer, say that you "
        "don't know. When referring to context, be specific and quote the context. "
        "Use five sentences maximum and keep the answer concise. "
        "Use the following pieces of retrieved context to answer "
        "the question."
        "\n\n"
        f"Context: {docsContent}" # RAG context is attached here
    )
    
    # The list of messages in the conversation.
    # Only adds messages that AREN'T tool calls, as tool calls
    # would hugely increase the input tokens used, and the LLM 
    # should (hopefully) have already said the useful info in its response
    # so it can use that instead.
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


# Add all the nodes.
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)


graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition, 
    # If retrieval is not needed, generate a general answer.
    # If it is, call the retrieval tool.
    {END: END, "tools": "tools"},
)

# An edge is also needed between the tool call and response generation
# to ensure the response has the RAG context.
graph_builder.add_edge("tools", "generate")

# After the response is generated, the graph is done.
graph_builder.add_edge("generate", END)

# Compile the graph
graph = graph_builder.compile()