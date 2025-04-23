# Streamlit provides the frontend UI for the chatbot.
import streamlit as st

# Import the directed graph (using LangGraph) from the chatbot script.
from Chatbot import graph

# Used to show if a message was written by the AI or the user,
from langchain_core.messages import AIMessage, HumanMessage

# Use Streamlit's wide layout, which will use the whole screen space rather than a small
# centre column. Also gives the page a title and icon which is shown in the 
# browser tab view.
st.set_page_config(layout = 'wide', page_title = 'University Artifically Intelligent Chatbot',
                   page_icon='📚')

# When the UI first opens, this is the first message that will already be in the chat.
defaultMsg = "Hello! I'm an assistant chatbot designed to help answer any questions you have about BCU."

# If there's no message history (app just opened, or history just cleared), show the default message.
if 'message_history' not in st.session_state:
    st.session_state.message_history = [AIMessage(content=defaultMsg)]
    
# Organises the app so that the left and right columns are smaller than the main chat UI.
clearHistBtn, chatHist, queryLog = st.columns([1, 8, 1])
queries = []

# In the left column, a button is placed that clears the chat history when clicked.
# It resets the entire conversation back to the original "Hello!" prompt.
with clearHistBtn:
    if st.button('Clear history'):
        st.session_state.message_history = [AIMessage(content=defaultMsg)]
        queries = []

# The main UI is in this column.
# The user inputs their prompt into a text box which is then sent off to the chatbot.
with chatHist:
    userInput = st.chat_input("Ask anything about BCU!")

    # When an input is confirmed, add it to the conversation history and get a response.
    if userInput:
        st.session_state.message_history.append(HumanMessage(content=userInput))

        # Prompts the LLM and RAG tool with the current conversation history.
        response = graph.invoke({
            'messages': st.session_state.message_history
        })

        # The response that gets returned still contains the whole history,
        # but also appends the latest LLM response. Therefore, make that the new
        # message history.
        st.session_state.message_history = response['messages']


    # Shows the running conversation history after any message is sent.
    # Iterates over the whole message history, showing HumanMessages for the user,
    # and AIMessages for the LLM.
    
    for i in range(1, len(st.session_state.message_history) + 1):
        currentMsg = st.session_state.message_history[-i]
        
        # ? If the message was written by the LLM:
        if isinstance(currentMsg, AIMessage):
            # The tool call is an AIMessage with no content, as the call is instead done within the metadata,
            # so it would show a blank box. To fix this, the message is checked to see if it's blank first.
            if currentMsg.content != "":
                # ? Use a robot profile picture. 
                messageBox = st.chat_message('assistant')
                messageBox.markdown(currentMsg.content)

            # For logging purposes, the tool call used in the message is stored.
            toolCalls = currentMsg.additional_kwargs.get("tool_calls")
            
            if toolCalls is not None:
                # The tool call is an array containing a dictionary of dictionaries, but I'm only looking
                # for the query given to the vector DB, stored in function.arguments.
                toolQuery = toolCalls[0].get("function").get("arguments")
                
                # Strangely, the query is formatted like a dictionary but is actually a string,
                # so I remove the text "query":" and also the closing speech marks.
                queries.append(toolQuery[10 : len(toolQuery)-2])
                      
        # ? Alternatively, if the user wrote it:
        elif isinstance(currentMsg, HumanMessage):
            # ? Use a person profile picture.
            messageBox = st.chat_message('user')
            messageBox.markdown(currentMsg.content)
            
        
    # ? If it isn't an AIMessage or HumanMessage, it must be a ToolMessage.
    # ? ToolMessages are created when the LLM gets context from the vector DB.
    # ? It still gets added to the message history, but there's no reason to output it.
    
        # else: 
        #     messageBox = st.chat_message(name = 'Tool', avatar = "🔍")
        #     messageBox.markdown(currentMsg.content)

# In the right column, the queries being sent by the retrieval agent to the DB are logged.
with queryLog:
    st.title("Agent's queries to vector DB")
    st.write(queries)