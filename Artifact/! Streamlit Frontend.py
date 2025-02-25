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
if 'message_history' not in st.session_state:
    st.session_state.message_history = [AIMessage(content=defaultMsg)]
    
# Organises the app so that the left column (which holds a debug button) is smaller 
# than the main chat UI.
debugBtn, chatHist, = st.columns([1, 9])

# In the left column, a button is placed that clears the chat history when clicked.
# It resets the entire conversation back to the original "Hello!" prompt.
with debugBtn:
    if st.button('[DEBUG] Clear history'):
        st.session_state.message_history = [AIMessage(content=defaultMsg)]


# The main UI is in this column.
# The user inputs their prompt into a text box which is then sent off to the chatbot.
with chatHist:
    user_input = st.chat_input("Ask anything about BCU!")

    # If the new message in the chat came from the user, show it as a HumanMessage.
    if user_input:
        st.session_state.message_history.append(HumanMessage(content=user_input))

        # Prompts the LLM and RAG tool with the current conversation history.
        response = graph.invoke({
            'messages': st.session_state.message_history
        })

        # The response that gets returned still contains the whole history,
        # but also appends the latest response. Therefore, make that the new
        # message history.
        st.session_state.message_history = response['messages']

    # Shows the running conversation history after any message is sent.
    # Iterates over the whole message history, showing HumanMessages for the user,
    # and AIMessages for the LLM.
    
    for i in range(1, len(st.session_state.message_history) + 1):
        currentMsg = st.session_state.message_history[-i]
        
        # ? If the message was written by the LLM:
        if isinstance(currentMsg, AIMessage):
            # ! The LLM returns a blank message following the tool call.
            # ! I'm not entirely sure why this is, but it can be hidden from the UI.
            if currentMsg.content != "":
                # ? Use a robot profile picture. 
                message_box = st.chat_message('assistant')
                message_box.markdown(currentMsg.content)
                
        # ? Alternatively, if the user wrote it:
        elif isinstance(currentMsg, HumanMessage):
            # ? Use a person profile picture.
            message_box = st.chat_message('user')
            message_box.markdown(currentMsg.content)
        
    # ? If it isn't an AIMessage or HumanMessage, it must be a ToolMessage.
    # ? ToolMessages are created when the LLM gets context from the vector DB.
    # ? It still gets added to the message history, but there's no reason to output it.
    
# ! Seems that this is a lot more expensive (literally) than the Jupyter notebook despite minimal changes to the backend.  # noqa: E501
# TODO: It could be because either context or convo history is being added more than once per message. This MUST be fixed if true.  # noqa: E501
# TODO: Responses can be a bit shaky at times. It's probably to do with the text splitter chunk sizes?  # noqa: E501
# TODO: Add old comments and update existing from Jupyter notebook to Chatbot.py. Many were written in Markdown so they got lost when cell contents were cut.  # noqa: E501
