import streamlit as st
import requests

# Set up the page configuration with a wide layout for a more spacious chat experience
st.set_page_config(page_title="Tax Law Assistant ChatBot", page_icon="💼", layout="wide")

# Main Title and Subheading
st.markdown("<h1 style='text-align: center;'>💼 Tax Law Assistant</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Your Expert Indian Tax Law Advisor</h3>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Sidebar for Instructions and Tips
st.sidebar.header("Instructions")
st.sidebar.write(
    """
    Welcome to the Tax Law Assistant chatbot.  
    You can ask questions about Indian Tax Law, such as deductions under Section 80C.  

    **How to Use:**
    - Type your question in the input box.
    - Click 'Send' to receive an answer.
    
    The responses may include formatted headings, equations, and references.
    """
)
st.sidebar.markdown("---")
st.sidebar.header("Quick Tips")
st.sidebar.markdown(
    """
    - Use **bold text** for emphasis.
    - Use _italics_ for subtle emphasis.
    - To include equations in your messages (if needed), you can use LaTeX notation, e.g.,  
      \\( E = mc^2 \\)
    """
)

# Initialize session state for storing chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def send_message(user_message):
    """
    Sends the user's message to the FastAPI server and appends the response to the session state.
    """
    # Append the user's message to the chat history
    st.session_state.messages.append({"sender": "User", "message": user_message})
    try:
        # Send the query to the FastAPI endpoint
        response = requests.post("https://ac4e042661303.notebooks.jarvislabs.net/ask", json={"question": user_message})
        if response.status_code == 200:
            data = response.json()
            if "result" in data:
                answer = data["result"]
                st.session_state.messages.append({"sender": "Assistant", "message": answer})
            else:
                error_msg = data.get("error", "Unknown error")
                st.session_state.messages.append({"sender": "Assistant", "message": f"Error: {error_msg}"})
        else:
            st.session_state.messages.append({"sender": "Assistant", "message": f"Error: {response.status_code}"})
    except Exception as e:
        st.session_state.messages.append({"sender": "Assistant", "message": f"Error: {e}"})

# Chat input form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Your Question:", placeholder="Ask about tax deductions, Section 80C, etc.")
    submitted = st.form_submit_button("Send")
    if submitted and user_input:
        send_message(user_input)

# Display chat history in a chat-like interface
for chat in st.session_state.messages:
    if chat["sender"] == "User":
        st.markdown(
            f"""
            <div style='text-align: right; background-color: #0052cc; color: #ffffff; padding: 10px; border-radius: 10px; margin: 8px;'>
                <strong>You:</strong> {chat['message']}
            </div>
            """, unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style='text-align: left; background-color: #36393f; color: #ffffff; padding: 10px; border-radius: 10px; margin: 8px;'>
                <strong>Assistant:</strong> {chat['message']}
            </div>
            """, unsafe_allow_html=True
        )
        
# --- OPTIONAL: Render LaTeX Equations ---
# If your assistant's answers include LaTeX equations (wrapped with $$ or \\( \\)), you can detect and render them.
# Here's a simple example: if the answer contains a marker "LATEX:" followed by an equation,
# you might extract and display it using st.latex.
#
# Example snippet (this is just illustrative and may require custom parsing logic):
#
# for chat in st.session_state.messages:
#     if chat["sender"] == "Assistant" and "LATEX:" in chat["message"]:
#         parts = chat["message"].split("LATEX:")
#         text_before = parts[0]
#         equation = parts[1].split()[0]  # Adjust parsing as needed
#         st.markdown(text_before)
#         st.latex(equation)
#         st.markdown(" ".join(parts[1].split()[1:]))
#
# You can customize the parsing logic based on how your assistant outputs equations.
