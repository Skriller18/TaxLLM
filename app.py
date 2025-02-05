import streamlit as st
from infer import TaxLawAssistant

@st.cache_resource
def load_assistant():
    return TaxLawAssistant("checkpoints/checkpoint_epoch_1000.pt")

st.title("Indian Tax Law Assistant")
st.sidebar.markdown("### Legal Disclaimer\nThis AI assistant provides general information about Indian tax laws...")

assistant = load_assistant()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your tax law question:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing tax provisions..."):
            response = assistant.generate(prompt)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})