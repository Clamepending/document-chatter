import streamlit as st
from utils import *

MODEL_NAME = "llama3.2:1b" #"deepseek-r1:8b" # 

ollama.pull(MODEL_NAME)

def initialize_chat_engine(uploaded_file):
    """Initialize all the components needed for the chat engine"""
    try:
        # Save uploaded file temporarily
        if uploaded_file:
            temp_path = f"./{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Initialize components with uploaded file
            documents = ingest_pdf(temp_path)
            chunks = split_documents(documents)
            # Clean up existing ChromaDB collection
            if os.path.exists("./chroma_db"):
                import shutil
                shutil.rmtree("./chroma_db")
            db = create_vector_db(chunks)
            llm = ChatOllama(model=MODEL_NAME)
            retriever = create_retriever(db, llm)
            chain_and_memory = create_chain(retriever, llm)
            
            # Clean up temp file
            os.remove(temp_path)
            
            return chain_and_memory
        return None
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def main():
    st.title("Document Chat Assistant")
    st.write("Upload a PDF and ask questions about it!")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file:
        # Initialize or get chat engine from session state
        if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
            with st.spinner("Processing PDF..."):
                chain_and_memory = initialize_chat_engine(uploaded_file)
                if chain_and_memory:
                    st.session_state.chain = chain_and_memory["chain"]
                    st.session_state.memory = chain_and_memory["memory"]
                    st.session_state.current_file = uploaded_file.name
                    st.session_state.messages = []  # Reset chat history for new file
                    st.success("PDF processed successfully!")
                else:
                    st.error("Failed to process PDF")
                    return
        
        # Initialize chat history in session state
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Display chat historyçç
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask your question about the document"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = query_document(st.session_state.chain, prompt)
                    st.write(response)
                    
                    # Save to memory
                    st.session_state.memory.save_context(
                        {"input": prompt},
                        {"output": response}
                    )
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("Please upload a PDF file to begin chatting.")

    # Add a button to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
        st.session_state.memory.clear_memory()

if __name__ == "__main__":
    main()