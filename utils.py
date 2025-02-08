import os

# load document

from langchain_community.document_loaders import UnstructuredPDFLoader

def ingest_pdf(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find file at path: {path}")
    loader = UnstructuredPDFLoader(path)
    documents = loader.load()
    
    # print(documents)
    return documents


# chunk up document

from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
        length_function=len,
        separators=["\n", "\r\n", "\r", "\t"]
    )
    chunks = splitter.split_documents(documents)
    return chunks


# vectorize

import ollama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def create_vector_db(chunks):
    
    persist_directory = "./chroma_db"
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
        
        
    ollama.pull("nomic-embed-text")
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="ARK2025",
        persist_directory=persist_directory
    )
    
    vector_store.persist()
    
    return vector_store
    


# mulitple query generation 

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama


def create_retriever(db, llm):
    query_prompt = PromptTemplate(
        input_variables = ["question"],
        template="""Create three different versions of this question to help find relevant information:
        Original question: {question}"""
    )
    
    retriever = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(),
        llm=llm,
        prompt=query_prompt
        
    )
    return retriever

# create chain

from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory

def create_chain(retriever, llm):
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based on the following context and chat history.
        If you cannot answer based on these, say so.
        
        Context: {context}
        Chat History: {chat_history}
        Current Question: {question}
        """
    )
    def load_memory(_):
        return memory.load_memory_variables({})["chat_history"]
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough(), "chat_history": load_memory}
        | prompt
        | llm
        | StrOutputParser()
    )
    return {"chain": chain, "memory": memory}


def query_document(chain, question):
    return chain.invoke(question)

