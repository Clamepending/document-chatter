# import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('punkt_tab')

from utils import *
import ollama


# mainplease

def main():
    documents = ingest_pdf("./ARK2025.pdf")
    question = "What are the predictions that ARK Invest has made for 2025?"
    chunks = split_documents(documents)
    db = create_vector_db(chunks)
    llm = ChatOllama(model="llama3.2:1b")
    retriever = create_retriever(db, llm)
    chain_and_memory = create_chain(retriever, llm)
    chain, memory = chain_and_memory["chain"], chain_and_memory["memory"]
    
    while True:
        question = input("Please ask your question! (or press q to quit)\n")
        if question.lower() == "q":
            break
        response = query_document(chain, question)
        print(response)
        
        memory.save_context(
            {"input": question}, {"output": response}
        )
    
    
main()    

