# Libbaries
import json
import boto3
import streamlit as st
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)



def data_ingestion():
    loader=PyPDFDirectoryLoader("pdfs")
    documents=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    
    docs=text_splitter.split_documents(documents)
    return docs



def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

def get_llama3_llm():
    llm=Bedrock(model_id="meta.llama3-70b-instruct-v1:0",client=bedrock,
                model_kwargs={'max_gen_len':512, 'temperature':0.2, 'top_p':0.9})
    
    return llm

prompt_template = """

Human: You are a weather analyst. You love to help anyone with weather information and you love to generate weather report in tabular format.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']


def main():
    st.set_page_config("Credit Report Generator")
    
    st.header("Content Conjurers")

    user_question = st.text_input("How can I help you?")

    with st.sidebar:
        st.title("")
        
        if st.button("Sync Documentation"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Generate Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm=get_llama3_llm()
            
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

if __name__ == "__main__":
    main()













