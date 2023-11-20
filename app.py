import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import openai
import cloudpickle
import copy




# Sidebar contents
with st.sidebar:
    st.title('LLM Doc Scanner App')
    st.markdown('''
    ## About
    This chat app is powered by OpenAI with this you can ask particular query related to a document you uploaded
    ''')
    add_vertical_space(5)
 
load_dotenv()
 
def main():
    st.header("Chat with PDF")
    
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
 
    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)
 
        
    #     if os.path.exists(f"{store_name}.pkl"):
    #         try:
    #             with open(f"{store_name}.pkl", "rb") as f:
    #                 VectorStore = pickle.load(f)
    #         except Exception as e:
    #             print(f"Error during unpickling: {e}")
    # # st.write('Embeddings Loaded from the Disk')s
    #     else:
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

            # st.write(VectorStore)
            # with open(f"{store_name}.pkl", "wb") as f:
            #     pickle.dump(VectorStore, f)
        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
 
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)
 
        if query and VectorStore is not None:
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)
 
if __name__ == '__main__':
    main()