import streamlit as st
from dotenv import load_dotenv
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from indexing import split_documents, store_documents
from generate_response import answer

load_dotenv()


def load_documents(docs):
    texts = []
    for doc in docs:
        # Create a temporary file to store the uploaded PDF content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(doc.read())
            temp_file.flush()
            # Load the PDF from the temporary file path
            reader = PyPDFLoader(temp_file.name)
            texts.append(reader)

    loader_all = MergedDataLoader(loaders=texts)
    return loader_all.load()


def main():
    st.set_page_config("QnA PDF")
    st.header("RAG-Based Research Paper Q&A Assistant: Unlocking Knowledge from PDFs")
    user_query = st.text_input("Insert a query about your documents")

    with st.sidebar:
        st.title("Menu:")
        st.markdown(
            """
            Guide:
            1. Upload your PDF Files
            2. Insert a query about your documents
            3. Click Submit & Process button
        """
        )
        docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                load_docs = load_documents(docs)
                chunks = split_documents(load_docs)
                vector_store = store_documents(load_docs)
                retriever = vector_store.as_retriever()

                response = answer(question=user_query, retriever=retriever)

    st.write("Reply: ", response)
    st.success("Done")


if __name__ == "__main__":
    main()
