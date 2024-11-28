import re
import os
from dotenv import load_dotenv
from collections import Counter
from operator import itemgetter

from langchain import hub
from langchain.load import dumps, loads
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from indexing import indexing

CHROMA_PATH = "chromadb"


def reciprocal_rank_fusion(results: list[list], k=60):
    """
    Reciprocal_rank_fusion that takes multiple lists of ranked documents
    and an optional parameter k used in the RRF formula
    """

    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results


def llm_model():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
    )
    return llm


def retrieve_documents():
    vector_store = indexing()

    return vector_store.as_retriever()


def query_expansion(llm, question):
    template = """
    You are a helpful assistant that generates multiple search queries based on a single input query. \n
    Generate multiple search queries related to: {question} \n
    Output (4 queries):
    """
    prompt_perspectives = ChatPromptTemplate.from_template(template)

    retriever = retrieve_documents()

    generate_queries = (
        prompt_perspectives | llm | StrOutputParser() | (lambda x: x.split("\n"))
    )

    # question = "What is the best algorithm to use if I have to work on text data?"

    retrieval_chain = generate_queries | retriever.map() | reciprocal_rank_fusion
    # expanded_queries = retrieval_chain.invoke({"question": question})
    return retrieval_chain


def main(question):
    llm = llm_model()
    template = """
    Answer the following question based on this context. 

    {context} 

    Question: {question} 
    """
    prompt = ChatPromptTemplate.from_template(template)
    # prompt = hub.pull("rlm/rag-prompt")

    retrieval_chain = query_expansion(llm, question)

    rag_chain = (
        {"context": retrieval_chain, "question": itemgetter("question")}
        | prompt
        | llm
        | StrOutputParser()
    )

    # question
    response = rag_chain.invoke({"question": question})
    print(response)


if __name__ == "__main__":
    main(
        "What is the meaning of Shifting value creation of machine learning and what is the cause of it?"
    )
