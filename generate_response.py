import argparse
from dotenv import load_dotenv
from operator import itemgetter

from langchain import hub
from langchain.load import dumps, loads
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from indexing import indexing

CHROMA_PATH = "chromadb"
DOCUMENTS_PATH = "data"


def reciprocal_rank_fusion(results: list[list], k=60):
    """
    Reciprocal_rank_fusion that takes multiple lists of ranked documents
    and an optional parameter k used in the RRF formula
    https://github.com/langchain-ai/langchain/blob/master/cookbook/rag_fusion.ipynb?ref=blog.langchain.dev
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


def retrieve_documents(path):
    vector_store = indexing(path)

    return vector_store.as_retriever()


def query_expansion(path, llm, question):
    # template = """
    # You are a helpful assistant that generates multiple search queries based on a single input query. \n
    # Generate multiple search queries related to: {question} \n
    # Output (4 queries):
    # """

    template = """
    You are an expert research assistant specialized in generating comprehensive and semantically 
    diverse search queries for machine learning and deep learning research literature. \n

    Given the input query: "{question}" \n
    
    Generate 5 distinct queries that: \n
    1. Capture different semantic angles of the original query \n
    2. Explore various levels of specificity and abstraction \n
    3. Consider alternative phrasings and technical terminologies \n
    4. Aim to uncover both broad and narrow contextual information \n
    
    Constraints:
    - Ensure queries are academically rigorous \n
    - Use precise technical language \n
    - Cover different perspectives of the research topic \n
    - Avoid redundant queries \n
    """
    prompt_perspectives = ChatPromptTemplate.from_template(template)

    retriever = retrieve_documents(path)

    retrieval_chain = (
        prompt_perspectives
        | llm
        | StrOutputParser()
        | (lambda x: x.split("\n"))
        | retriever.map()
        | reciprocal_rank_fusion
    )

    return retrieval_chain


def answer(path, question):

    llm = llm_model()
    template = """
    Answer the following question based on this context. 

    {context} 

    Question: {question} 
    """
    prompt = ChatPromptTemplate.from_template(template)
    # prompt = hub.pull("rlm/rag-prompt")

    retrieval_chain = query_expansion(path, llm, question)

    rag_chain = (
        {"context": retrieval_chain, "question": itemgetter("question")}
        | prompt
        | llm
        | StrOutputParser()
    )

    # question
    response = rag_chain.invoke({"question": question})
    # print(response)
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="Question query text")
    parser.add_argument("--docs", "-d", type=str, help="Documents path")
    args = parser.parse_args()

    question = args.query
    docs_path = args.docs

    response = answer(path=docs_path, question=question)
    print(response)


if __name__ == "__main__":
    main()
