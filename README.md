# RAG-Based ML Research Paper Q&A Assistant

RAG project a QnA system for machine learning research papers in PDF

## How To Run

1. Create .env file

   ```
   touch .env
   ```

2. Insert necessary values in .env file

   ```
   GOOGLE_API_KEY=
   LANGCHAIN_API_KEY=
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
   LANGCHAIN_PROJECT="qna-paper-rag"
   ```

3. Run the program, you can choose run method:

   ### CLI

   ```
   python generate_response.py -d data "your_query"
   ```

   Note: </br>
   -d or --data for directory where you put your PDF file/s

   ### GUI Streamlit

   1. Run Streamlit Application

      ```
      streamlit run app.py
      ```

   2. To open Streamlit GUI, visit this link in browser

      ```
      http://localhost:8501/
      ```
