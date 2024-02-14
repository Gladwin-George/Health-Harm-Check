import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA


from joblib import dump, load

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Google API key not found in environment variables.")

genai.configure(api_key=GOOGLE_API_KEY)

# Load the data
loader = PyPDFDirectoryLoader("data")
data = loader.load_and_split()

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
context = "\n".join(str(p.page_content) for p in data)
texts = text_splitter.split_text(context)

#embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

#Create a vector index
vector_index = Chroma.from_texts(texts, embeddings).as_retriever()

#model
model = ChatGoogleGenerativeAI(model="gemini-pro", 
                               temperature=0.3, 
                               google_api_key=GOOGLE_API_KEY,
                               convert_system_message_to_human=True)

#retrieve the relevant documents
qa_chain = RetrievalQA.from_chain_type(
    model,
    retriever=vector_index,
    return_source_documents=True
)


template = """
  Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
  provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
  Context:\n {context}?\n
  Question: \n{question}\n

  Answer:
"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)# Run chain

qa_chain = RetrievalQA.from_chain_type(
    model,
    retriever=vector_index,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

question = input("Enter your question: ")  # Prompt the user to enter a question
result = qa_chain.invoke({"query": question})
print(result["result"])


