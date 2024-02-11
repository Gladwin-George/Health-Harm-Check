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

#
from flask import Flask, render_template, request, jsonify, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_mail import Mail, Message
#

#---------------------------------------------------------------
import cv2
import easyocr
import numpy as np
import csv
from flask import make_response
from flask import redirect, url_for
#---------------------------------------------------------------


# Load environment variables
load_dotenv()

#
app = Flask(__name__)
#app.secret_key = 'your secret key'  # Set the secret key for your Flask application to use sessions
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///conversations.db'  # SQLite database (you can change this to another database)
db = SQLAlchemy(app)
migrate = Migrate(app, db)
#


#
#-----------------------------------------------------------------
#mailing

#email
EMAIL_ID = os.getenv('EMAIL')#os.environ.get("EMAIL_ID")
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')#os.environ.get("EMAIL_PASSWORD")

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = EMAIL_ID#'gladwinfiverr@gmail.com'
app.config['MAIL_PASSWORD'] = EMAIL_PASSWORD#'xfln ajez ektr gjru'
app.config['MAIL_DEFAULT_SENDER'] = EMAIL_ID#'gladwinfiverr@gmail.com'

mail = Mail(app)

#-----------------------------------------------------------------
#

# Define a model for storing conversations in the database
class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.String(500))
    response = db.Column(db.String(500))
#


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
#---------------------------------------------------------------
# Load harmful ingredients from CSV into a dictionary
harmful_ingredients_dict = {}
with open('harmful_ingredients.csv', 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the first row (headings)
    for row in csv_reader:
        if len(row) >= 2 and row[0] and row[1]:  # Check if both columns have data
            ingredient_name = row[0].strip().lower()  # Assuming ingredient name is in the first column
            harmful_ingredient_description = row[1].strip()  # Assuming harmful ingredient description is in the second column
            harmful_ingredients_dict[ingredient_name] = harmful_ingredient_description

#---------------------------------------------------------------

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=['GET', 'POST'])
def chat():
    msg = request.form['msg']
    input = msg
    print(msg)
  
    result = qa_chain.invoke({"query": input})
    response = result["result"]
    print("Response: ", response)


    try:
        # Save the conversation to the database
        conversation_entry = Conversation(message=msg, response=result["result"])
        db.session.add(conversation_entry)
        db.session.commit()
    except Exception as e:
        print(f"Error saving conversation to the database: {e}")
        db.session.rollback()  # Rollback changes in case of an error
        return jsonify({"error": "Failed to save conversation to the database"})

    # Save the conversation to a text file
    save_to_text_file(msg, result["result"])

    return str(result["result"])    
    

def save_to_text_file(msg, result):
    with open('conversations.txt', 'a', encoding='utf-8') as file:
        file.write(f'user: {msg}\n')
        file.write(f'chatbot: {result}\n')


#-----------------------------------------------------------------
#mailing
@app.route('/send_email', methods=['POST'])
def send_email():
    try:
        # Read conversation data from the text file
        with open('conversations.txt', 'r', encoding='utf-8') as file:
            conversation_data = file.read()

        # Create and send an email
        msg = Message('Chat Conversation', sender=EMAIL_ID, recipients=[EMAIL_ID])
        msg.body = conversation_data
        mail.send(msg)

        return jsonify({"success": True, "message": "Email sent successfully!"})
    except Exception as e:
        print(f"Error sending email: {e}")
        return jsonify({"error": "Failed to send email"})
#-----------------------------------------------------------------
#harmful ingredients
@app.route('/harmful', methods=['GET', 'POST'])
def harmful():
    error = None
    harmful_ingredients = []

    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            error = "No file uploaded"

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            error = "No selected file"

        # If the file exists and is allowed, proceed with OCR and detection
        if file and not error:
            filename = 'uploaded_image.jpg'
            file.save(filename)

            # Step 1: Extract text from the image
            img = cv2.imread(filename)
            reader = easyocr.Reader(['en'])
            text_results = reader.readtext(filename)
            extracted_text = ' '.join([result[1] for result in text_results])

            # Step 3: Identify harmful ingredients
            for ingredient_name, description in harmful_ingredients_dict.items():
                if ingredient_name in extracted_text.lower():
                    harmful_ingredients.append((ingredient_name, description))

            os.remove(filename)  # Remove the uploaded image

        return jsonify({'error': error, 'harmful_ingredients': harmful_ingredients})

    return render_template('chat.html')
    
#---------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, port=5001)

