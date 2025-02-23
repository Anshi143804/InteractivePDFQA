import streamlit as st
from PyPDF2 import PdfReader
import os
import base64

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain 
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"
print("GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docx):
    text=""
    for pdf in pdf_docx:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter( chunk_size=10000, chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks , embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template="""
    Answer the question lik eyou know a lot and want to teach everyone like a student from basics like they dont know anything from scratch with examples and supporting theories , if unable to do so dont provide wrong answer just say "oops question out of my league " and move forward , make sure with every answer give examples like you are teaching a 5 year old kid.
    Context: \n{context}?\n
    Question: \n{question}\n 
    
    Answer:
    """
    model=ChatGoogleGenerativeAI(model="gemini-pro" , temperature=0.4)
    prompt=PromptTemplate(template=prompt_template , input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db=FAISS.load_local("faiss_index", embeddings , allow_dangerous_deserialization=True)
    docs=new_db.similarity_search(user_question)
    chain=get_conversational_chain()

    response=chain(
        {"input_documents":docs , "question": user_question}
         , return_only_outputs=True)
    print(response)
    st.write("Reply: " ,response["output_text"])


def display_pdf(pdf_path):
    with open(pdf_path, "rb") as pdf_file:
        base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Chat with PDFs using Gemini", layout="wide")
    st.title("📚 Chat with PDFs using Gemini AI")

    with st.sidebar:
        st.header("Upload PDFs 📄")
        pdf_docs = st.file_uploader("Upload multiple PDFs", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing PDFs..."):
                # Save PDFs to temp folder
                for pdf in pdf_docs:
                    pdf_path = os.path.join("temp", pdf.name)
                    with open(pdf_path, "wb") as f:
                        f.write(pdf.read())
                # Extract text, split into chunks, and store in FAISS
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDFs processed and vector store created!")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📄 PDF Preview")
        if pdf_docs:
            selected_pdf = st.selectbox("Select a PDF to view:", [pdf.name for pdf in pdf_docs])
            if selected_pdf:
                pdf_path = os.path.join("temp", selected_pdf)
                display_pdf(pdf_path)

    with col2:
        st.subheader("💬 Ask Questions about the PDFs")
        user_question = st.text_input("What would you like to know?")
        if user_question:
            user_input(user_question)

if __name__=="__main__":
    main()


