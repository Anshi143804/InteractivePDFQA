
# InteractivePDFQA

A Streamlit app that allows you to upload and preview multiple PDFs, ask AI-powered questions about the content using Google’s Gemini Pro, and receive accurate answers. Built with LangChain for text processing, FAISS for vector storage, and Gemini's LLM for conversational AI.

## Features
- **Upload PDFs**: Supports multiple file uploads.
- **PDF Preview**: View the uploaded PDFs side-by-side.
- **AI Q&A**: Ask questions about the content of the PDFs and get intelligent responses.
- **Real-time Processing**: Uses Google Gemini Pro and LangChain for seamless AI interaction.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Anshi143804/InteractivePDFQA.git
cd InteractivePDFQA
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your API key:
- Create a `.env` file in the root directory.
- Add your Google Gemini Pro API key:
```
GOOGLE_API_KEY=your_api_key_here
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

- **Upload PDFs** using the sidebar.
- **Ask questions** in the main chat interface.
- The AI will generate responses based on the content of the PDFs.

## Folder Structure
```
.
├── app.py
├── requirements.txt
├── .env (not included, add your own API key)
├── temp/ (stores uploaded PDFs temporarily)
└── README.md
```

## Dependencies
- Streamlit
- PyPDF2
- LangChain
- FAISS
- Google Generative AI
- dotenv

## Contributing
Contributions are welcome! Feel free to fork the repository, create a new branch, and submit a pull request.

## License
This project is licensed under the MIT License.

## Acknowledgements
- [LangChain](https://www.langchain.com/)
- [Google Gemini AI](https://ai.google/)
- [Streamlit](https://streamlit.io/)

---
