# RAG App for vision and mission evaluation from CSV/Excel File 
This project is learning material from my course as a data science tutor at [Superprof](https://www.superprof.co.id/belajar-data-analyst-data-science-machine-learning-dan-konsultasi-project-dengan-python-sql-tableau.html) This project is learning material from my course as a data science tutor at Superprof.

## Installation

Clone this repository:

```bash
git clone https://github.com/ramadhaykp12/rag_practice.git
cd rag-qna-webapp
```

Install all required dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Configuration

Before running the app, create a `.env` file in the project root directory and add your API keys:

```bash
GOOGLE_API_KEY="your_gemini_api_key"
HUGGINGFACEHUB_API_TOKEN="your_huggingface_token"
```

You can obtain:

* **Gemini API Key** from [Google AI Studio](https://aistudio.google.com/app/apikey)
* **Hugging Face Token** from [Hugging Face Access Tokens](https://huggingface.co/settings/tokens)

---

## Running the App

Once everything is configured, start the Streamlit app:

```bash
streamlit run app.py
```

Then open the provided local URL in your browser (usually `http://localhost:8501`).

---
