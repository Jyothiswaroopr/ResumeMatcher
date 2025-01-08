import fitz
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import os
import tempfile

# Download NLTK stopwords
nltk.download("stopwords")

def preprocess_text(text):
    """Clean and preprocess text."""
    stop_words = set(stopwords.words("english"))
    words = text.lower().split()
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return " ".join(words)

def extract_and_preprocess(pdf_path):
    """Extract and preprocess text from a PDF."""
    try:
        file = fitz.open(pdf_path)
        text = ""
        for page in file:
            text += page.get_text()
        file.close()
        return preprocess_text(text)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def match_resumes(job_desc_path, resume_path):
    """Compare job description with resume."""
    job_desc = extract_and_preprocess(job_desc_path)
    resume = extract_and_preprocess(resume_path)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([job_desc, resume])
    similarity = cosine_similarity(vectors[0], vectors[1])
    return similarity[0][0]

def match_resumes_in_folder(job_desc_path, resumes_folder_path):
    """Compare job description with all resumes in a folder."""
    if not os.path.exists(resumes_folder_path):
        return None, "Resumes folder does not exist."

    resume_files = [f for f in os.listdir(resumes_folder_path) if f.endswith('.pdf')]
    if not resume_files:
        return None, "No resumes found in the folder."

    resume_similarity = {}
    for resume_file in resume_files:
        resume_path = os.path.join(resumes_folder_path, resume_file)
        similarity = match_resumes(job_desc_path, resume_path)
        resume_similarity[resume_file] = similarity
    return resume_similarity, None

# Streamlit app
st.title("Resume Matcher")

job_desc_file = st.file_uploader("Upload Job Description (PDF)", type="pdf")
resumes_folder = "resumes"

if job_desc_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(job_desc_file.read())
        temp_job_desc_path = temp_file.name

    scores, error = match_resumes_in_folder(temp_job_desc_path, resumes_folder)
    if error:
        st.error(error)
    else:
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        st.write("**Resume Similarity Ranking**:")
        for idx, (filename, score) in enumerate(sorted_scores):
            st.write(f"{idx + 1}. {filename} - Match Score: {score * 100:.2f}%")
