from flask import Flask, request, render_template, redirect, url_for, flash
import os
from io import BytesIO
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import string
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
import numpy as np



app = Flask(__name__)
app.secret_key = os.urandom(24)

load_dotenv()
stemmer = PorterStemmer()
_stop_words_cache = None  # Global cache

# Load Azure config
AZURE_CONN = os.getenv("AZURE_CONNECTION_STRING")
CONTAINER = os.getenv("AZURE_CONTAINER_NAME", "quiz4container")

if not AZURE_CONN:
    raise RuntimeError("AZURE_CONNECTION_STRING not set!")

# Azure Blob client setup
blob_service = BlobServiceClient.from_connection_string(AZURE_CONN)
container_client = blob_service.get_container_client(CONTAINER)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload_txt():
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename.endswith(".txt"):
            try:
                file_data = file.read()
                blob_client = container_client.get_blob_client(file.filename)
                blob_client.upload_blob(file_data, overwrite=True)

                content = file_data.decode("utf-8", errors="ignore")
                flash("File uploaded successfully!")
                return render_template("upload.html", filename=file.filename, content=content)
            except Exception as e:
                flash(f"Upload failed: {e}")
                return redirect(url_for("upload_txt"))
        else:
            flash("Please upload a valid .txt file.")
            return redirect(url_for("upload_txt"))

    return render_template("upload.html", filename=None, content=None)


@app.route("/process", methods=["GET", "POST"])
def process_text():
    if request.method == "POST":
        uploaded_file = request.files.get("file")
        if not uploaded_file or not uploaded_file.filename.endswith(".txt"):
            flash("Please upload a valid .txt file.")
            return redirect(url_for("process_text"))

        filename = uploaded_file.filename

        # Check if StopWords.txt exists
        stop_blob = container_client.get_blob_client("StopWords.txt")
        if not stop_blob.exists():
            flash("StopWords.txt is missing. Upload it before processing text.")
            return redirect(url_for("process_text"))

        # Load stop words
        stop_data = stop_blob.download_blob().readall().decode("utf-8")
        stop_data  = stop_data.encode('ascii', errors='ignore').decode()
        stop_words = set(
            word.strip('"\'').lower()
            for word in stop_data.replace("\n", " ").split()
        )

        # Read uploaded file's content
        raw_text = uploaded_file.read().decode("utf-8")

        # -------- STEP 1: Remove stop words from raw text --------
        raw_text  = raw_text.encode('ascii', errors='ignore').decode()
        words = raw_text.split()
        partially_cleaned = []
        removed_count_step1 = 0

        for word in words:
            if word.lower() not in stop_words:
                partially_cleaned.append(word)
            else:
                removed_count_step1 += 1

        step1_text = " ".join(partially_cleaned)

        # -------- STEP 2: Remove punctuation EXCEPT apostrophes --------
        # Keep apostrophes by excluding them from the regex
        no_punct_except_apostrophe = re.sub(r"[^\w\s']+", "", step1_text)

        # -------- STEP 3: Remove stop words again --------
        final_words = []
        removed_count_step2 = 0

        for word in no_punct_except_apostrophe.split():
            if word.lower() not in stop_words:
                final_words.append(word.lower())
            else:
                removed_count_step2 += 1

        step3_text = " ".join(final_words)

        # -------- STEP 4: Remove ALL punctuation including apostrophes --------
        final_cleaned_text = re.sub(r"[^\w\s]", "", step3_text)

        # Final count and output
        total_removed = removed_count_step1 + removed_count_step2

        # Save original file to blob
        original_blob = container_client.get_blob_client(filename)
        original_blob.upload_blob(raw_text.encode("utf-8"), overwrite=True)

        # Save processed file to blob
        new_name = filename.replace(".txt", "_processed.txt")
        processed_blob = container_client.get_blob_client(new_name)
        processed_blob.upload_blob(final_cleaned_text.encode("utf-8"), overwrite=True)

        return render_template(
            "processed.html",
            filename=filename,
            new_filename=new_name,
            cleaned_text=final_cleaned_text,
            removed_count=total_removed
        )

    return render_template("process.html")

#hello

@app.route("/letter_frequency", methods=["GET", "POST"])
def letter_frequency():
    frequency = None
    txt_files = [
        blob.name for blob in container_client.list_blobs()
        if blob.name.endswith(".txt") and "_processed" not in blob.name
    ]

    if request.method == "POST":
        filename = request.form.get("filename")
        if not filename:
            flash("No file selected.")
            return redirect(url_for("letter_frequency"))

        blob_client = container_client.get_blob_client(filename)
        content = blob_client.download_blob().readall().decode("utf-8", errors="ignore")

        # Get only letters and make them uppercase
        letters_only = [char.upper() for char in content if char.isalpha()]
        total_letters = len(letters_only)

        # Count letters
        letter_counts = dict(Counter(letters_only))

        # Include both count and percentage for A-Z
        frequency = {
            chr(c): {
                "count": letter_counts.get(chr(c), 0),
                "percent": (letter_counts.get(chr(c), 0) / total_letters * 100) if total_letters > 0 else 0.0
            }
            for c in range(ord('A'), ord('Z') + 1)
        }

    return render_template("letter_frequency.html", files=txt_files, frequency=frequency)



@app.route("/search", methods=["GET", "POST"])
def search_text():
    results = []

    if request.method == "POST":
        query = request.form.get("query", "").strip().lower()

        if not query:
            flash("Please enter a search term.")
            return redirect(url_for("search_text"))

        # List all blobs
        blobs = container_client.list_blobs()
        for blob in blobs:
            if blob.name.endswith(".txt") and "_processed" not in blob.name:
                blob_client = container_client.get_blob_client(blob.name)
                content = blob_client.download_blob().readall().decode("utf-8", errors="ignore")
                
                for i, line in enumerate(content.splitlines(), 1):
                    if query in line.lower():
                        results.append({
                            "filename": blob.name,
                            "line_number": i,
                            "line": line.strip()
                        })

    return render_template("search.html", results=results)


def clean_text(text):
    global _stop_words_cache

    # Load stop words from Azure Blob once
    if _stop_words_cache is None:
        stop_blob = container_client.get_blob_client("StopWords.txt")
        if not stop_blob.exists():
            flash("StopWords.txt is missing. Upload it before processing text.")
            raise FileNotFoundError("StopWords.txt not found in Azure Blob Storage.")

        stop_data = stop_blob.download_blob().readall().decode("utf-8", errors="ignore")
        _stop_words_cache = set(
            word.strip(string.punctuation).lower() 
            for word in stop_data.split()
        )

    stop_words = _stop_words_cache

    text = text.encode('ascii', errors='ignore').decode()
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)



@app.route("/semantic_search", methods=["GET", "POST"])
def semantic_search():
    results = []

    blobs = [
    blob.name for blob in container_client.list_blobs()
    if blob.name.endswith("_processed.txt")
    ]

    docs = []
    filenames = []

    for blob_name in blobs:
        blob_client = container_client.get_blob_client(blob_name)
        content = blob_client.download_blob().readall().decode("utf-8", errors="ignore")
        cleaned = clean_text(content)
        docs.append(cleaned)
        filenames.append(blob_name)

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        top_k = 5

        # Load stopwords first if not already
        stop_blob = container_client.get_blob_client("StopWords.txt")
        stop_data = stop_blob.download_blob().readall().decode("utf-8")
        global stop_words
        stop_words = set(stop_data.lower().split())

        # Vectorize using bi-gram TF-IDF
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        doc_matrix = vectorizer.fit_transform(docs)

        # Clean and vectorize query
        query_cleaned = clean_text(query)
        query_vec = vectorizer.transform([query_cleaned])

        # Compute cosine similarity scores
        scores = (doc_matrix @ query_vec.T).toarray().flatten()
        scored_docs = [(filenames[i], scores[i]) for i in range(len(filenames)) if scores[i] > 0]
        top_docs = sorted(scored_docs, key=lambda x: -x[1])[:top_k]

        results = top_docs

    return render_template("semantic_search.html", results=results)


# if __name__ == "__main__":
#     app.run(debug=True)

