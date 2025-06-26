from flask import Flask, request, render_template, redirect, url_for, flash
import os
from io import BytesIO
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import string
import re
from collections import Counter
import numpy as np



app = Flask(__name__)
app.secret_key = os.urandom(24)

load_dotenv()

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



@app.route("/char_frequency", methods=["GET", "POST"])
def char_frequency():
    frequency = None
    input_text = ""
    updated_text = ""
    old_char = ""
    new_char = ""

    if request.method == "POST":
        input_text = request.form.get("T", "").strip()
        old_char = request.form.get("old_char", "")
        new_char = request.form.get("new_char", "")

        if not input_text:
            flash("Please enter a string.")
            return redirect(url_for("char_frequency"))

        # Convert to lowercase
        updated_text = input_text.lower()

        # Replace character if applicable
        if old_char and len(old_char) == 1:
            updated_text = updated_text.replace(old_char, new_char)

        # Remove punctuation
        updated_text = updated_text.translate(str.maketrans('', '', string.punctuation))

        # Count characters excluding spaces
        chars_only = [char for char in updated_text if not char.isspace()]
        total_chars = len(chars_only)
        char_counts = dict(Counter(chars_only))

        # Build frequency dictionary
        frequency = {
            char: {
                "count": char_counts[char],
                "percent": (char_counts[char] / total_chars * 100) if total_chars > 0 else 0.0
            }
            for char in sorted(char_counts)
        }

    return render_template(
        "char_frequency.html",
        frequency=frequency,
        input_text=input_text,
        updated_text=updated_text,
        old_char=old_char,
        new_char=new_char
    )

@app.route("/question11", methods=["GET", "POST"])
def question11():
    word_count = 0
    input_string = ""
    char_results = {}

    if request.method == "POST":
        input_string = request.form.get("S", "").strip()

        if input_string:
            # Count words (split by space)
            words = input_string.split()
            word_count = len(words)

            # Clean and lowercase words for comparison
            cleaned_words = [
                word.strip(string.punctuation).lower()
                for word in words
            ]

            # Unique characters in the original string (excluding spaces)
            chars = sorted(set(input_string.lower()) - {' '})

            for char in chars:
                matching = [
                    word for word in cleaned_words
                    if word.startswith(char)
                ]
                char_results[char] = matching if matching else ["-"]

    return render_template(
        "question11.html",
        input_string=input_string,
        word_count=word_count,
        char_results=char_results
    )


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



if __name__ == "__main__":
    app.run(debug=True)

