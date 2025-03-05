# ===========================
# 📌 Import necessary modules
# ===========================
from flask import Flask, render_template, request, jsonify, url_for
from transformers import pipeline  # ✅ Zero-Shot Classification
import json
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import spacy
import os
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.stem import PorterStemmer
import re

app = Flask(__name__, template_folder="templates")
# ✅ Zero-Shot Classification Initialization
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# ===========================
# 📌 Initialize NLP components
# ===========================
#nltk.download("punkt")  
stemmer = PorterStemmer()
nlp = spacy.load("en_core_web_sm")

# ===========================
# 📌 Load Event Data
# ===========================
def load_events(filename="data/helmet_all_events_with_translated_descriptions.csv"):
    df = pd.read_csv(filename, encoding="utf-8")  
    df.columns = df.columns.str.strip().str.lower()
    event_texts = df["title"].astype(str) + " " + df["location"].astype(str) + " " + df["description_translated"].astype(str)
    return df.to_dict("records"), event_texts.tolist()

# ===========================
# 📌 Load Library Data (For Map)
# ===========================
LIBRARY_FILE = "data/libraries_with_coordinates.csv"  

def load_libraries(filename=LIBRARY_FILE):
    """Loading library data and parsing the `services` field"""
    try:
        df = pd.read_csv(filename)

        # ✅ 确保 `lat` 和 `lng` 存在，并删除 `NaN`
        if "lat" not in df.columns or "lng" not in df.columns:
            raise KeyError("❌ CSV file missing 'lat' or 'lng' columns!")

        df = df.dropna(subset=["lat", "lng"])
        df["lat"] = df["lat"].astype(float)
        df["lng"] = df["lng"].astype(float)

        # ✅ Parsing `services` to fix formatting errors
        def parse_services(value):
            if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
                try:
                    return json.loads(value.replace("'", '"'))  # Turning single quotes into double quotes in JSON format
                except json.JSONDecodeError:
                    print(f"⚠️ JSON parsing failure: {value}")  # Print faulty values
                    return []
            return []  

        df["services"] = df["services"].apply(parse_services)
        return df.to_dict(orient="records")

    except FileNotFoundError:
        print("❌ CSV Document not found!")
        return []
    except Exception as e:
        print(f"❌ Failure to parse CSV: {e}")
        return []

# Load data
events, event_documents = load_events()
libraries = load_libraries()

print(f"✅ Loaded {len(events)} events.")
print(f"✅ Loaded {len(libraries)} libraries.")

# ===========================
# 📌 Initialize search models
# ===========================

d = {
    "and": "&", "AND": "&",
    "or": "|", "OR": "|",
    "not": "1 -", "NOT": "1 -",
    "(": "(", ")": ")"
}

cv_events = CountVectorizer(lowercase=True, binary=True, token_pattern=r"(?u)\b\w+\b", ngram_range=(1,2))
cv_events.fit(event_documents)
sparse_matrix_events = cv_events.transform(event_documents)
t2i_events = cv_events.vocabulary_

tfidf_vectorizer_events = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,3))
tfidf_matrix_events = tfidf_vectorizer_events.fit_transform(event_documents)

semantic_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
doc_embeddings_events = semantic_model.encode(event_documents, convert_to_tensor=False)
# ===========================
# 📌 Search API
# ===========================
def classify_event(description):
    """
    Uses Zero-Shot Classification (Hugging Face) to classify event descriptions.

    Parameters:
        description (str): The event description.

    Returns:
        str: The predicted category label.
    """
    candidate_labels = ["sports", "music", "technology", "art", "education", "business", "health", "science"]
    
    result = classifier(description, candidate_labels=candidate_labels, multi_label=False)
    
    return result["labels"][0]  

def get_search_results(query, mode, top_n=20):
    """
    Perform search based on the selected mode and classify event categories.

    Parameters:
        query (str): The search query entered by the user.
        mode (str): The search method to use ("boolean", "tfidf", or "semantic").
        top_n (int): The maximum number of results to return.

    Returns:
        list: A list of search results with predicted categories.
    """
    if mode == "boolean":
        results = search_boolean(query, top_n)
    elif mode == "tfidf":
        results = search_tfidf(query, top_n)
    elif mode == "semantic":
        results = search_semantic(query, top_n)
    else:
        return []  # Return an empty list if an invalid mode is provided
    
    print(f"🔵 Mode: {mode} | Query: {query} | Results: {len(results) if isinstance(results, list) else 'summary'} items")

    # ✅ Sorting with Zero-Shot
    for result in results:
        if "description" in result and result["description"]:  
            result["category"] = classify_event(result["description"])
        else:
            result["category"] = "Unknown"


    return results

def safe_json(obj):
    """
    Recursively traverse the data and convert NaN values to None (null in JSON).
    
    Parameters:
        obj (any): The object (list, dict, float, etc.) to process.
    
    Returns:
        any: The processed object with NaN values replaced by None.
    """
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()} # Recursively process dictionary values
    elif isinstance(obj, list):
        return [safe_json(v) for v in obj] # Recursively process list elements
    elif isinstance(obj, float) and (np.isnan(obj) or obj == float("nan")):
        return None # Replace NaN with None to avoid JSON errors
    else:
        return obj # Return the object unchanged if no modification is needed

# ===========================
# 📌 Flask API Routes
# ===========================
@app.route("/libraries", methods=["GET"])
def get_libraries():
    """Returns JSON data for all libraries"""
    try:
        libraries = load_libraries(LIBRARY_FILE)
        if not libraries:
            return jsonify({"error": "Unable to load library data"}), 500

        return jsonify(libraries)

    except Exception as e:
        return jsonify({"error": f"server error: {str(e)}"}), 500

@app.route("/")
def home():
    """
    Render the homepage.
    
    Returns:
        HTML: The welcome page template.
    """
    return render_template("welcome.html")

@app.route("/results")
def results():
    """
    Render the search results page.
    
    Returns:
        HTML: The results page template.
    """
    return render_template("results.html")

@app.route("/search", methods=["GET"])
def search():
    """
    Perform a search query based on user input and return JSON results.
    
    Query Parameters:
        - query (str): The search keyword(s).
        - category (str): The category to search in ("events", "services", or "libraries").
        - mode (str): The search method to use ("boolean", "tfidf", or "semantic").
    
    Returns:
        JSON: A dictionary containing the search results and optional visualization URLs.
    """
    query = request.args.get("query", "").strip()
    mode = request.args.get("mode", "boolean")

    # Return an error response if no query is provided
    if not query:
        return jsonify({"error": "No query provided"}), 400
    print(f"🔍 Received search request: {query}, mode: {mode}")  # Debug 
    
    # Retrieve search results
    search_results = get_search_results(query, mode)

    # Ensure search_results is always a list (avoid JSON errors)
    if not isinstance(search_results, list):
        search_results = []  

    # Variables to store optional visualization URLs
    wordcloud_url = generate_wordcloud(search_results)
    category_pie_url = generate_category_pie_chart(search_results)

    # Return JSON response containing the search results and visualization links
    return jsonify({
        "results": search_results,
        "wordcloud": wordcloud_url or "",
        "category_pie": category_pie_url or ""
    })

# ===========================
# 📌 Visualization Functions
# ===========================
def generate_wordcloud(search_results):
    """
    Generate a word cloud from the event descriptions.

    Parameters:
        search_results (list): List of search results containing event descriptions.

    Returns:
        str: The file path of the saved word cloud image.
    """
    if not search_results or not any("description" in result for result in search_results):
        return None  
    
    # Combine all event descriptions
    text = " ".join([result["description"] for result in search_results if "description" in result])

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

    # Save the plot
    plot_path = "static/wordcloud.png"
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(plot_path, format="png")  
    plt.close()

    return url_for("static", filename=os.path.basename(plot_path))

def generate_category_pie_chart(search_results):
    """
    Generate a pie chart for event categories.

    Parameters:
        search_results (list): List of search results containing event categories.

    Returns:
        str: The file path of the saved pie chart image.
    """
    if not search_results or not any("category" in result for result in search_results):
        return None  

    # Count event categories
    category_counts = Counter([result["category"] for result in search_results if "category" in result])

    # Generate pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(category_counts.values(), labels=category_counts.keys(), autopct="%1.1f%%", startangle=140)

    # Save the plot
    plot_path = "static/category_pie.png"
    plt.savefig(plot_path, format="png")  
    plt.close()

    return url_for("static", filename=os.path.basename(plot_path))


# ===========================
# 📌 Helper functions
# ===========================
def clean_results(results):
    """
    Iterate through search results and replace NaN values with None (to prevent JSON errors).
    
    Parameters:
        results (list): List of search results containing possible NaN values.
    
    Returns:
        list: Processed search results with NaN replaced by None.
    """
    for item in results:
        for key, value in item.items():
            if isinstance(value, float) and np.isnan(value):  # Check if value is NaN
                item[key] = None  # Replace NaN with None
    return results

def rewrite_token(t, t2i):
    """Convert a query token into a valid Boolean search expression."""
    d = {"and": "&", "AND": "&", "or": "|", "OR": "|", "not": "1 -", "NOT": "1 -", "(": "(", ")": ")"}
    
    if t in d:
        return d[t]
    if t in t2i:
        return f'td_matrix[t2i["{t}"]]'
    # Return an empty array for unknown terms
    return 'np.zeros(td_matrix.shape[1], dtype=int)' 

def rewrite_query(query, t2i):
    """Rewrite the query for Boolean search."""
    return " ".join(rewrite_token(t, t2i) for t in query.split())

def stem_query(query):
    """a, Perform stemming on the query."""
    words = nltk.word_tokenize(query)
    return " ".join([stemmer.stem(word) for word in words])


def preprocess_query(query):
    """Process query to extract exact matches and apply stemming to remaining words."""
    exact_matches = re.findall(r'"(.*?)"', query)  # Extract quoted phrases
    remaining_text = re.sub(r'"(.*?)"', "", query).strip()
    remaining_words = remaining_text.split() if remaining_text else [] 

    stemmed_words = [stemmer.stem(word) for word in remaining_words]
    return exact_matches + stemmed_words

def wildcard_search(query, t2i):
    """d, Handle wildcard searches by converting `*` into a regex pattern."""
    regex_query = query.replace("*", ".*")  
    matched_terms = [term for term in t2i.keys() if re.fullmatch(regex_query, term)]

    print(f"🔎 Wildcard search query: {query}")
    print(f"🟢 Wildcard Matched terms: {matched_terms}")  

    return matched_terms

def merge_duplicate_events(results):
    """Merge events with the same title and location but different dates."""
    merged_events = {}
    
    for event in results:
        key = (event["title"], event["location"])  
        if key in merged_events:
            merged_events[key]["date"].append(event["date"])  
        else:
            merged_events[key] = event
            merged_events[key]["date"] = [event["date"]]  

    for event in merged_events.values():
        event["date"] = ", ".join(event["date"])  
    return list(merged_events.values())  

# ===========================
# 📌 Boolean Search
# ===========================
class NOTError(Exception):
    """Exception raised when NOT operation affects all documents."""
    pass

def search_boolean(query, top_n=20):
    """Perform Boolean search in Flask API (JSON Response)."""
    try:
        t2i = t2i_events
        td_matrix = sparse_matrix_events.T.todense()
        data_source = events

        print(f"🟢 Available words in t2i_events: {list(t2i_events.keys())[:10]}")
        print(f"📊 Shape of td_matrix: {td_matrix.shape}")


        if not any(op in query.upper() for op in ["AND", "OR", "NOT"]) and len(query.split()) > 1:
            return []

        query_terms = [word.lower() for word in query.split() if word.lower() in t2i]
        if not query_terms:
            print(f"❌ No valid terms found in query '{query}'")
            return []

        hits_matrix = np.zeros(td_matrix.shape[1], dtype=int)  
        for term in query.split():
            if term.lower() in t2i:
                print(f"🔍 Searching for term: {term.lower()} (index {t2i[term.lower()]})")
                term_vector = np.asarray(td_matrix[t2i[term.lower()], :]).flatten()  
                print(f"📊 Term Vector: {term_vector}")  
                hits_matrix |= term_vector 

        print(f"🎯 Hits matrix after search: {hits_matrix}")  
        hits = list(hits_matrix.nonzero()[0])  
        if not hits:
            print(f"❌ No results found for '{query}'")
            return []


        results = []
        for idx in hits[:top_n]:
            item = data_source[idx]
            results.append({
                "title": item.get("title", "N/A"),
                "location": item.get("location", "N/A"),
                "date": item.get("date", "N/A"),
                "description": item.get("description_translated", "N/A")[:200] + "...",
                "url": item.get("link", "#"),
                "image_url": item.get("image url", "#"),
                "score": 1.0
            })
        print(f"🟢 Before merge: {len(results)} results")
        results = merge_duplicate_events(results)
        print(f"🟢 After merge: {len(results)} results")
        return clean_results(results)  
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return []
# ===========================
# 📌 TF-IDF Search (with a,Stemming, b,Exact Match, d,Wildcard)
# ===========================
def search_tfidf(query, top_n=20):
    """Perform TF-IDF search, supporting stemming, exact match, wildcard, and multi-word phrases for a specific category."""
    try:
        tfidf_vectorizer = tfidf_vectorizer_events
        tfidf_matrix = tfidf_matrix_events
        data_source = events

        if "*" in query:
            wildcard_matches = wildcard_search(query, t2i_events)
            if not wildcard_matches:
                raise ValueError("❌ No matching wildcard terms found.")
            query_text = " ".join(wildcard_matches)
        else:
            processed_query = preprocess_query(query)
            print(f"🔵 Processed query {processed_query}")  # Debug
            query_text = " ".join(processed_query)
            print(f"✅ Final query text {query_text}")  # Debug


        query_vector = tfidf_vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()


        ranked_indices = similarities.argsort()[::-1][:top_n]
        ranked_indices = [idx for idx in ranked_indices if similarities[idx] > 0]
        hits = ranked_indices[:top_n]  

        print(f"📊 Ranked indices {ranked_indices}")  # Debug
        print(f"📊 Similarity scores  {[similarities[idx] for idx in ranked_indices]}")  # Debug


        results = []
        for idx in hits[:top_n]:
            item = data_source[idx]
            similarity_score = float(similarities[idx])
            results.append({
                "title": item.get("title", "N/A"),
                "location": item.get("location", "N/A"),
                "date": item.get("date", "N/A"),
                "description": item.get("description_translated", "N/A")[:200] + "...",
                "url": item.get("link", "#"),
                "image_url": item.get("image url", "#"),
                "score": similarity_score
            })
            
        results = merge_duplicate_events(results)
        return clean_results(results)  

    except Exception as e:
        print(f"❌ Error processing TF-IDF query '{query}' for events: {e}")
        return []
# ===========================
# 📌 Semantic Search (BERT)
# ===========================
def search_semantic(query, top_n=20):
    """Perform semantic search using BERT embeddings for a specific category."""
    try:
        doc_embeddings = doc_embeddings_events
        data_source = events
        
        query_embedding = semantic_model.encode([query], convert_to_tensor=False)
        similarities = cosine_similarity(query_embedding, doc_embeddings).flatten()

        ranked_indices = similarities.argsort()[::-1][:top_n]
        ranked_indices = [idx for idx in ranked_indices if similarities[idx] > 0]
        hits = ranked_indices[:top_n]  

        print(f"📊 Ranked indices {ranked_indices}")  # Debug
        print(f"📊 Similarity scores {[similarities[idx] for idx in ranked_indices]}")  # Debug

        results = []
        for idx in hits[:top_n]:
            item = data_source[idx]
            results.append({
                "title": item.get("title", "N/A"),
                "location": item.get("location", "N/A"),
                "date": item.get("date", "N/A"),
                "description": item.get("description_translated", "N/A")[:200] + "...",
                "url": item.get("link", "#"),
                "image_url": item.get("image url", "#"),
                "score": float(similarities[idx])
            })

        results = merge_duplicate_events(results)
        return clean_results(results)  
    except Exception as e:
        print(f"❌ Error processing Semantic query '{query}' for events: {e}")
        return []

# ===========================
# 📌 Run Flask Server
# ===========================
if __name__ == "__main__":
    app.run(debug=True)
