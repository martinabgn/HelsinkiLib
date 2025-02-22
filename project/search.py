# ===========================
# 📌 Import necessary modules
# ===========================
from flask import Flask, render_template, request, jsonify, url_for
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
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

# ===========================
# 📌 Initialize NLP components
# ===========================
#nltk.download("punkt")  
stemmer = PorterStemmer()
nlp = spacy.load("en_core_web_sm")

# ===========================
# 📌 Load dataset from csv
# ===========================
def load_events(filename="data/helmet_all_events_with_translated_descriptions.csv"):
    """Load event data from a CSV file and format it for search processing."""
    df = pd.read_csv(filename, encoding="utf-8")  
    df.columns = df.columns.str.strip().str.lower()  
    # Prepare event descriptions for searching
    event_texts = df["title"].astype(str) + " " + df["location"].astype(str)
    event_names = df["title"].astype(str)

    return df.to_dict("records"), event_texts.tolist(), event_names.tolist()  

def load_library_services(filename="data/library_services_binary.csv"):
    """Load library service data from a CSV file with automatic encoding detection."""
    df = pd.read_csv(filename, encoding="ISO-8859-1")
    df.columns = df.columns.str.strip().str.lower()  
    if "library name" not in df.columns:
        raise KeyError(f"❌ Missing 'library name' in CSV file, Columns found: {df.columns}")

    # Extract service names from the dataset
    service_columns = [col for col in df.columns if col != "library name"]

    services_list = []
    for _, row in df.iterrows():
        library = row["library name"]
        available_services = [service for service in service_columns if str(row[service]).strip().lower() in ["yes", "1", "true"]]
        services_list.append({"library_name": library, "services": ", ".join(available_services)})

    service_texts = [f"{item['library_name']} {' '.join(item['services'].split(','))}" for item in services_list if item["library_name"] != "Library Name"]
    library_names = [item["library_name"] for item in services_list]

    return services_list, service_texts, library_names

def load_libraries(filename="data/helmet_library_details.csv"):
    """Load library details from a CSV file with automatic encoding detection."""
    df = pd.read_csv(filename, encoding="utf-8-sig")  
    df.columns = df.columns.str.strip().str.lower()  

    if "library name" in df.columns:
        df = df.rename(columns={"library name": "library_name"})
    else:
        raise KeyError(f"❌ Missing 'library name' in CSV file. Columns found: {df.columns}")

    if "address" not in df.columns:
        raise KeyError(f"❌ Missing 'library name' in CSV file. Columns found: {df.columns}")

    library_texts = df["library_name"].astype(str) + " " + df["address"].astype(str)
    library_names = df["library_name"].astype(str)

    return df.to_dict("records"), library_texts.tolist(), library_names.tolist()

events, event_documents, event_names = load_events()
library_services, service_documents, service_names = load_library_services()
libraries, library_documents, library_names = load_libraries()

print(f"✅ Loaded {len(events)} events.")
print(f"✅ Loaded {len(library_services)} library services.")
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

# Boolean Search (CountVectorizer)
cv_events = CountVectorizer(lowercase=True, binary=True, token_pattern=r"(?u)\b\w+\b", ngram_range=(1,2))
cv_services = CountVectorizer(lowercase=True, binary=True, token_pattern=r"(?u)\b\w+\b", ngram_range=(1,2))
cv_libraries = CountVectorizer(lowercase=True, binary=True, token_pattern=r"(?u)\b\w+\b", ngram_range=(1,2))

print(f"📊 Total number of event_documents: {len(event_documents)}")
print(f"🟢 Sample event_documents: {event_documents[:5]}")  

cv_events.fit(event_documents)
cv_services.fit(service_documents)
cv_libraries.fit(library_documents)

sparse_matrix_events = cv_events.transform(event_documents)
sparse_matrix_services = cv_services.transform(service_documents)
sparse_matrix_libraries = cv_libraries.transform(library_documents)

t2i_events = cv_events.vocabulary_
t2i_services = cv_services.vocabulary_
t2i_libraries = cv_libraries.vocabulary_

# TF-IDF Vectorizer for keyword-based search
tfidf_vectorizer_events = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,3))
tfidf_vectorizer_services = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,3))
tfidf_vectorizer_libraries = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,3))

print(f"✅ Number of words in t2i_events: {len(t2i_events)}")
print(f"🟢 Sample words: {list(t2i_events.keys())[:10]}")  

tfidf_matrix_events = tfidf_vectorizer_events.fit_transform(event_documents)
tfidf_matrix_services = tfidf_vectorizer_services.fit_transform(service_documents)
tfidf_matrix_libraries = tfidf_vectorizer_libraries.fit_transform(library_documents)


# **Semantic Search Model (BERT)**
semantic_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
doc_embeddings_events = semantic_model.encode(event_documents, convert_to_tensor=False)
doc_embeddings_services = semantic_model.encode(service_documents, convert_to_tensor=False)
doc_embeddings_libraries = semantic_model.encode(library_documents, convert_to_tensor=False)

# ===========================
# 📌 Search API
# ===========================
def get_search_results(query, category, mode, top_n=20):
    """
    Perform search based on the selected mode.
    
    Parameters:
        query (str): The search query entered by the user.
        category (str): The category of data to search within ("events", "services", or "libraries").
        mode (str): The search method to use ("boolean", "tfidf", or "semantic").
        top_n (int): The maximum number of results to return.
    
    Returns:
        list: A list of search results matching the query.
    """
    if mode == "boolean":
        results = search_boolean(query, category, top_n)
    elif mode == "tfidf":
        results = search_tfidf(query, category, top_n)
    elif mode == "semantic":
        results = search_semantic(query, category, top_n)
    else:
        return []  # Return an empty list if an invalid mode is provided
    
    print(f"🔵 Mode: {mode} | Query: {query} | Category: {category} | Results: {len(results)} items")
    return results  # Return the search results as a list

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
# 📌 Flask API Routes ： This flask part used for debugging can be deleted after the flask code file is complete.
# ===========================
    
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
    category = request.args.get("category", "events")
    mode = request.args.get("mode", "boolean")

    # Return an error response if no query is provided
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    # Retrieve search results
    search_results = get_search_results(query, category, mode)

    # Ensure search_results is always a list (avoid JSON errors)
    if not isinstance(search_results, list):
        search_results = []  

    # Variables to store optional visualization URLs
    score_plot_url = None
    ner_plot_url = None

    # Generate a histogram for TF-IDF search results if applicable
    if mode == "tfidf" and search_results:
        plot_path = generate_score_distribution_plot([result["score"] for result in search_results if "score" in result])
        score_plot_url = url_for("static", filename=os.path.basename(plot_path))

    # Generate a Named Entity Recognition (NER) plot for Semantic search results if applicable
    if mode == "semantic" and search_results:
        plot_path = generate_ner_plot(search_results)
        ner_plot_url = url_for("static", filename=os.path.basename(plot_path))
    
    # Return JSON response containing the search results and visualization links
    return jsonify({
        "results": search_results,
        "score_plot": score_plot_url or "",
        "ner_plot": ner_plot_url or ""
    })


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

def generate_score_distribution_plot(scores):
    """
    Generate a histogram of search result relevance scores using Seaborn.
    
    Parameters:
        scores (list): A list of numerical relevance scores.
    
    Returns:
        str: The file path of the saved plot image.
    """
    if not scores:
        return None  # Avoid crashing if scores list is empty
    plt.figure(figsize=(7, 5))
    sns.histplot(scores, bins=10, kde=True, color="blue", edgecolor="black") # Plot histogram
    
    plt.xlabel("Relevance Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Search Relevance Scores")
    
    plot_path = "static/score_plot.png"
    plt.savefig(plot_path, format="png")  
    plt.close()
    print("📊 Final Scores for Histogram:", scores)
    
    return plot_path  


def generate_ner_plot(search_results):
    """
    Perform Named Entity Recognition (NER) on search results and generate a bar chart.
    
    Parameters:
        search_results (list): List of search results containing text descriptions.
    
    Returns:
        str: The file path of the saved NER plot image.
    """
    # Ensure there are results with descriptions before proceeding
    if not search_results or not any("description" in result for result in search_results):
        return None  
    # Concatenate all descriptions into a single text string
    text = " ".join([result["description"] for result in search_results if "description" in result])
    doc = nlp(text)

    # Count occurrences of each entity type
    entity_labels = [ent.label_ for ent in doc.ents]
    entity_counts = Counter(entity_labels)
    # Create a bar plot for entity distribution
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(entity_counts.keys()), y=list(entity_counts.values()), palette="viridis")
    
    plt.xlabel("Entity Type")
    plt.ylabel("Count")
    plt.title("Named Entity Recognition Analysis")
    
    plot_path = "static/ner_plot.png" 
    plt.savefig(plot_path, format="png")  
    plt.close()

    return plot_path  

# ===========================
# 📌 Boolean Search
# ===========================
class NOTError(Exception):
    """Exception raised when NOT operation affects all documents."""
    pass

def search_boolean(query, category, top_n=20):
    """Perform Boolean search in Flask API (JSON Response)."""
    try:
        if category == "events":
            t2i = t2i_events
            td_matrix = sparse_matrix_events.T.todense()
            data_source = events
        elif category == "services":
            t2i = t2i_services
            td_matrix = sparse_matrix_services.T.todense()
            data_source = library_services
        elif category == "libraries":
            t2i = t2i_libraries
            td_matrix = sparse_matrix_libraries.T.todense()
            data_source = libraries
        else:
            return [] 
        
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
            if category == "events":
                results.append({
                    "title": item.get("title", "N/A"),
                    "location": item.get("location", "N/A"),
                    "date": item.get("date", "N/A"),
                    "description": item.get("description", "N/A")[:200] + "...",
                    "url": item.get("link", "#"),  
                    "image_url": item.get("image url", "#"),  
                    "score": 1.0
                })
            elif category == "services":
                results.append({
                    "library_name": item.get("library_name", "N/A"),
                    "services": item.get("services", "N/A"),
                    "score": 1.0
                })
            elif category == "libraries":
                results.append({
                    "library_name": item.get("library_name", "N/A"),
                    "address": item.get("address", "N/A"),
                    "contact_info": item.get("contact info", "N/A"),  
                    "services": " | ".join(str(item.get("services", "N/A")).split()),
                    "score": 1.0
                })

        if category == "events":
            print(f"🟢 Before merge: {len(results)} results")
            results = merge_duplicate_events(results)
            print(f"🟢 After merge: {len(results)} results")

        return clean_results(results)  
    except Exception as e:
        return [] 
# ===========================
# 📌 TF-IDF Search (with a,Stemming, b,Exact Match, d,Wildcard)
# ===========================
def search_tfidf(query, category, top_n=20):
    """Perform TF-IDF search, supporting stemming, exact match, wildcard, and multi-word phrases for a specific category."""
    try:

        if category == "events":
            tfidf_vectorizer = tfidf_vectorizer_events
            tfidf_matrix = tfidf_matrix_events
            data_source = events
        elif category == "services":
            tfidf_vectorizer = tfidf_vectorizer_services
            tfidf_matrix = tfidf_matrix_services
            data_source = library_services
        elif category == "libraries":
            tfidf_vectorizer = tfidf_vectorizer_libraries
            tfidf_matrix = tfidf_matrix_libraries
            data_source = libraries
        else:
            raise ValueError("❌ Invalid category! Please choose 'events', 'services', or 'libraries'.")


        if "*" in query:
            wildcard_matches = wildcard_search(query, t2i_libraries if category == "libraries" else t2i_events if category == "events" else t2i_services)
            if not wildcard_matches:
                raise ValueError("❌ No matching wildcard terms found.")
            query_text = " ".join(wildcard_matches)
        else:
            processed_query = preprocess_query(query)
            print(f"🔵 Processed query ({category}): {processed_query}")  # Debug
            query_text = " ".join(processed_query)
            print(f"✅ Final query text ({category}): {query_text}")  # Debug


        query_vector = tfidf_vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()


        ranked_indices = similarities.argsort()[::-1][:top_n]
        ranked_indices = [idx for idx in ranked_indices if similarities[idx] > 0]
        hits = ranked_indices[:top_n]  

        print(f"📊 Ranked indices ({category}): {ranked_indices}")  # Debug
        print(f"📊 Similarity scores ({category}): {[similarities[idx] for idx in ranked_indices]}")  # Debug


        results = []
        for idx in hits[:top_n]:
            item = data_source[idx]
            similarity_score = float(similarities[idx])  
            if category == "events":
                results.append({
                    "title": item.get("title", "N/A"),
                    "location": item.get("location", "N/A"),
                    "date": item.get("date", "N/A"),
                    "description": item.get("description", "N/A")[:200] + "...",
                    "url": item.get("link", "#"),  
                    "image_url": item.get("image url", "#"),  
                    "score": similarity_score  
                })
            elif category == "services":
                results.append({
                    "library_name": item.get("library_name", "N/A"),
                    "services": item.get("services", "N/A"),
                    "score": similarity_score  
                })
            elif category == "libraries":
                results.append({
                    "library_name": item.get("library_name", "N/A"),
                    "address": item.get("address", "N/A"),
                    "contact_info": item.get("contact info", "N/A"),  
                    "services": " | ".join(str(item.get("services", "N/A")).split()),
                    "score": similarity_score  
                })


        if category == "events":
            results = merge_duplicate_events(results)

        return clean_results(results)  

    except ValueError as e:
        print(f"⚠️ ValueError: {e}")
    except Exception as e:
        print(f"❌ Error processing TF-IDF query '{query}' for {category}: {e}")

    return []  
# ===========================
# 📌 Semantic Search (BERT)
# ===========================
def search_semantic(query, category, top_n=20):
    """Perform semantic search using BERT embeddings for a specific category."""
    try:

        if category == "events":
            doc_embeddings = doc_embeddings_events
            data_source = events
        elif category == "services":
            doc_embeddings = doc_embeddings_services
            data_source = library_services
        elif category == "libraries":
            doc_embeddings = doc_embeddings_libraries
            data_source = libraries
        else:
            raise ValueError("❌ Invalid category! Please choose 'events', 'services', or 'libraries'.")

        
        query_embedding = semantic_model.encode([query], convert_to_tensor=False)


        similarities = cosine_similarity(query_embedding, doc_embeddings).flatten()


        ranked_indices = similarities.argsort()[::-1][:top_n]
        ranked_indices = [idx for idx in ranked_indices if similarities[idx] > 0]
        hits = ranked_indices[:top_n]  

        print(f"📊 Ranked indices ({category}): {ranked_indices}")  # Debug
        print(f"📊 Similarity scores ({category}): {[similarities[idx] for idx in ranked_indices]}")  # Debug


        results = []
        for idx in hits[:top_n]:
            item = data_source[idx]
            if category == "events":
                results.append({
                    "title": item.get("title", "N/A"),
                    "location": item.get("location", "N/A"),
                    "date": item.get("date", "N/A"),
                    "description": item.get("description", "N/A")[:200] + "...",
                    "url": item.get("link", "#"),  
                    "image_url": item.get("image url", "#"),  
                    "score": float(similarities[idx])
                })
            elif category == "services":
                results.append({
                    "library_name": item.get("library_name", "N/A"),
                    "services": item.get("services", "N/A"),
                    "score": float(similarities[idx]) 
                })
            elif category == "libraries":
                results.append({
                    "library_name": item.get("library_name", "N/A"),
                    "address": item.get("address", "N/A"),
                    "contact_info": item.get("contact info", "N/A"),  
                    "services": " | ".join(str(item.get("services", "N/A")).split()),
                    "score": float(similarities[idx])
                })


        if category == "events":
            results = merge_duplicate_events(results)

        return clean_results(results)  

    except ValueError as e:
        print(f"⚠️ ValueError: {e}")
    except Exception as e:
        print(f"❌ Error processing Semantic query '{query}' for {category}: {e}")

    return jsonify(results)  


# ===========================
# 📌 Run Flask Server
# ===========================
if __name__ == "__main__":
    app.run(debug=True)
