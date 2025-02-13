# ===========================
# 📌 Import necessary modules
# ===========================
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.stem import PorterStemmer
import re

# ===========================
# 📌 Initialize NLP components
# ===========================
#nltk.download("punkt")  
stemmer = PorterStemmer()

# ===========================
# 📌 Load dataset from csv
# ===========================
def load_events(filename="project/data/helmet_all_events_with_translated_descriptions.csv"):
    """Load event data from a CSV file and format it for search processing."""
    df = pd.read_csv(filename, encoding="utf-8")  
    df.columns = df.columns.str.strip().str.lower()  

     # Prepare event descriptions for searching
    event_texts = df["title"].astype(str) + " " + df["location"].astype(str)
    event_names = df["title"].astype(str)
    
    return df.to_dict("records"), event_texts.tolist(), event_names.tolist()

def load_library_services(filename="project/data/library_services_binary.csv"):
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

def load_libraries(filename="project/data/helmet_library_details_modified.csv"):
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

tfidf_matrix_events = tfidf_vectorizer_events.fit_transform(event_documents)
tfidf_matrix_services = tfidf_vectorizer_services.fit_transform(service_documents)
tfidf_matrix_libraries = tfidf_vectorizer_libraries.fit_transform(library_documents)


# **Semantic Search Model (BERT)**
semantic_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
doc_embeddings_events = semantic_model.encode(event_documents, convert_to_tensor=False)
doc_embeddings_services = semantic_model.encode(service_documents, convert_to_tensor=False)
doc_embeddings_libraries = semantic_model.encode(library_documents, convert_to_tensor=False)


# ===========================
# 📌 Helper functions
# ===========================
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

def search_boolean(query, category, top_n=20):
    """Perform Boolean search on a specific category (events, services, libraries)."""
    try:

        if category == "events":
            t2i = t2i_events
            td_matrix = sparse_matrix_events.todense().T
            data_source = events
        elif category == "services":
            t2i = t2i_services
            td_matrix = sparse_matrix_services.todense().T
            data_source = library_services
        elif category == "libraries":
            t2i = t2i_libraries
            td_matrix = sparse_matrix_libraries.todense().T
            data_source = libraries
        else:
            raise ValueError("❌ Invalid category! Please choose 'events', 'services', or 'libraries'.")


        if not any(op in query for op in ["AND", "OR", "NOT"]) and len(query.split()) > 1:
            raise SyntaxError("❌ Please enter your query in the correct format: 'word OPERATOR word' (e.g., 'music AND concert').")


        unknown_words = [word for word in query.split() if word not in d and word not in t2i]
        if unknown_words:
            raise ValueError(f"❌ Unfortunately, we could not find {' '.join(unknown_words)} in our {category} data. Please, try another query.")


        rewritten_query = " ".join([f'td_matrix[t2i["{t}"]]' if t in t2i else 'np.zeros(td_matrix.shape[1], dtype=int)' for t in query.split()])
        print(f"🔎 After Boolean search  ({category}): {rewritten_query}")  # Debug


        hits_matrix = eval(rewritten_query)
        print(f"📌 hits_matrix shape ({category}): {hits_matrix.shape}")
        print(f"📌 nonzero indices ({category}): {hits_matrix.nonzero()}")


        if "NOT" in query.upper() and np.all(hits_matrix):
            raise NOTError("⚠️ Your query term appears in EVERY document in our data!")


        hits = list(hits_matrix.nonzero()[1])
        if not hits:
            raise ValueError("❌ No matching results found.")


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
            results = merge_duplicate_events(results)

        return results

    except SyntaxError as e:
        print(f"⚠️ SyntaxError: {e}")
    except ValueError as e:
        print(f"⚠️ ValueError: {e}")
    except NOTError as e:
        print(f"⚠️ NOTError: {e}")
    except KeyError as e:
        print(f"⚠️ KeyError: Missing key {e} in dataset.")
    except Exception as e:
        print(f"❌ Error processing query '{query}' for {category}: {e}")

    return {}  

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
            results = merge_duplicate_events(results)

        return results

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
            results = merge_duplicate_events(results)

        return results

    except ValueError as e:
        print(f"⚠️ ValueError: {e}")
    except Exception as e:
        print(f"❌ Error processing Semantic query '{query}' for {category}: {e}")

    return [] 

# ===========================
# 📌 Main Program Loop
# ===========================
while True:
    print("\n🔍 Choose a category:")
    print("1️⃣ Events")
    print("2️⃣ Library Services")
    print("3️⃣ Libraries")
    print("4️⃣ Quit")

    category_choice = input("Enter your choice (1/2/3/4): ").strip()
    if category_choice == "4":
        print("👋 Exiting program.")
        break

    category_map = {"1": "events", "2": "services", "3": "libraries"}
    category = category_map.get(category_choice, None)

    if not category:
        print("❌ Invalid category selection! Please choose 1, 2, or 3.")
        continue

    print("\n🔍 Choose a search mode:")
    print("1️⃣ Boolean Search")
    print("2️⃣ TF-IDF Search")
    print("3️⃣ Semantic Search")

    search_mode = input("Enter your choice (1/2/3): ").strip()

    user_query = input("Enter your query: ").strip()

    if search_mode == "1":
        results = search_boolean(user_query, category)
    elif search_mode == "2":
        results = search_tfidf(user_query, category)
    elif search_mode == "3":
        results = search_semantic(user_query, category)
    else:
        print("⚠️ Invalid search mode! Please select 1, 2, or 3.")
        continue

    # 显示搜索结果
    if results:
       print("\n🔍 Search Results:")
       for i, item in enumerate(results):
            if "title" in item:  # Events
                print(f"{i+1}. {item['title']} - {item.get('location', 'Unknown Location')}")
                print(f"   📅 Date: {item.get('date', 'Unknown Date')}")
                print(f"   📝 Description: {item.get('description', 'No Description')}")
                print(f"   🔗 URL: {item.get('url', '#')}")
                print(f"   🖼️ Image: {item.get('image_url', '#')}")
                print(f"   🏆 Score: {item['score']:.4f}\n")
            elif "library_name" in item:  # Services & Libraries
                print(f"{i+1}. {item['library_name']}")
                if "services" in item:  # Library Services
                    print(f"   📚 Services: {item.get('services', 'No Services Listed')}")
                if "address" in item:  # Libraries
                    print(f"   📍 Address: {item.get('address', 'Unknown Address')}")
                    print(f"   📧 Contact: {item.get('contact_info', 'No Contact Info')}")
                    print(f"   📚 Services: {item.get('services', 'No Services Listed')}")
                    print(f"   🏆 Score: {item['score']:.4f}\n")
    else:
        print("❌ No matching results found.")
