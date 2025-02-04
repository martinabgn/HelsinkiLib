# ===========================
# 📌 Import necessary modules
# ===========================
import requests
from bs4 import BeautifulSoup
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
nltk.download("punkt")  
stemmer = PorterStemmer()

# ===========================
# 📌 Fetch documents from URL
# ===========================
def fetch_documents(url):
    """Fetch articles from the given URL and return a dictionary with article names and content."""
    try:
        response = requests.get(url)
        response.raise_for_status()  
        content = response.text
        soup = BeautifulSoup(content, 'html.parser')
        articles = soup.find_all('article')

        article_dict = {}
        for article in articles:
            name = article.get("name")
            if not name:
                name = f"Unnamed_{len(article_dict) + 1}"
            content = article.get_text(separator=" ", strip=True)
            if name in article_dict:
                name = f"{name}_{len(article_dict) + 1}"
            article_dict[name] = content
        return article_dict
    except requests.RequestException as e:
        print(f"Error fetching documents from URL: {e}")
        return {}

# ===========================
# 📌 Load dataset
# ===========================
url = "https://raw.githubusercontent.com/martinabgn/HelsinkiLib/week2/enwiki-20181001-corpus.1000-articles.txt"
article_dict = fetch_documents(url)
    
if not article_dict:
    print("No documents to process. Exiting.")
    exit()

documents, names = list(article_dict.values()), list(article_dict.keys())

# ===========================
# 📌 Initialize search models
# ===========================
# Boolean Search Model
cv = CountVectorizer(lowercase=True, binary=True, token_pattern=r"(?u)\b\w+\b")  
sparse_matrix = cv.fit_transform(documents)

# TF-IDF Search Model
tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Semantic Search Model (BERT)
semantic_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
doc_embeddings = semantic_model.encode(documents, convert_to_tensor=False)

# ===========================
# 📌 Data structures for search
# ===========================
terms = cv.get_feature_names_out()
t2i = cv.vocabulary_

dense_matrix = sparse_matrix.todense()
td_matrix = dense_matrix.T

# Operator dictionary for Boolean search
d = {"and": "&", "AND": "&",
     "or": "|", "OR": "|",
     "not": "1 -", "NOT": "1 -",
     "(": "(", ")": ")"}

# ===========================
# 📌 Helper functions
# ===========================
def rewrite_token(t):
    """Convert a query token into a valid Boolean search expression."""
    if t in d:
        return d[t]
    if t in t2i:
        return f'td_matrix[t2i["{t}"]]'
    # Return an empty array for unknown terms
    return 'np.zeros(td_matrix.shape[1], dtype=int)' 

def rewrite_query(query):
    """Rewrite the query for Boolean search."""
    return " ".join(rewrite_token(t) for t in query.split())

def stem_query(query):
    """a, Perform stemming on the query."""
    words = nltk.word_tokenize(query)
    return " ".join([stemmer.stem(word) for word in words])

def preprocess_query(query):
    """b, Extract exact matches (quoted words) and apply stemming to the rest."""
    # Extract quoted words
    exact_matches = re.findall(r'"(.*?)"', query)  
    remaining_text = re.sub(r'"(.*?)"', "", query).strip()
    remaining_words = remaining_text.split() if remaining_text else [] 

    stemmed_words = [stemmer.stem(word) for word in remaining_words]
    return exact_matches + stemmed_words

def wildcard_search(query):
    """d, Handle wildcard searches by converting `*` into a regex pattern."""
    regex_query = query.replace("*", ".*")  
    matched_terms = [term for term in t2i.keys() if re.fullmatch(regex_query, term)]

    print(f"🔎 Wildcard search query: {query}")
    print(f"🟢 Wildcard Matched terms: {matched_terms}")  

    return matched_terms

# ===========================
# 📌 Boolean Search
# ===========================
class NOTError(Exception):
    """Exception raised when NOT operation affects all documents."""
    pass

def search_query(query, top_n=5, truncate_m=200):
    """Perform Boolean search."""
    try:
        # Validate query format
        if not any(op in query for op in ["AND", "OR", "NOT"]) and len(query.split()) > 1:
            raise SyntaxError
        
        # Check for unknown words
        unknown_words = [word for word in query.split() if word not in d and word not in t2i]
        if unknown_words:
            raise ValueError(f"Unfortunately, we could not find {' '.join(unknown_words)} in our data. Please, try another query.")

        # Process Boolean query
        rewritten_query = rewrite_query(query)
        hits_matrix = eval(rewritten_query)  
        
        # Check NOT operation
        if "NOT" in query.upper() and np.all(hits_matrix):
            raise NOTError
        
        hits_list = list(hits_matrix.nonzero()[1])  
        total_hits = len(hits_list)

        if total_hits == 0:
            print("No matching documents found.")
            return

        print(f"Query: {query}")
        print(f"Total matching documents: {total_hits}")
        print(f"Showing top {min(top_n, total_hits)} documents:\n")

        for i, doc_idx in enumerate(hits_list[:top_n]):
            truncated_content = " ".join(documents[doc_idx].split()[:truncate_m])  
            print(f"Matching doc #{i+1} (Index {doc_idx}): {truncated_content}...\n")
        if total_hits > top_n:
            print(f"Only showing the top {top_n} results. Refine your query to see more.")

    except SyntaxError:
        print("❌ Please enter your query in the correct format: word OPERATOR word (e.g., hello AND world)")

    # Handle the unknownword
    except ValueError as e:  
        print(f"❌ {e}")

    except NOTError:  
        print("⚠️ Wow! Your query term appears in EVERY document in our data!")
    except Exception as e:
        print(f"Error processing query '{query}': {e}")

# ===========================
# 📌 TF-IDF Search (with a,Stemming, b,Exact Match, d,Wildcard)
# ===========================
def search_tfidf(query, top_n=5, truncate_m=200):
    """Perform TF-IDF search, supporting stemming, exact match, and wildcard search."""
    try:
        if "*" in query:
            wildcard_matches = wildcard_search(query)
            if not wildcard_matches:
                print("❌ No matching wildcard terms found.")
                return
            query_text = " ".join(wildcard_matches)
        else:
            processed_query = preprocess_query(query)
            # View query after stemming
            print(f"🔵 Processed query: {processed_query}")  
            query_text = " ".join(processed_query)
            # View the text that is ultimately used to calculate TF-IDF.
            print(f"✅ Final query text: {query_text}")  
        
        # Perform TF-IDF calculation on the query
        query_vector = tfidf_vectorizer.transform([query_text])  
        # Calculate the similarity between the query and all documents
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()  
        
        # Sort by descending similarity
        ranked_indices = similarities.argsort()[::-1]  
        # Filter out documents with a similarity of 0
        ranked_indices = [idx for idx in ranked_indices if similarities[idx] > 0]  
        
        # View sorted index
        print(f"📊 Ranked indices: {ranked_indices}") 
        # View similarity score
        print(f"📊 Similarity scores: {similarities[ranked_indices]}")  

        total_hits = len(ranked_indices)
        if total_hits == 0:
            print("No matching documents found.")
            return

        print(f"🔍 Query: {query}")
        print(f"📄 Total matching documents: {total_hits}")
        print(f"📌 Showing top {min(top_n, total_hits)} documents:\n")

        for i, doc_idx in enumerate(ranked_indices[:top_n]):
            truncated_content = " ".join(documents[doc_idx].split()[:truncate_m])
            print(f"🔹 Matching doc #{i+1} (Index {doc_idx}, Score: {similarities[doc_idx]:.4f}): {names[doc_idx]}")
            print(f"📄 {truncated_content}...\n")

        if total_hits > top_n:
            print(f"🔽 Only showing the top {top_n} results. Refine your query to see more.")

    except ValueError as e:
        print(f"❌ {e}")  
    except Exception as e:
        print(f"⚠️ Error processing query '{query}': {e}")

# ===========================
# 📌 Semantic Search (BERT)
# ===========================
def search_semantic(query, top_n=5, truncate_m=200):
    """Perform semantic search using BERT embeddings."""
    try:
        # Compute the BERT embedding for the query
        query_embedding = semantic_model.encode([query], convert_to_tensor=False)  
        # Calculate similarity
        similarities = cosine_similarity(query_embedding, doc_embeddings).flatten()  
        # Sort by similarity
        ranked_indices = similarities.argsort()[::-1]  
        # Filter out documents with a similarity of 0
        ranked_indices = [idx for idx in ranked_indices if similarities[idx] > 0]  

        total_hits = len(ranked_indices)
        if total_hits == 0:
            print("No matching documents found.")
            return

        print(f"🔍 Query: {query} (Semantic Search)")
        print(f"📄 Total matching documents: {total_hits}")
        print(f"📌 Showing top {min(top_n, total_hits)} documents:\n")

        for i, doc_idx in enumerate(ranked_indices[:top_n]):
            truncated_content = " ".join(documents[doc_idx].split()[:truncate_m])
            print(f"🔹 Matching doc #{i+1} (Index {doc_idx}, Score: {similarities[doc_idx]:.4f}): {names[doc_idx]}")
            print(f"📄 {truncated_content}...\n")
        if total_hits > top_n:
            print(f"🔽 Only showing the top {top_n} results. Refine your query to see more.")

    except Exception as e:
        print(f"⚠️ Error processing query '{query}': {e}")


# ===========================
# 📌 Main Program Loop
# ===========================
while True:
    print("\n🔍 Choose a search mode:")
    print("1️⃣ Boolean Search")
    print("2️⃣ TF-IDF Search")
    print("3️⃣ Semantic Search")
    print("4️⃣ Quit")

    choice = input("Enter your choice (1/2/3/4): ").strip()

    if choice == "4":
        print("👋 Exiting program.")
        break

    user_query = input("Enter your query: ").strip()
    if choice == "1":
        search_query(user_query, top_n=5, truncate_m=50)
    elif choice == "2":
        search_tfidf(user_query, top_n=5, truncate_m=50)
    elif choice == "3":
        search_semantic(user_query, top_n=5, truncate_m=50)
    else:
        print("⚠️ Invalid choice! Please select 1, 2, or 3.")