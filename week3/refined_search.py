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

nltk.download("punkt")
stemmer = PorterStemmer()


# ===========================
# 📌 Multi-Word Phrase Extraction (NEW)
# ===========================
def extract_phrases(text, n=3):
    """Extract n-gram phrases from text."""
    words = nltk.word_tokenize(text)
    phrases = [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]
    return words + phrases  # Combine unigrams with phrases


def preprocess_text(text):
    """Preprocess text by tokenizing, stemming, and generating phrases."""
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    words = nltk.word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words]

    # Generate bigrams and trigrams
    bigrams = extract_phrases(text, 2)
    trigrams = extract_phrases(text, 3)

    return ' '.join(stemmed_words + bigrams + trigrams)


# ===========================
# 📌 Fetch Documents (No changes here)
# ===========================
def fetch_documents(url):
    """Fetch articles from a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        content = response.text
        soup = BeautifulSoup(content, 'html.parser')
        articles = soup.find_all('article')

        article_dict = {}
        for article in articles:
            name = article.get("name", f"Unnamed_{len(article_dict) + 1}")
            content = article.get_text(separator=" ", strip=True)
            article_dict[name] = content
        return article_dict
    except requests.RequestException as e:
        print(f"Error fetching documents: {e}")
        return {}


# ===========================
# 📌 Load Dataset (No changes here)
# ===========================
url = "https://raw.githubusercontent.com/martinabgn/HelsinkiLib/week2/enwiki-20181001-corpus.1000-articles.txt"
article_dict = fetch_documents(url)

if not article_dict:
    print("No documents to process. Exiting.")
    exit()

documents, names = list(article_dict.values()), list(article_dict.keys())

# ===========================
# 📌 Initialize Search Models (With Multi-Word Support)
# ===========================
# Boolean Search Model
cv = CountVectorizer(lowercase=True, binary=True, token_pattern=r"(?u)\b\w+\b")
sparse_matrix = cv.fit_transform(documents)

# TF-IDF Search Model (Now supports phrases)
tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 3))
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Semantic Search Model (No changes)
semantic_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
doc_embeddings = semantic_model.encode(documents, convert_to_tensor=False)


# ===========================
# 📌 Boolean Search Updates (Phrase Handling)
# ===========================
def rewrite_query(query):
    """Rewrite the query for Boolean search, supporting multi-word phrases."""
    query = query.replace('"', '')  # Remove quotes for processing
    return " ".join(rewrite_token(t) for t in query.split())


def search_query(query, top_n=5, truncate_m=200):
    """Perform Boolean search with phrase support."""
    try:
        if '"' in query:
            query = query.replace('"', '')  # Treat phrases as regular words

        rewritten_query = rewrite_query(query)
        hits_matrix = eval(rewritten_query)

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
            print(f"Matching doc #{i + 1} (Index {doc_idx}): {truncated_content}...\n")

    except Exception as e:
        print(f"Error processing query '{query}': {e}")


# ===========================
# 📌 TF-IDF Search Updates (Now Supports Multi-Word Queries)
# ===========================
def search_tfidf(query, top_n=5, truncate_m=200):
    """Perform TF-IDF search with multi-word phrase support."""
    try:
        processed_query = preprocess_text(query)
        query_vector = tfidf_vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        ranked_indices = similarities.argsort()[::-1]
        ranked_indices = [idx for idx in ranked_indices if similarities[idx] > 0]

        total_hits = len(ranked_indices)
        if total_hits == 0:
            print("No matching documents found.")
            return

        print(f"🔍 Query: {query}")
        print(f"📄 Total matching documents: {total_hits}")
        print(f"📌 Showing top {min(top_n, total_hits)} documents:\n")

        for i, doc_idx in enumerate(ranked_indices[:top_n]):
            truncated_content = " ".join(documents[doc_idx].split()[:truncate_m])
            print(f"🔹 Matching doc #{i + 1} (Index {doc_idx}, Score: {similarities[doc_idx]:.4f}): {names[doc_idx]}")
            print(f"📄 {truncated_content}...\n")

    except Exception as e:
        print(f"⚠️ Error processing query '{query}': {e}")


# ===========================
# 📌 Semantic Search (No changes, benefits from phrase handling)
# ===========================
def search_semantic(query, top_n=5, truncate_m=200):
    """Perform semantic search using BERT embeddings."""
    try:
        query_embedding = semantic_model.encode([query], convert_to_tensor=False)
        similarities = cosine_similarity(query_embedding, doc_embeddings).flatten()
        ranked_indices = similarities.argsort()[::-1]
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
            print(f"🔹 Matching doc #{i + 1} (Index {doc_idx}, Score: {similarities[doc_idx]:.4f}): {names[doc_idx]}")
            print(f"📄 {truncated_content}...\n")

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