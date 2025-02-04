import requests
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Load document from URL
def fetch_documents(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        content = response.text
        # Split the content into documents by </article>
        documents = content.split("</article>")
        return documents
    except requests.RequestException as e:
        print(f"Error fetching documents from URL: {e}")
        return []

# Retrieve documents
url = "https://raw.githubusercontent.com/martinabgn/HelsinkiLib/week2/enwiki-20181001-corpus.1000-articles.txt"
documents = fetch_documents(url)

# Verify if documents were loaded successfully
if not documents:
    print("No documents to process. Exiting.")
    exit()

# Display document information
print(f"Extracted {len(documents)} documents.")

# Create a term-document matrix
cv = CountVectorizer(lowercase=True, binary=True, token_pattern=r"(?u)\b\w+\b") 
sparse_matrix = cv.fit_transform(documents)

# Check if CountVectorizer successfully built the matrix
if len(cv.get_feature_names_out()) == 0:
    print("No terms were indexed by CountVectorizer. Check your documents.")
    exit()
print("Terms indexed by CountVectorizer:", cv.get_feature_names_out())
print("Shape of sparse_matrix:", sparse_matrix.shape)

terms = cv.get_feature_names_out()
t2i = cv.vocabulary_

# Transpose the matrix
dense_matrix = sparse_matrix.todense()
td_matrix = dense_matrix.T

d = {"and": "&", "AND": "&",
     "or": "|", "OR": "|",
     "not": "1 -", "NOT": "1 -",
     "(": "(", ")": ")"}

def rewrite_token(t):
    # Check if the token is a Boolean operator
    if t in d:
        return d[t]
    # Check if the query word exists in the vocabulary
    if t in t2i:
        return 'td_matrix[t2i["{:s}"]]'.format(t)
    # If the word is not in the vocabulary, return a vector of all zeros
    return 'np.zeros(td_matrix.shape[1], dtype=int)'

def rewrite_query(query):
    return " ".join(rewrite_token(t) for t in query.split())

# Search function with pagination and content truncation
def search_query(query, top_n=5, truncate_m=50):
    """
    Search and display search results, support pagination and content truncation.
    :param query: query entered by the user
    :param top_n: maximum number of documents to display
    :param truncate_m: maximum number of words to display per document
    """
    try:
        hits_matrix = eval(rewrite_query(query))  # Execute the query
        hits_list = list(hits_matrix.nonzero()[1])  # Get the indices of matching documents
        total_hits = len(hits_list)

        if total_hits == 0:
            print("No matching documents found.")
            return

        print(f"Query: {query}")
        print(f"Total matching documents: {total_hits}")
        print(f"Showing top {min(top_n, total_hits)} documents:\n")

        # Iterate through matching documents, display results with pagination and truncation
        for i, doc_idx in enumerate(hits_list[:top_n]):
            truncated_content = " ".join(documents[doc_idx].split()[:truncate_m])  # Truncate document content
            print(f"Matching doc #{i+1} (Index {doc_idx}): {truncated_content}...\n")

        if total_hits > top_n:
            print(f"Only showing the top {top_n} results. Refine your query to see more.")
    except Exception as e:
        print(f"Error processing query '{query}': {e}")


# Main program loop
# while True: Part of User Input
# Example queries
search_query("example AND nothing", top_n=5, truncate_m=200)