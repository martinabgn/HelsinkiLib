import requests
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# 从 URL 加载文档
def fetch_documents(url):
    """
    从远程 URL 下载文件内容。
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # 检查请求是否成功
        content = response.text
        # 按 </article> 分割为文档
        documents = content.split("</article>")
        return documents
    except requests.RequestException as e:
        print(f"Error fetching documents from URL: {e}")
        return []

# 获取文档
url = "https://raw.githubusercontent.com/martinabgn/HelsinkiLib/week2/enwiki-20181001-corpus.1000-articles.txt"
documents = fetch_documents(url)

# 检查文档是否成功加载
if not documents:
    print("No documents to process. Exiting.")
    exit()

# 输出文档信息
print(f"Extracted {len(documents)} documents.")

# 2. 创建术语-文档矩阵
cv = CountVectorizer(lowercase=True, binary=True, token_pattern=r"(?u)\b\w+\b")
sparse_matrix = cv.fit_transform(documents)

# 检查 CountVectorizer 是否成功构建
if len(cv.get_feature_names_out()) == 0:
    print("No terms were indexed by CountVectorizer. Check your documents.")
    exit()
print("Terms indexed by CountVectorizer:", cv.get_feature_names_out())
print("Shape of sparse_matrix:", sparse_matrix.shape)

terms = cv.get_feature_names_out()
t2i = cv.vocabulary_

# 转置矩阵
dense_matrix = sparse_matrix.todense()
td_matrix = dense_matrix.T

# 布尔查询动态解析
d = {"and": "&", "AND": "&",
     "or": "|", "OR": "|",
     "not": "1 -", "NOT": "1 -",
     "(": "(", ")": ")"}

def rewrite_token(t):
    # 检查是否为布尔操作符
    if t in d:
        return d[t]
    # 检查查询词是否存在于词汇表
    if t in t2i:
        return 'td_matrix[t2i["{:s}"]]'.format(t)
    # 如果词不在词汇表中，返回全 0 向量
    return 'np.zeros(td_matrix.shape[1], dtype=int)'

def rewrite_query(query):
    return " ".join(rewrite_token(t) for t in query.split())

# 搜索功能，支持分页和内容截断
def search_query(query, top_n=5, truncate_m=50):
    """
    搜索并显示查询结果，支持分页和内容截断。
    :param query: 用户输入的查询
    :param top_n: 显示的最多文档数量
    :param truncate_m: 每篇文档显示的最大单词数
    """
    try:
        hits_matrix = eval(rewrite_query(query))  # 执行查询
        hits_list = list(hits_matrix.nonzero()[1])  # 获取匹配文档索引
        total_hits = len(hits_list)

        if total_hits == 0:
            print("No matching documents found.")
            return

        print(f"Query: {query}")
        print(f"Total matching documents: {total_hits}")
        print(f"Showing top {min(top_n, total_hits)} documents:\n")

        # 遍历匹配文档，分页显示并截断内容
        for i, doc_idx in enumerate(hits_list[:top_n]):
            truncated_content = " ".join(documents[doc_idx].split()[:truncate_m])  # 截断文档内容
            print(f"Matching doc #{i+1} (Index {doc_idx}): {truncated_content}...\n")

        if total_hits > top_n:
            print(f"Only showing the top {top_n} results. Refine your query to see more.")
    except Exception as e:
        print(f"Error processing query '{query}': {e}")

        # 主程序循环
while True:
    user_query = input("Enter your query (type 'quit' to exit): ")
    if user_query.lower() == "quit":
        break
    search_query(user_query, top_n=5, truncate_m=50)