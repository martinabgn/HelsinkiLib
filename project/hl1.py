from flask import Flask, render_template, request
import search

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("welcome.html")

@app.route("/results")
def results():
    query = request.args.get("query", "").strip()
    category = request.args.get("category", "all")  # Default to "all" if missing
    mode = request.args.get("mode", "boolean")  # Default to "boolean"

    if not query:
        return render_template("results.html", query=query, results=[])

    print(f"🔎 Search Query: {query}, Category: {category}, Mode: {mode}")  # Debugging log

    search_results = search.get_search_results(query, category, mode)
    return render_template("results.html", query=query, results=search_results)

if __name__ == "__main__":
    app.run(debug=True)
