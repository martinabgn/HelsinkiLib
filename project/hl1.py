from flask import Flask, render_template, request
import search  # Import the whole search module

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("welcome.html")

@app.route("/results")
def results():
    query = request.args.get("query", "").strip()
    if not query:
        return render_template("results.html", query=query, results=[])

    search_results = search.get_search_results(query)  # Call the function from search.py
    return render_template("results.html", query=query, results=search_results)

if __name__ == "__main__":
    app.run(debug=True)
