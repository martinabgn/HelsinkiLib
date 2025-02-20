from flask import Flask, render_template, request
import sqlite3

app = Flask(__name__)

# Function to connect to SQLite
def get_db_connection():
    conn = sqlite3.connect("library.db")
    conn.row_factory = sqlite3.Row  # Allows access by column name
    return conn

@app.route("/", methods=["GET", "POST"])
def search():
    query = request.form.get("query", "").lower()
    results = []

    if query:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Search query in all text-based columns
        cursor.execute("""
            SELECT * FROM library_data
            WHERE 
                title LIKE ? OR 
                author LIKE ? OR 
                location LIKE ? OR 
                category LIKE ?
        """, (f"%{query}%", f"%{query}%", f"%{query}%", f"%{query}%"))

        results = cursor.fetchall()
        conn.close()

    return render_template("index.html", query=query, results=results)

if __name__ == "__main__":
    app.run(debug=True)
