from flask import Flask, render_template, request, redirect, url_for
import pandas as pd

app = Flask(__name__)

# Load the books from the pickle file
books_df = pd.read_pickle('data/book_data.pkl')

# Select and rename columns for easier access in templates
books_df = books_df[['image_url', 'book_title', 'book_authors', 'book_rating',
                     'book_rating_count', 'book_review_count', 'book_desc', 'genres', 'most_similar']]
books_df["id"] = books_df.index

# Define number of books per page
BOOKS_PER_PAGE = 20

# Function to fetch books for a specific page
def get_books_for_page(page_number):
    start_idx = (page_number - 1) * BOOKS_PER_PAGE
    end_idx = start_idx + BOOKS_PER_PAGE
    return books_df.iloc[start_idx:end_idx].to_dict(orient='records')

# Sample recommendation logic (replace with your own logic)
def get_recommendations(selected_book_id):
    # Placeholder recommendation logic: returning the next two books in the list
    recommendations = [book for book in books_df.to_dict(orient='records') if book['id'] != selected_book_id]
    return recommendations[:2]

@app.route('/')
@app.route('/<int:page>')
def book_list(page=1):
    books = get_books_for_page(page)
    return render_template('book_list.html', books=books, page=page)

@app.route('/recommendations', methods=['POST'])
def recommendations():
    selected_book_id = int(request.form['book_id'])
    print(selected_book_id)
    selected_book = next((book for book in books_df.to_dict(orient='records') if book['id'] == selected_book_id), None)
    if not selected_book:
        return redirect(url_for('book_list'))

    recommended_books = get_recommendations(selected_book_id)
    return render_template('recommendations.html', selected_book=selected_book, recommended_books=recommended_books)

if __name__ == '__main__':
    app.run(debug=True)
