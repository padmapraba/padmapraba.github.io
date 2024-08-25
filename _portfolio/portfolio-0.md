---
title: "Book Recommender"
excerpt: "Recommendations using word embeddings<br/>"
collection: portfolio
---


## Building a Book Recommendation System with FAISS and Sentence Embeddings

In this project, I developed a book recommendation system by leveraging the power of embeddings and FAISS (Facebook AI Similarity Search). The goal was to identify and recommend books similar to a given query or a specific book title. Below, I walk through the key steps and concepts involved in creating this system.

## Web Scraping Goodreads Data

To start, I used Python's `requests` library and `BeautifulSoup` to scrape book data from Goodreads. I extracted various attributes such as book titles and URLs for further processing. After scraping, the data was cleaned and organized into a structured format using `pandas`. This structured data formed the basis for our recommendation system.

## Embedding Book Descriptions

Next, I needed to convert book descriptions into numerical vectors, known as embeddings, that could be used for similarity comparison. For this, I used the `SentenceTransformer` library, specifically the `all-MiniLM-L6-v2` model, which generates high-quality sentence embeddings.

Embeddings are a way of representing text data in a continuous vector space, where similar texts have similar vector representations. By embedding book descriptions, I could capture semantic similarities between books based on their content.

## Using FAISS for Efficient Similarity Search

With embeddings ready, the next step was to perform similarity searches. This is where FAISS comes into play. FAISS is a library developed by Facebook AI that enables fast and efficient similarity search, even on large datasets.

I used FAISS to create an index of the book embeddings. This index allows us to quickly find books that are most similar to a given query or another book's description.

### How FAISS Works

FAISS operates by creating an index structure that allows efficient querying. The primary method used here is L2 distance (Euclidean distance), which measures how far apart two vectors (in our case, book embeddings) are. When a query is made, FAISS returns the closest vectors in the index, which correspond to the most similar books.

## Querying Similar Books

To find similar books, I first convert the query (e.g., a book description or a short phrase) into an embedding using the same SentenceTransformer model. I then search this query embedding against our FAISS index to retrieve a list of books that are most similar.

For instance, if you input a query like "pirates and sailors fight," the system generates an embedding for this phrase and searches the FAISS index for books with descriptions that are closest in vector space to the query embedding. The result is a list of books that match the theme or content of the query.

## Conclusion

This project demonstrates how to build a recommendation system using modern NLP techniques and tools. By combining sentence embeddings with FAISS, I created a scalable and efficient system capable of providing high-quality book recommendations. Whether for personal use or integration into a larger platform, this approach can be adapted to various domains beyond books, making it a versatile solution for content-based recommendation systems.
