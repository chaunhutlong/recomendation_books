import json
import re
import pandas as pd
import torch
import uuid
import requests
from PIL import Image
import io
from clip import clip
from elasticsearch import Elasticsearch
from flask import Flask, request, jsonify
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

QUANTIZATION_BITS = 16  # Number of bits for quantization
SIMILARITY_THRESHOLD = 0.8  # Similarity threshold for visual search

# Connect to Elasticsearch
es = Elasticsearch(
    cloud_id="book-similarity"
             ":YXNpYS1lYXN0MS5nY3AuZWxhc3RpYy1jbG91ZC5jb20kZGY4OWFlYTE4MGU3NDQwY2I2ODVjYzZmMTVhMDIzMzgkY2I5NGZjOTJiYjY2NDNkZmIxMjlmNDBkYzJjOTViODQ=",
    basic_auth=("elastic", "iwQoVJ3lqWcWKe45sH6J9KjH"),
)

book_images_index = "book_images_index"
book_embeddings_index = 'book_embeddings_index'

try:
    with open("./image-embeddings-mappings.json", "r") as config_file:
        config = json.load(config_file)
        if not es.indices.exists(index=book_images_index):
            es.indices.create(index=book_images_index, settings=config["settings"], mappings=config["mappings"])
            print("Created index")
        else:
            print("Index already exists")

    with open("./book-embeddings-mappings.json", "r") as config_file:
        config = json.load(config_file)
        if not es.indices.exists(index=book_embeddings_index):
            es.indices.create(index=book_embeddings_index, settings=config["settings"], mappings=config["mappings"])
            print("Created index")
        else:
            print("Index already exists")
except Exception as e:
    print(e)

model, preprocess = clip.load("ViT-B/32")


@app.route('/delete-duplicate-images', methods=['DELETE'])
def delete_duplicate_images():
    # Define the field to check for duplicates (in this case, 'image_id')
    duplicate_field = 'image_id'

    # Define the query to find duplicates
    query = {
        "aggs": {
            "duplicateCount": {
                "terms": {
                    "field": duplicate_field,
                    "min_doc_count": 2,
                    "size": 10000
                }
            }
        },
        "size": 0
    }

    # Search for duplicates using the query
    response = es.search(index=book_images_index, body=query)

    # Iterate over the duplicate groups
    for group in response['aggregations']['duplicateCount']['buckets']:
        # Get the duplicate documents for each group
        duplicate_documents = es.search(
            index=book_images_index,
            body={
                "query": {
                    "match": {
                        duplicate_field: group['key']
                    }
                }
            }
        )['hits']['hits']

        # Delete all duplicate documents except for the first one
        for duplicate_document in duplicate_documents[1:]:
            es.delete(index=book_images_index, id=duplicate_document['_id'])

    return "Duplicate images deleted successfully"


@app.route('/book-images', methods=['POST'])
def receive_book_images():
    book_image_data_list = request.get_json()
    quantized_embedding_list = []

    # Iterate over the list of book image data
    for book_image_data in book_image_data_list:
        # Extract the necessary data from each book image data dictionary
        image_id = book_image_data['id']
        embedding_id = book_image_data.get('embeddingId', None)
        book_image_url = book_image_data['url']

        # Retrieve the book image from the URL
        book_image_file = io.BytesIO(requests.get(book_image_url).content)

        # Read the image file and extract the embedding
        embedding = extract_embedding(book_image_file)

        # Quantize the embedding
        embedding = quantize_embedding(embedding)

        # Convert the embedding to a list and flatten the list
        embedding = embedding.tolist()
        embedding = [item for sublist in embedding for item in sublist]

        # Generate embedding_id for the book image data if the embedding_id is None
        if embedding_id is None:
            embedding_id = generate_embedding_id()

        quantized_embedding_list.append(embedding_id)

        # Store the book image embedding in Elasticsearch
        store_book_image(image_id, book_image_url, embedding_id, embedding)

    return jsonify(quantized_embedding_list)


@app.route('/book-features', methods=['POST'])
def embedding_books():
    # get book data and review data from request
    data = request.get_json()
    books = data['books']
    reviews = data['reviews']

    # Preprocess book data
    df_books = preprocess_book_data(pd.DataFrame(books))

    df_reviews = pd.DataFrame(reviews)
    # Group the reviews by book_id and calculate the count and mean rating for each book
    df_agg_reviews = df_reviews.groupby('book_id').agg({'rating': ['count', 'mean']}).reset_index()

    # Flatten the column hierarchy in df_agg_reviews
    df_agg_reviews.columns = [' '.join(col).strip() for col in df_agg_reviews.columns.values]

    # Merge the book data with the review data
    df_books = df_books.merge(df_agg_reviews, on='book_id', how='left')

    # Fill the missing ratings with 0 count and mean rating
    df_books['rating count'] = df_books['rating count'].fillna(0)
    df_books['rating mean'] = df_books['rating mean'].fillna(0)

    # Scale the rating count to a value between 0 and 1
    scaler = MinMaxScaler()
    df_books['rating count scaled'] = scaler.fit_transform(df_books[['rating count']])

    # Calculate the final rating based on the scaled count and mean rating
    df_books['final_rating'] = df_books['rating count scaled'] * df_books['rating mean']

    # Create a tf-idf matrix based on the book summaries
    tfidf_matrix = create_tfidf_matrix(df_books['title'])

    # Convert the tf-idf matrix to a dense matrix
    tfidf_matrix_dense = tfidf_matrix.toarray()

    # use PCA to reduce the dimensionality of the tf-idf matrix
    pca = PCA(n_components=512)
    tfidf_matrix_dense = pca.fit_transform(tfidf_matrix_dense)

    # Iterate over the book data
    for index, row in df_books.iterrows():
        # Store the book embedding in Elasticsearch
        store_book_embedding(row['book_id'], tfidf_matrix_dense[index], row['genre_ids'], row['final_rating'])

    return jsonify({'message': 'Book embeddings stored in Elasticsearch.'})


def preprocess_book_data(df):
    # Clean the title columns
    df['title'] = df['title'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x).lower()))
    return df


def create_tfidf_matrix(text_data):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text_data)
    return tfidf_matrix


def store_book_embedding(book_id, book_embedding, genre_ids, final_rating):
    # Store the book embedding in Elasticsearch
    es.index(
        index=book_embeddings_index,
        document={
            'book_id': book_id,
            'embedding': book_embedding,
            'genre_ids': genre_ids,
            'rating': final_rating
        }
    )


def get_user_ratings(user_id):
    # Call api to spring boot to get user ratings http://localhost:8080/api/reviews/users/{user_id}
    user_ratings = requests.get(f'http://localhost:8080/api/reviews/users/{user_id}').json()

    return user_ratings


@app.route('/recommendations/books/<int:book_id>', methods=['GET'])
def get_recommendations_for_book(book_id):
    page = request.args.get('page', default=1, type=int)
    page_size = request.args.get('limit', default=10, type=int)

    query = {
        "match": {
            "book_id": book_id
        }
    }
    book = es.search(index=book_embeddings_index, query=query)['hits']['hits'][0]['_source']

    # get all genre ids
    aggregation_query = {
        "unique_genres": {
            "terms": {
                "field": "genre_ids",
                "size": 100
            }
        }
    }

    # get all genre ids
    result = es.search(index=book_embeddings_index, aggregations=aggregation_query, size=0)

    # Extract the genre buckets from the aggregation result
    genre_buckets = result['aggregations']['unique_genres']['buckets']

    # Retrieve the genre IDs from the buckets
    all_genre_ids = [bucket['key'] for bucket in genre_buckets]

    genre_ids = book['genre_ids']
    # another genre ids is not the same as genre ids
    another_genre_ids = list(set(all_genre_ids) - set(genre_ids))

    # Retrieve the book with similar book embeddings from Elasticsearch with book id not equal to the given book id
    query = {
        "script_score": {
            "query": {
                "bool": {
                    "must_not": [
                        {"match": {"book_id": book_id}}
                    ],
                    "should": [
                        {"terms": {"genre_ids": genre_ids}},
                        {"terms": {"genre_ids": another_genre_ids}},
                    ],
                }
            },
            "script": {
                "source":
                    "((cosineSimilarity(params.query_vector, 'embedding') + 1) * params.title_weight) "
                    "+ params.rating_weight * doc['rating'].value "
                    "+ (params.genre_weight * _score)",
                "params": {
                    "query_vector": book['embedding'],
                    "genre_ids": book['genre_ids'],
                    "genre_weight": 1,
                    "rating_weight": 0.4,
                    "title_weight": 3.5
                }
            }
        }
    }

    # Calculate the starting index based on the page and page_size
    start_index = page * page_size

    # Retrieve the top similar books with pagination
    search_results = es.search(index=book_embeddings_index, query=query, size=page_size, from_=start_index)['hits'][
        'hits']

    # Return the paginated list of similar books with the book id and score
    return jsonify([{'book_id': int(book['_source']['book_id']), 'score': book['_score']} for book in search_results])


@app.route('/recommendations', methods=['GET'])
def get_recommendations_for_homepage():
    # Retrieve the books that have the most ratings
    query = {
        "script_score": {
            "query": {
                "match_all": {}
            },
            "script": {
                "source": "doc['rating'].value"
            }
        }
    }

    # Calculate the starting index based on the page and page_size

    # Retrieve the top 100 books with the most ratings
    search_results = es.search(index=book_embeddings_index, query=query, size=20)['hits'][
        'hits']

    # Return the list of books with the book id and score
    return jsonify([{'book_id': int(book['_source']['book_id']), 'score': book['_score']} for book in search_results])


def store_book_image(image_id, book_image_url, embedding_id, embedding):
    try:
        document = {
            "book_image_url": book_image_url,
            "embedding_id": embedding_id,
            "embedding": embedding
        }

        es.index(index=book_images_index, id=image_id, body=document)
    except Exception as error:
        print(error)


def generate_embedding_id():
    # Generate a unique embedding ID
    return str(uuid.uuid4())


@app.route('/visual-search', methods=['POST'])
def perform_visual_search():
    # Retrieve the uploaded image file
    image_file = request.files['image']

    # Get the pagination parameters from the request
    page = request.args.get('page', default=1, type=int)
    page_size = request.args.get('limit', default=10, type=int)

    # Read the image file and extract the embedding
    query_embedding = extract_embedding(image_file)

    # Normalize the query embedding
    query_embedding = normalize_embedding(query_embedding)

    # Quantize the query embedding
    query_embedding = quantize_embedding(query_embedding)

    # Convert the embedding to a list
    query_embedding = query_embedding.tolist()

    # flatten the list
    query_embedding = [item for sublist in query_embedding for item in sublist]

    # Search for similar images using Elasticsearch
    similar_images = search_similar_images(query_embedding, page, page_size)

    # Return the ranked book images as the API response
    response = {
        "results": [],
        "page": page,
        "limit": page_size,
        "total_results": similar_images['total']['value'],
    }
    for image in similar_images['hits']:
        response['results'].append({
            "image_id": image['_id'],
            "book_image_url": image['_source']['book_image_url'],
            "embedding_id": image['_source']['embedding_id'],
            "score": image['_score']
        })

    return jsonify(response)


def search_similar_images(query_embedding, page=1, results_per_page=10):
    try:
        # Calculate the starting index for pagination
        start_index = (page - 1) * results_per_page

        # Build the Elasticsearch k-NN query for similar images with pagination
        query = {
            "from": start_index,
            "size": results_per_page,
            "_source": {
                "includes": ["book_image_url", "embedding_id"]
            },
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": results_per_page * 5,
                "num_candidates": 10000
            }
        }

        # Search for similar images using Elasticsearch
        response = es.search(index=book_images_index, knn=query['knn'],
                             _source=query['_source'], size=query["size"], from_=query['from'])

        # Return the list of similar images
        return response['hits']
    except Exception as error:
        print(error)
        return []


def extract_embedding(image_file):
    # Read the image file using PIL
    img = Image.open(image_file)

    # Preprocess the image
    preprocessed_image = preprocess(img)

    # Convert the preprocessed image to a tensor
    input_tensor = preprocessed_image.unsqueeze(0)

    # Calculate the embedding using the CLIP model
    with torch.no_grad():
        embedding = model.encode_image(input_tensor)

    return embedding


def normalize_embedding(embedding):
    # Normalize the embedding
    normalized_embedding = embedding / torch.norm(embedding)

    return normalized_embedding


def quantize_embedding(embedding):
    # Scale the embedding to the desired range
    scaled_embedding = embedding * (2 ** QUANTIZATION_BITS - 1)

    # Quantize the scaled embedding
    quantized_embedding = scaled_embedding.round().int()

    return quantized_embedding


if __name__ == '__main__':
    app.run()
