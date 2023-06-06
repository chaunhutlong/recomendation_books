import json
import re
import nltk
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


nltk.download('stopwords')

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


@app.route('/embedding-books', methods=['POST'])
def embedding_books():
    # get book data and review data from request
    data = request.get_json()
    books = data['books']

    # Preprocess book data
    df_books = preprocess_book_data(pd.DataFrame(books))

    # Create TF-IDF matrix
    tfidf_matrix = create_tfidf_matrix(df_books['title'])

    # Convert TF-IDF matrix to a dense vector
    tfidf_matrix_dense = tfidf_matrix.toarray()

    # Use PCA to reduce the dimensionality of the TF-IDF matrix
    pca = PCA(n_components=512)
    tfidf_matrix_dense = pca.fit_transform(tfidf_matrix_dense)

    # Store book embeddings in Elasticsearch with the book id as the document id and genre ids as a list of integers
    for i, book in enumerate(books):
        genres = [genre['id'] for genre in book['genres']] if 'genres' in book else []
        store_book_embedding(book['id'], tfidf_matrix_dense[i].tolist(), genres)

    return jsonify({'message': 'Book embeddings stored in Elasticsearch.'})


def preprocess_book_data(df):
    # Clean the title columns
    df['title'] = df['title'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x).lower()))
    return df


def create_tfidf_matrix(text_data):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text_data)
    return tfidf_matrix


def store_book_embedding(book_id, book_embedding, genre_ids):
    # Store the book embedding in Elasticsearch
    es.index(
        index=book_embeddings_index,
        document={
            'book_id': book_id,
            'embedding': book_embedding,
            'genre_ids': genre_ids
        }
    )


def get_user_ratings(user_id):
    # Call api to spring boot to get user ratings http://localhost:8080/api/reviews/users/{user_id}
    user_ratings = requests.get(f'http://localhost:8080/api/reviews/users/{user_id}').json()

    return user_ratings


@app.route('/recommendations/books/<int:book_id>', methods=['GET'])
def get_recommendations_for_book(book_id):
    # Retrieve the book with the given book id from Elasticsearch
    query = {
        "match": {
            "book_id": book_id
         }
    }

    book = es.search(index=book_embeddings_index, query=query)['hits']['hits'][0]['_source']

    # Retrieve the book with similar book embeddings from Elasticsearch with book id not equal to the given book id
    query = {
        "script_score": {
            "query": {
                # get book with similar genre ids with the given book and exclude the given book
                "bool": {
                    "must": [
                        {"terms": {"genre_ids": book['genre_ids']}}
                    ],
                    "must_not": [
                        {"match": {"book_id": book_id}}
                    ]
                }
            },
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                "params": {
                    "query_vector": book['embedding']
                }
            }
        }
    }

    # Retrieve the top 100 similar books
    search_results = es.search(index=book_embeddings_index, query=query, size=100)['hits']['hits']

    # Return the list of similar books with the book id
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

    # Get the pagination parameters from the request
    page = request.args.get('page', default=1, type=int)
    results_per_page = request.args.get('results_per_page', default=10, type=int)

    # Search for similar images using Elasticsearch
    similar_images = search_similar_images(query_embedding, page, results_per_page)

    # Return the ranked book images as the API response
    response = {
        "page": page,
        "results_per_page": results_per_page,
        "total_results": similar_images['total']['value'],
        "results": []
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
