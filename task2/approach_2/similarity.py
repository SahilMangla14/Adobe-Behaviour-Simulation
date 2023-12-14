
import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
import joblib
import nltk
import numpy as np
from nltk import pos_tag, word_tokenize, ne_chunk
from nltk.corpus import stopwords
nltk.download('punkt')  # for tokenization
nltk.download('averaged_perceptron_tagger')  # for POS-tagging
nltk.download('maxent_ne_chunker')  # for NER
nltk.download('words')  # for NER
nltk.download('stopwords')  # for NER

from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

class CosineSimilarity:
    def __init__(self, texts):
        self.texts = texts
        pass
    # COSINE SIMILARITY FUNCTIONS---------------------------------------------------------------------------------------------

    def get_cosine_similarity(self,word1, word2): # using bert embeddings
        # Use BERT to find embeddings for each word
        word1_embedding = model(tokenizer.encode(word1, return_tensors='pt'))[0][:,0,:].detach().numpy()
        word2_embedding = model(tokenizer.encode(word2, return_tensors='pt'))[0][:,0,:].detach().numpy()

        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(word1_embedding, word2_embedding)

        # Return the cosine similarity score
        return similarity_matrix[0, 0]
    
    def get_top_k_similar_words(self,list1, list2, k=1):
        # Initialize a dictionary to store similarity scores
        similarity_scores = {}

        # Iterate through each pair of words in the two lists
        for word1 in list1:
            for word2 in list2:
                # Calculate cosine similarity
                similarity = self.get_cosine_similarity(word1, word2)
                
                # Store the similarity score in the dictionary
                key = (word1, word2)
                similarity_scores[key] = similarity

        # Sort the similarity scores in descending order
        sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

        # Get the top-k similar pairs
        top_k_pairs = sorted_scores[:k]

        return top_k_pairs
    
    def example_run(self):
        kmeans = self.train_and_save_cluster()

        keywords = self.get_keywords_from_media(kmeans, media_desc='A group of people are standing in a room')
        print("Keywords for media description 'A group of people are standing in a room':", keywords)
        
        print("NOW GETTING SIMILARITY SCORES, PLEASE WAIT...")
        # Example usage with two lists of keywords
        keywords_list1 = ["women", "blue shirt", "vests", "working", "working in factory", "factory"]
        keywords_list2 = ["empowerment", "blue shirts", "workforce", "factory", "film production"]

        # Specify the value of k (top-k similar pairs)
        top_k = 6

        # Get the top-k similar pairs
        top_k_pairs = self.get_top_k_similar_words(keywords_list1, keywords_list2, k=top_k)

        # Print the top-k similar pairs and their scores
        for pair, score in top_k_pairs:
            word1, word2 = pair
            print(f"Similarity between '{word1}' and '{word2}': {score}")
