import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
import joblib
import nltk
import numpy as np
from nltk import pos_tag, word_tokenize, ne_chunk
nltk.download('punkt')  # for tokenization
nltk.download('averaged_perceptron_tagger')  # for POS-tagging
nltk.download('maxent_ne_chunker')  # for NER
nltk.download('words')  # for NER
nltk.download('stopwords')  # for NER

from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

class BERTClustering:
    def __init__(self, number_of_clusters=100):
        self.number_of_clusters = number_of_clusters
        pass
    # BERT + NER FUNCTIONS-----------------------------------------------------------------------------------------------------------
    def train_and_save_cluster(self, texts):
        # Step 2: Apply KMeans clustering to obtain cluster assignments
        try:
            embeddings_array = np.load('data/media_bert_embeddings_array.npy') # load if already saved
        except Exception as e:
            embeddings_list = [model(tokenizer.encode(text, return_tensors='pt'))[0][:, 0, :].detach().numpy() for text in texts]
            embeddings_array = torch.cat([torch.from_numpy(embeddings) for embeddings in embeddings_list]).numpy()

        num_clusters = self.number_of_clusters  # Adjust based on your optimal number of clusters -> maybe 10
        kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
        bert_clusters = kmeans.fit_predict(embeddings_array)

        joblib.dump(kmeans, 'saved/kmeans_bert_model.pkl')  # save model
        np.save('saved/bert_clusters.npy', bert_clusters)  # save clusters

        return bert_clusters
    
    def get_cluster_keywords(self,texts, clusters):

        results = list(zip(texts, clusters))
        cluster_keywords = [[] for i in range(self.number_of_clusters)]

        for text, cluster_id in results:

            # Tokenize and POS-tag the text
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)

            # Apply NER using NLTK
            tree = ne_chunk(pos_tags)
            named_entities = [chunk.label() for chunk in tree if hasattr(chunk, 'label')]
            
            # Extract action verbs
            action_verbs = [word for (word, pos) in pos_tags if pos.startswith("VB")]

            named_entities = [entity.lower() for entity in named_entities]

            keywords = named_entities + action_verbs

            for keyword in keywords:
                cluster_keywords[cluster_id].append(keyword)
                
        return cluster_keywords
    
    def get_keywords_from_media(self,kmeans, media_desc:str):
        embedding = model(tokenizer.encode(media_desc, return_tensors='pt'))[0][:, 0, :].detach().numpy()
        current_cluster_id = kmeans.predict(embedding)[0]

        clusters = np.load('saved/bert_clusters.npy')
        cluster_keywords = self.get_cluster_keywords(self.texts, clusters)
        keywords = cluster_keywords[current_cluster_id]
        return keywords 
