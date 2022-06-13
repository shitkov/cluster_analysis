import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances


class Clusterizer:
    
    
    def __init__(self, model_path):
        self.model = AutoModel.from_pretrained(model_path) 
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    
    def predict(self, texts):
        embeddings = self._get_embeddings(texts)
        labels, centers = self._get_labels_centers(embeddings)
        top_texts = self._get_top(texts, embeddings, labels, centers)
        return top_texts
    
    
    def _get_embeddings(self, texts):
        embeddings = []
        print('Get texts embeddings:')
        for text in tqdm(texts):
            encoded_input = self.tokenizer(text, padding=True, truncation=True, max_length=64, return_tensors='pt')
            with torch.no_grad():
                output = self.model(**encoded_input)
            embeddings.append((output.pooler_output)[0].numpy())
        return np.asarray(embeddings)


    def _get_labels_centers(self, embeddings, k_min=1):
        clusters = [x for x in range(2, len(embeddings))]
        metrics = []
        print('Calculating the optimal number of clusters:')
        for i in tqdm(clusters):
            metrics.append((KMeans(n_clusters=i, random_state=42).fit(embeddings)).inertia_)
        k = self._elbow(k_min, clusters, metrics)
        k_means = KMeans(n_clusters=k, random_state=42).fit(embeddings)
        labels = k_means.labels_
        centers = k_means.cluster_centers_
        return labels, centers


    def _get_top(self, texts, embeddings, labels, centers):
        data = pd.DataFrame()
        data['text'] = texts
        data['label'] = labels
        data['embedding'] = list(embeddings)
        top_texts_list = []
        qnt_dict = {}

        for label in range(len(centers)):
            qnt_dict[label] = len(data[data['label'] == label])
            
        for i in range(len(centers)):
            cluster = data[data['label'] == i]
            embeddings = list(cluster['embedding'])
            texts = list(cluster['text'])
            distances = [euclidean_distances(centers[0].reshape(1, -1), e.reshape(1, -1))[0][0] for e in embeddings]
            scores = list(zip(texts, distances))
            top_1 = sorted(scores, key=lambda x: x[1])[:1]
            top_texts = list(zip(*top_1))[0]
            top_texts_list.append((top_texts[0], qnt_dict[i]))
        
        top_texts_list = sorted(top_texts_list, key=lambda x: x[1], reverse=True)
        texts_list = list(zip(*top_texts_list))[0]
        return list(texts_list)


    def _elbow(self, k_min, clusters, metrics):
        score = []

        for i in range(k_min, clusters[-3]):
            y1 = np.array(metrics)[:i + 1]
            y2 = np.array(metrics)[i:]

            df1 = pd.DataFrame({'x': clusters[:i + 1], 'y': y1})
            df2 = pd.DataFrame({'x': clusters[i:], 'y': y2})

            reg1 = LinearRegression().fit(np.asarray(df1.x).reshape(-1, 1), df1.y)
            reg2 = LinearRegression().fit(np.asarray(df2.x).reshape(-1, 1), df2.y)

            y1_pred = reg1.predict(np.asarray(df1.x).reshape(-1, 1))
            y2_pred = reg2.predict(np.asarray(df2.x).reshape(-1, 1))    

            score.append(mean_squared_error(y1, y1_pred) + mean_squared_error(y2, y2_pred))

        return np.argmin(score) + k_min