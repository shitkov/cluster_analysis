[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shitkov/cluster_analysis/blob/master/cluster_analysis_news.ipynb)
# Cluster analysis based on RIA news headlines
1. Create embeddings: LaBSE/USE/LASER.
2. Create optimum clusters number with k-means and linear regression.
3. Produse optimum clusters.
4. Find nearest texts to center of each cluster.
5. Summarize texts for each cluster.
---
1. It is very important what data is being analyzed. If you take tweets, then without preprocessing everything will be pretty sad.
2. Final summarisation does not work well. Tries to compose one from different news, instead of highlighting the essence of the collection. Instead of a sumarizer, it is better to use manual marking of the resulting categories with subsequent training of the classifier.
