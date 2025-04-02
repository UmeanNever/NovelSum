import json
import numpy as np
import os
import faiss
import argparse
from utils import load_data, load_dir_data

def kmeans_entropy(n_clusters: int, ref_data, query_data) -> float:
    kmeans = faiss.Kmeans(ref_data.shape[1], n_clusters, niter=20, verbose=False, seed=42)
    kmeans.train(ref_data)

    _, filtered_labels = kmeans.index.search(query_data, 1)
    filtered_labels = filtered_labels.reshape(-1)

    _, counts = np.unique(filtered_labels, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy)

def main():
    parser = argparse.ArgumentParser(description="Calculate entropy using KMeans clustering.")
    parser.add_argument("--ref_data_dir", type=str, required=True, help="Directory containing reference data JSON files.")
    parser.add_argument("--query_data_path", type=str, required=True, help="Path to the query data JSON file.")
    parser.add_argument("--n_clusters", type=int, default=100, help="Number of clusters for KMeans.")
    
    args = parser.parse_args()

    ref_data = load_dir_data(args.ref_data_dir)
    query_data = load_data(args.query_data_path)

    entropy = kmeans_entropy(args.n_clusters, ref_data, query_data)

    print(f"The partition entropy for this dataset is: {entropy}")
    

if __name__ == "__main__":
    main()