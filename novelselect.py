import os
import json
import random
import argparse
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from tqdm import tqdm
from utils import *

def distance_weighted(matrix: torch.Tensor, power: float = 1.0) -> torch.Tensor:
    sorted_values, sorted_indices = torch.sort(matrix, dim=1)
    batch_size, seq_len = matrix.shape
    
    steps = torch.arange(1, seq_len + 1, device=matrix.device, dtype=matrix.dtype)
    weights = 1 / torch.pow(steps, power)
    weights = weights.expand(batch_size, -1)
    
    weighted = torch.zeros_like(matrix)
    weights = weights.to(weighted.dtype)
    
    weighted.scatter_(1, sorted_indices, weights)
    return matrix * weighted

def bi_density_weighted_distance(matrix: torch.Tensor, 
                               selected_density: torch.Tensor, 
                               unselected_density: torch.Tensor = None, 
                               distance_power: float = 1.0) -> torch.Tensor:
    # the density for x_{i, j} = unselected_density[i] + selected_density[j]
    density_matrix = unselected_density.unsqueeze(1) + selected_density.unsqueeze(0)
    dense_weighted = matrix * density_matrix
    novelty_weighted = distance_weighted(dense_weighted, power=distance_power)
    
    return novelty_weighted

def save_results(selected_data, indices, output_text_dir: str, k: int, density_power: float, distance_power: float):
    os.makedirs(output_text_dir, exist_ok=True)
    filename = f"novelselect_{k}_dense_{density_power}_dist_{distance_power}"
    with open(os.path.join(output_text_dir, f"{filename}.json"), 'w') as f:
        json.dump(selected_data, f, indent=2)
    with open(os.path.join(output_text_dir, f"{filename}_indices.json"), 'w') as f:
        json.dump(indices, f, indent=2)

def novelselect(embeddings, k: int, gpu_id: int = 0, neighbors: int = 10,
                  density_power: float = 0.5, distance_power: float = 1.0, 
                  batch_size: int = 10000, seed: int = None):
    set_seed(seed)
    device = set_device(gpu_id)
    n_samples = embeddings.shape[0]
    
    with torch.cuda.amp.autocast():
        embeddings_tensor = torch.tensor(embeddings, device=device)
        embeddings_tensor = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1) # normalize
        embeddings_tensor = embeddings_tensor.half() # for memory efficiency
        
        faiss_index = FaissIndex(embeddings, gpu_id)
        density_map = torch.tensor(
            faiss_index.local_density(
                embeddings, 
                n_neighbors=neighbors,
                power=density_power
            ), 
            device=device,
            dtype=torch.float16
        )
    
    centers = [np.random.randint(0, n_samples)] # first point is random
    selected_embeddings = embeddings_tensor[centers]
    
    with torch.cuda.device(device):
        for i in tqdm(range(1, k), desc="Selecting centers"):
            with torch.cuda.amp.autocast():
                # compute distances in batches
                cos_distances_list = []
                for start_idx in range(0, n_samples, batch_size):
                    end_idx = min(start_idx + batch_size, n_samples)
                    batch_emb = embeddings_tensor[start_idx:end_idx] # batch of embeddings
                    
                    sims = torch.mm(batch_emb, selected_embeddings.t())
                    cos_dist = 1 - sims # cosine distance
                    sel_dense = density_map[centers]
                    un_sel_dense = density_map[start_idx:end_idx] # density of current batch
                    
                    weighted = bi_density_weighted_distance(
                        cos_dist, sel_dense, un_sel_dense, distance_power
                    )
                    mean_dist = weighted.mean(dim=1)
                    cos_distances_list.append(mean_dist)
                
                distances = torch.cat(cos_distances_list, dim=0) # concatenate all batches
                max_idx = distances.argmax()
                centers.append(max_idx.item())
                selected_embeddings = embeddings_tensor[centers]
            
            if i % 10 == 0:
                torch.cuda.empty_cache()
    
    return centers

def main():
    parser = argparse.ArgumentParser(
        description="Diversity-aware K-Center Greedy sample selection"
    )

    parser.add_argument("--text_dir", type=str, required=True,
                       help="Directory containing text data")
    
    parser.add_argument("--figure_dir", type=str, required=True,
                       help="Base directory for embedding figures")
    
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Base directory for output")
    
    parser.add_argument("--k", type=int, default=10000,
                       help="Number of samples to select")
    
    parser.add_argument("--neighbors", type=int, default=10,
                        help="Number of neighbors for density calculation")
    
    parser.add_argument("--density_power", type=float, default=0.5,
                       help="Power parameter for density calculation")
    
    parser.add_argument("--distance_power", type=float, default=1.0,
                       help="Power parameter for distance weighting")
    
    parser.add_argument("--gpu_id", type=int, default=0,
                       help="GPU ID to use (0-indexed)")
    
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    set_seed(args.seed)

    print(f"Loading embeddings from {args.figure_dir}")
    embeddings = load_dir_data(args.figure_dir)
    print(f"Loading text data from {args.text_dir}")
    text = load_dir_data(args.text_dir)
    print(f"Loaded embeddings: {embeddings.shape}, text: {len(text)} items")
    
    assert len(embeddings) == len(text), "Embeddings and text counts don't match!"
    
    print(f"Running K-center greedy algorithm on GPU {args.gpu_id}")
    centers_indices = novelselect(
        embeddings,
        k=args.k,
        gpu_id=args.gpu_id,
        neighbors=args.neighbors,
        density_power=args.density_power,
        distance_power=args.distance_power,
        seed=args.seed
    )
    
    selected_text = text[centers_indices].tolist()
    save_results(
        selected_text, centers_indices, args.output_dir, args.k, args.density_power, args.distance_power
    )
    print("Selection completed successfully!")

if __name__ == "__main__":
    main()