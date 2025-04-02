import json
import random
import pathlib
import numpy as np
import torch
import faiss
import os
from tqdm import tqdm


def set_device(gpu_id: int):
    if gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        return torch.device(f'cuda:{gpu_id}')
    else:
        return torch.device('cpu')
    
def set_seed(seed: int) -> None:
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

class FaissIndex:
    def __init__(self, data: np.ndarray, gpu_id: int):
        self.gpu_id = gpu_id
        self.device = set_device(gpu_id)
        self.data = data.astype(np.float32).reshape(-1, data.shape[-1])
        self.data = np.unique(self.data, axis=0)  # Remove duplicates
        self.res = faiss.StandardGpuResources()
        self.index = faiss.IndexFlatL2(self.data.shape[1])
        self.gpu_index = faiss.index_cpu_to_gpu(self.res, gpu_id, self.index)
        self.gpu_index.add(self.data)

    def search(self, query: np.ndarray, n_neighbors: int):
        query = query.astype(np.float32)
        distances, indices = self.gpu_index.search(query, n_neighbors + 1)  # +1 to exclude self
        return distances[:, 1:], indices[:, 1:]  # Exclude the first neighbor (self)

    def local_density(self, query_data: np.ndarray, n_neighbors=10, regularization=1e-9, power=1):
        distances, _ = self.search(query_data, n_neighbors)
        avg_distances = np.mean(distances, axis=-1)
        densities = 1 / np.power((avg_distances + regularization), power)
        return densities

def load_data(file: str) -> np.ndarray:
    with open(file, "r") as f:
        data = json.load(f)
    return np.array(data)

def load_dir_data(json_dir: str, desc: str = "Loading data") -> np.ndarray:
    files = sorted(pathlib.Path(json_dir).glob("*.json"), key=lambda x: int(x.stem))
    data_list = []
    
    for file in tqdm(files, desc=desc):
        data = load_data(str(file))  # 加载单个 JSON 文件的数据
        data_list.append(data)
    
    if not data_list:
        print(f"Warning: No JSON files found in {json_dir}")
        return np.array([])
        
    return np.concatenate(data_list)

def get_files_paths(directory):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return []

    files = sorted(
        pathlib.Path(directory).glob("*.json"),
        key=lambda x: int(x.stem)
    )
    return [str(file) for file in files]