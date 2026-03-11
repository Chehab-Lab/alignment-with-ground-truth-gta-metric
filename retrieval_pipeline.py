import os
import json
import torch
import numpy as np
import faiss
from tqdm import tqdm
from typing import List
from datasets import get_dataset
from encoders import get_encoder, get_features
from torch.utils.data import DataLoader

def exists(path):
    return os.path.exists(path)

def _collate_fn(batch):
    images, labels = zip(*batch) 
    return list(images), torch.tensor(labels)

def _get_embeddings(encoder, dataset, img_processor, target_dim, device):
    embeddings = []
    all_labels = []
    for batch_images, labels in tqdm(dataset):
        batch_images = img_processor(batch_images, return_tensors="pt")["pixel_values"].to(device)
        batch_emb = get_features(encoder, batch_images, target_dim, device)
        embeddings.append(batch_emb)
        all_labels.append(labels)
    embeddings = torch.cat(embeddings)
    labels = torch.cat(all_labels)
    embeddings = embeddings.cpu().numpy()
    labels = labels.cpu().numpy()
    return embeddings, labels

def mean_average_precision(embeddings, labels, k):
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    _, neighbors = index.search(embeddings, k)
    neighbors = neighbors[:, 1:] 
    retrieved_labels = labels[neighbors]
    matches = (retrieved_labels == labels[:, None])
    cumulative_matches = np.cumsum(matches, axis=1)
    ranks = np.arange(1, k) 
    precision_at_k = cumulative_matches / ranks
    precision_at_k *= matches
    total_relevant = np.sum(labels[:, None] == labels[None, :], axis=1) - 1
    normalizer = np.minimum(total_relevant, k - 1)
    normalizer = np.where(normalizer == 0, 1, normalizer)
    AP = np.sum(precision_at_k, axis=1) / normalizer
    return np.mean(AP)

def evaluate_retrieval(encoder_name: str, 
                       dataset_name: str, 
                       target_dim: int,
                       k_list: List[int] = [5, 9],
                       batch_size: int = 64,
                       device: str = "cuda",
                       checkpoint_folder: str = "./checkpoints", 
                       checkpoint_name: str = "results",
                       verbose: bool = True):

    if verbose: print("\nVerifying checkpoints....") 
    if not exists(checkpoint_folder): os.mkdir(checkpoint_folder)
        
    checkpoint_file = os.path.join(checkpoint_folder, checkpoint_name+".json")

    encoder, img_processor = get_encoder(encoder_name, device=device)
    
    dataset = get_dataset(dataset_name)

    dataset = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=_collate_fn
    )
    
    if verbose: print(f"\nGetting image embeddings....")

    embeddings, labels = _get_embeddings(encoder, dataset, img_processor, target_dim, device)


    if verbose: print("\nComputing mAP@k....")

    mAP_results = []
    for k in k_list:
        result = mean_average_precision(embeddings, labels, k)
        mAP_results.append({f'mAP@{k}': result})

    if verbose: print("\nSaving checkpoint....")

    results = {
        'encoder': encoder_name,
        'dataset': dataset_name,
        'mAP': mAP_results
    }
    
    checkpoint = json.load(open(checkpoint_file)) if exists(checkpoint_file) else []
    checkpoint.append(results)
    json.dump(checkpoint, open(checkpoint_file, 'w'), ensure_ascii=True, indent=4)

def _test_retreival_pipeline():
    encoder_name = "microsoft/resnet-50"
    dataset_name = "gpr1200"
    evaluate_retrieval(encoder_name, dataset_name, 2048, [5], batch_size=256)