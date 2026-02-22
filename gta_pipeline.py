import numpy as np
import random, os, json, gc, math
from metrics import Metric, gta_kmeans, gta_hierarchical
from encoders import get_features, get_encoder
from datasets import get_dataset
from utils import stratified_sample, random_sample 

def probe(encoder_name, dataset_name, encoder_target_dim,
          stratify_sample = True, sample_size=500, 
          image_size= 224, random_state=42,
          relative_clustering_scales= [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
          metric = Metric.HIERARCHICAL,
          chkpt_path="./chkpt", chkpt_name="checkpoint",  verbose=True):
    
    encoder, processor = get_encoder(encoder_name)
    dataset = get_dataset(dataset_name, 'train', processor=None)

    # Set random seed
    random.seed(random_state)
    np.random.seed(random_state)

    # Create checkpoint
    if verbose: print("Checking path ...")
    if not os.path.exists(chkpt_path):
        os.mkdir(chkpt_path)
    
    # Take a subset
    if verbose: print(f"Sampling {sample_size} images ...")
    if stratify_sample:
        sample_data = stratified_sample(dataset, sample_size)
    else:
        sample_data = random_sample(dataset, sample_size)
    if verbose: print("Clearing dataset from memory ...")
    del dataset
    gc.collect()

    # Extracting images and labels
    if verbose: print("Extracting images and labels ...")
    all_images = []
    image_labels= []
    for idx, (image, label) in enumerate(sample_data):
        all_images.append(image)
        image_labels.append(label)
    if verbose: print("Clearing sample from memory ...")
    del sample_data
    gc.collect()

    # Get the features of each image and augmentations
    if verbose: print("Getting images embeddings ...")
    features = []
    batch_size = 64  
    for i in range(0, len(all_images), batch_size):
        batch_images = all_images[i:i+batch_size]
        batch_processed = processor(batch_images, return_tensors='pt')['pixel_values']
        batch_features = get_features(encoder, batch_processed, encoder_target_dim, "cuda")
        batch_features = batch_features.cpu().numpy()
        features.append(batch_features)
    features = np.vstack(features).astype('float64')

    if verbose: print("Clearing images from memory ...")
    del all_images
    gc.collect()

    if verbose: print("Clearing model from memory ...")
    del encoder
    gc.collect()
    
    # Compute GTA metric
    if verbose: print("Computing metrics ...")
    to_integer = lambda x: int(x) if float(x).is_integer() else math.ceil(x)
    clustering_scales = [to_integer(relative_clustering_scale * sample_size) for relative_clustering_scale in relative_clustering_scales]
    if metric == Metric.KMEANS:
        gta_values_dict = gta_kmeans(features, image_labels, clustering_scales)
    elif metric == Metric.HIERARCHICAL:
        gta_values_dict = gta_hierarchical(features, image_labels, clustering_scales)

    if verbose: print("Clearing embeddings from memory ...")
    del features
    gc.collect()

    # Store the metric in checkpoint format
    if verbose: print("Saving to chekpoint ...")
    config = {
        'stratify_sample': stratify_sample,
        'sample_size': sample_size,
        'encoder_target_dim': encoder_target_dim,
        'image_size': image_size,
        'random_state': random_state,
    }
    
    results = {
        'encoder': encoder_name,
        'dataset': dataset_name,
        'config': config,
        'gta_values': gta_values_dict,
    }

    # Write to checkpoint
    chkpt_file = os.path.join(chkpt_path, f"{chkpt_name}.json")
    if os.path.exists(chkpt_file):
        chkpt = json.load(open(chkpt_file, "r"))
    else:
        chkpt = []
    chkpt.append(results)
    json.dump(chkpt, open(chkpt_file, "w"), ensure_ascii=True, indent=4)