import numpy as np
import os
from tqdm import tqdm

np.random.seed(42)

save_folder = "./path_to_save/"
output_file_path = os.path.join(save_folder, f"r2v_align.txt")
os.makedirs(save_folder, exist_ok=True)


for n in range(0, 30):
    path_to_valid_latents_folder = f'/home2/path_to_valid_latents_folder/align/epoch_{n}'
    image_folder = f"{path_to_valid_latents_folder}/image"
    text_folder = f"{path_to_valid_latents_folder}/text"

    image_npz_files = [f for f in os.listdir(image_folder) if f.endswith('.npz')]
    text_npz_files = [f for f in os.listdir(text_folder) if f.endswith('.npz')]

    image_npz_files.sort()
    text_npz_files.sort()

    n_samples = len(image_npz_files)

    image_data = np.zeros((n_samples, 512))
    text_data = np.zeros((n_samples, 512))


    for idx, npz_file in enumerate(tqdm(image_npz_files, desc="Loading images")):
        file_path = os.path.join(image_folder, npz_file)
        image_data[idx] = np.load(file_path)["arr"][0]

    for idx, npz_file in enumerate(tqdm(text_npz_files, desc="Loading texts")):
        file_path = os.path.join(text_folder, npz_file)
        text_data[idx] = np.load(file_path)["arr"][0]

    print(f"Data shape: {image_data.shape}")

    image_data_norm = image_data / np.linalg.norm(image_data, axis=1, keepdims=True)
    text_data_norm = text_data / np.linalg.norm(text_data, axis=1, keepdims=True)
    similarity_matrix = text_data_norm @ image_data_norm.T


    k_values = [5, 10, 50, 100]
    list_texts = []

    clip_recalls = {}
    for k in k_values:

        top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, :k]
        hits = 0
        for i in range(n_samples):
            if i in top_k_indices[i]:
                hits += 1
        clip_recalls[k] = hits / n_samples

    random_recalls = {}
    random_similarity = np.random.randn(n_samples, n_samples)

    for k in k_values:
        random_top_k = np.argsort(-random_similarity, axis=1)[:, :k]
        random_hits = 0
        for i in range(n_samples):
            if i in random_top_k[i]:
                random_hits += 1
        random_recalls[k] = random_hits / n_samples

    for k in k_values:
        write_str = f"n={n}, K={k}, Recall = {clip_recalls[k]}, rand= {random_recalls[k]}"
        print(write_str)
        list_texts.append(write_str)

    with open(output_file_path, "a") as file:
        file.write("\n\n")
        file.write("\n".join(list_texts))