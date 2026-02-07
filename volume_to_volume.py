import numpy as np
import torch
import tqdm
import pandas as pd
import os
from numpy.linalg import norm
from scipy.spatial.distance import cdist  # 新增导入


def find_top_k_indices(values, k):
    # Use a combination of 'sorted' and 'enumerate' to sort the values while keeping track of indices
    sorted_values_with_indices = sorted(enumerate(values), key=lambda x: x[1], reverse=True)
    # Extract the indices of the top 50 values
    top_k_indices = [index for index, value in sorted_values_with_indices[:k]]
    return top_k_indices


def calc_similarity(arr1, arr2):
    oneandone = 0
    oneorzero = 0
    zeroandzero = 0
    for k in range(len(arr1)):
        if arr1[k] == 0 and arr2[k] == 0:
            zeroandzero += 1
        if arr1[k] == 1 and arr2[k] == 1:
            oneandone += 1
        if arr1[k] == 0 and arr2[k] == 1:
            oneorzero += 1
        if arr1[k] == 1 and arr2[k] == 0:
            oneorzero += 1
    return (oneandone / (oneandone + oneorzero))


save_folder = "./path_to_save/"
output_file_path = os.path.join(save_folder, f"v2v_align.txt")
os.makedirs(save_folder, exist_ok=True)

for n in range(0, 20):
    data_folder = f"./path_to_valid_latents_folder/align/epoch_{n}/image/"
    npz_files = [f for f in tqdm.tqdm(os.listdir(data_folder)) if f.endswith('.npz')]

    image_data_list = []
    accs = []

    # Load each .npz file and use the filename (without extension) as the accession number
    for npz_file in tqdm.tqdm(npz_files):
        file_path = os.path.join(data_folder, npz_file)
        image_data = np.load(file_path)["arr"][0]
        image_data_list.append(image_data)
        accs.append(npz_file.replace("npz", "nii.gz"))

    # Concatenate all loaded image data
    image_data = np.array(image_data_list)
    print(image_data.shape)

    # Load the validation labels
    df = pd.read_csv("/home2/Reports_label/valid_predicted_labels.csv")

    image_data_for_second = []
    accs_for_second = []

    # Filter the image data 2693
    for k in tqdm.tqdm(range(image_data.shape[0])):
        acc_second = accs[k]
        row_second = df[df['VolumeName'] == acc_second]
        num_path = np.sum(row_second.iloc[:, 1:].values[0])
        if num_path != 0:
            image_data_for_second.append(image_data[k])
            accs_for_second.append(accs[k])

    image_data_for_second = np.array(image_data_for_second)
    print(image_data_for_second.shape)

    cosine_dist = cdist(image_data, image_data_for_second, metric='cosine')
    cosine_similarity_matrix = 1.0 - cosine_dist  # (n_total, n_valid)

    print(f"Similarity matrix shape: {cosine_similarity_matrix.shape}")
    k_list = [1, 5, 10, 50]
    list_outs = {}
    ratios_external = []
    # Calculate the similarity for each image in the dataset
    for k in k_list:
        for i in tqdm.tqdm(range(image_data.shape[0])):
            first = image_data[i]
            acc_first = accs[i]
            row_first = df[df['VolumeName'] == acc_first]
            row_first = row_first.iloc[:, 1:].values[0]
            crosses = cosine_similarity_matrix[i].tolist()
            ratios_internal = []
            top_k_indices = find_top_k_indices(crosses, k)
            for index in top_k_indices:
                acc_second = accs_for_second[index]
                row_second = df[df['VolumeName'] == acc_second]
                row_second = row_second.iloc[:, 1:].values[0]
                ratio = calc_similarity(row_first, row_second)
                ratios_internal.append(ratio)

            ratios_external.append(np.mean(np.array(ratios_internal)))
        list_outs[k] = np.mean(np.array(ratios_external))

    list_write = []
    for k in k_list:
        write_str = f"n={n}, K={k}, MAP = {list_outs[k]}"
        print(write_str)
        list_write.append(write_str)

    with open(output_file_path, "a") as file:
        file.write("\n\n")
        file.write("\n".join(list_write))
