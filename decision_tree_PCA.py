import argparse

import numpy as np

from utility import load_data
from decision_tree import impurity, get_best_feature_to_split, split
from pca import calculate_eigen_of_cov_mat, choose_k_largest_eigen_transform

parser = argparse.ArgumentParser()
parser.add_argument('--xlsx', help='need a xlsx file. \
                    e.g. --xlsx example.xlsx', dest='XLSX')
args = parser.parse_args()



dataset, labels  = load_data(args.XLSX)
features, class_col = [row[:-1] for row in dataset], [row[-1] for row in dataset]

mean_vector, e_values, e_vectors, norm_Features = calculate_eigen_of_cov_mat(features)
k = 20
components = choose_k_largest_eigen_transform(norm_Features, e_vectors, mean_vector, k)

class_col = np.array(class_col, dtype='int32')


new_dataset = np.zeros((len(dataset), len(dataset[0])))
new_dataset[:,:-1] = components
new_dataset[:,-1] = class_col

new_dataset = new_dataset.tolist()

class_values = list(set(row[-1] for row in dataset))

root = get_best_feature_to_split(new_dataset, 'entropy')

split(root, class_values, max_depth=3, min_size=5, depth=1, mode='entropy')

print(root)

