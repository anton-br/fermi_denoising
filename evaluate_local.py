#!/usr/bin/env python
import sys, os, os.path
import warnings
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tqdm
import argparse
from sklearn.neighbors import BallTree
warnings.filterwarnings('ignore')
from scipy.spatial import distance as sc_distance

from sys import argv


def distance(true_coords_vector, coords_vector):
    if len(coords_vector) == 0 and len(true_coords_vector) != 0:
        res = 0
        return np.sum(sc_distance.cdist(np.array([[100,100]]), true_coords_vector, 'euclidean')) + 16*len(true_coords_vector)
    if len(coords_vector) != 0 and len(true_coords_vector) == 0:
        return np.sum(sc_distance.cdist(np.array([[100,100]]), coords_vector, 'euclidean')) + 16*len(coords_vector)
    if len(coords_vector) == 0 and len(true_coords_vector) == 0:
        return 0
    true_coords = np.vstack({tuple(row) for row in true_coords_vector})#np.unique(true_coords_vector, axis=0)
    coords = np.vstack({tuple(row) for row in coords_vector})#np.unique(coords_vector, axis=0)
    tree_true_coords = BallTree(true_coords)
    tree_coords = BallTree(coords)
    distance_from_true_array, _ = tree_true_coords.query(coords)
    distance_from_found_array, _ = tree_coords.query(true_coords)
    distance_from_found = np.sum(distance_from_found_array)
    distance_from_true = np.sum(distance_from_true_array)
    return np.sum(distance_from_found_array) + np.sum(distance_from_true_array) 

def distance_extend(true_coords_vector, coords_vector):
    #print("extend")
    if len(coords_vector) == 0 and len(true_coords_vector) != 0:
        res = 0
        return np.sum(sc_distance.cdist(np.array([[100,100]]), true_coords_vector, 'euclidean')) + 25*len(true_coords_vector), [[0, len(true_coords_vector)], [0, 200*200-len(true_coords_vector)]]
    if len(coords_vector) != 0 and len(true_coords_vector) == 0:
        return np.sum(sc_distance.cdist(np.array([[100,100]]), coords_vector, 'euclidean')) + 25*len(coords_vector), [[0, 0], [len(coords_vector), 200*200-len(coords_vector)]]
    if len(coords_vector) == 0 and len(true_coords_vector) == 0:
        return 0, [[0, 0], [0, 200*200]]
    true_coords = np.vstack({tuple(row) for row in true_coords_vector})#np.unique(true_coords_vector, axis=0)
    coords = np.vstack({tuple(row) for row in coords_vector})#np.unique(coords_vector, axis=0)
    tree_true_coords = BallTree(true_coords)
    tree_coords = BallTree(coords)
    distance_from_true_array, _ = tree_true_coords.query(coords)
    distance_from_found_array, _ = tree_coords.query(true_coords)
    #print(distance_from_true_array)
    #print(distance_from_found_array)
    threshold = (1.6)**2
    if False:
        TP = np.sum((distance_from_true_array<threshold)*1)
        FN = np.sum((distance_from_true_array>threshold)*1)
        FP = np.sum((distance_from_found_array>threshold)*1)
    else:
        TP = np.sum((distance_from_found_array<threshold)*1)
        FN = np.sum((distance_from_found_array>threshold)*1)
        FP = np.sum((distance_from_true_array>threshold)*1)
    m = [[TP, FN],[FP, 200*200-FN-len(true_coords_vector)]]
    distance_from_found = np.sum(distance_from_found_array)
    distance_from_true = np.sum(distance_from_true_array)
    #print(np.sum(distance_from_found_array) + np.sum(distance_from_true_array))
    return np.sum(distance_from_found_array) + np.sum(distance_from_true_array), np.array(m)


def distance_for_threshold(true_coords_vector, matrix):
    means, stds, maxs = [], [], []
    side = 3
    for coords in true_coords_vector:
        x1 = max(coords[0] - side // 2, 0) 
        x2 = max(coords[0] + (side - side // 2), matrix.shape[0]) 
        y1 = max(coords[1] - side // 2, 0) 
        y2 = max(coords[1] + (side - side // 2), matrix.shape[1]) 
        new_mean = np.mean(matrix[x1:x2, y1:y2])
        new_max = np.max(matrix[x1:x2, y1:y2])
        new_std = np.std(matrix[x1:x2, y1:y2])
        means.append(new_mean)
        stds.append(new_std)
        maxs.append(new_max)
    return means, stds, maxs


def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def addMask(x, y):
    masked_x = np.zeros((len(x), len(x[0])))
    K = 1
    S = int(K / 2)
    T = K - S
    for pair in y:
        pair = pair.astype(int)
        i,j = pair[0], pair[1]
        masked_x[np.max([0, i-S]):np.min([199,i+T]), np.max([0, j-S]):np.min([199,j+T])] = 1
    return masked_x

import PIL

def change(x0, y0, x1, y1, reduced,im_np):
    if x1 >= 200 or y1 >= 200 or x1 < 0 or y1 < 0:
        return reduced
    proba = im_np[x0, y0]
    if reduced[x1,y0]==1:
        if im_np[x1, y1] < proba:
            reduced[x1, y1] = 0
        else:
            reduced[x0, y0] = 0
    return reduced

def localize(x0, y0, reduced, im_np):
    reduced = change(x0, y0, x0-1, y0, reduced,im_np)
    reduced = change(x0, y0, x0+1, y0, reduced,im_np)
    reduced = change(x0, y0, x0, y0-1, reduced,im_np)
    reduced = change(x0, y0, x0, y0+1, reduced,im_np)
    return reduced
    
    
    
def keepPoints(reduced, im_np, x, y_):
    n = len(x)
    for i in range(n):
        for j in range(n):
            reduced = localize(x[i], y_[i], reduced, im_np)
            
    return reduced

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''Score results with known reference (for training). 
       Each file during testing should be saved individually with the same name as input to its own folder.''')
    parser.add_argument('--i', nargs='?', const='./data/reference_data', default='./data/reference_data', type=str, help='path to the folder with the outputs of the neural network')
    parser.add_argument('--r', nargs='?', const='./data/reference_data', default='./data/reference_data', type=str, help='path to the folder with corresponding reference data')
    parser.add_argument('--s', nargs='?', const='test', default='test', type=str, help='subfolder of the data set which should be evaluated (both folders given by --i and --r should contain a folder with this name')
    args = parser.parse_args(argv[1:])


    input_dir = os.path.join(args.i, args.s)
    ref_dir = os.path.join(args.r, args.s)

    ref_names = os.listdir(ref_dir)
    res_names = os.listdir(input_dir)
    distances1 = []
    distances2 = []
    distances_ = []
    matrix = np.zeros((2,2))
    roc = []
    stars = 0
    print(res_names)
    for i in tqdm.tqdm(range(len(res_names))): 
        Y_train = np.load(os.path.join(ref_dir, res_names[i].split('.')[0] + '.npy'))
        y_train = np.load(os.path.join(input_dir, res_names[i].split('.')[0] + '.npy'))
        dist, m = distance_extend(np.array(Y_train), np.array(y_train))

        stars += len(Y_train)

        dist, m = distance_extend(np.array(Y_train), np.array(y_train))
        if dist != np.inf:
            distances_.append(dist) 
        matrix = matrix + m
        try:
            SEN = m[0][0]/(m[0][0]+m[0][1])
            SPE = m[1][1]/(m[1][1]+m[1][0])
            roc.append([1-SPE, SEN])
        except:
            continue
    print(distances_) 
    print(str(np.mean(distances_)))
    print(str(np.std(distances_)))
    print(matrix)
    print(stars)
    plt.hist(distances_, bins=100)
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.title('Histogram of distance distribution on '+args.s+' set')
    plt.grid(True)
    plt.show()
    


