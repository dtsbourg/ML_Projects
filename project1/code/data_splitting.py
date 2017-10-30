import numpy as np
from auxiliary import *

# Contains the function to split the data in 8 subsets


def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0,size):
        list_of_objects.append( list() )
    return list_of_objects


def split_data_boson(ids, x, y):  
    ids_split = init_list_of_objects(8)
    y_split = init_list_of_objects(8)
    x_split = init_list_of_objects(8)
    
    for i in range(x.shape[0]):
        if x[i,22] == 0:
            j = 0
            ranges = [(1,4), (7,12), (13,22)] # column 29 is null
            keep_idx = build_idx(ranges)
            ranges2 = [(0,4), (7,12), (13,22)] # column 29 is null
            keep_idx2 = build_idx(ranges2)
            if x[i,0] == -999:
                ids_split[j].append(ids[i])
                y_split[j].append(y[i])
                x_split[j].append(x[i,keep_idx])
            else :
                ids_split[j+1].append(ids[i])
                y_split[j+1].append(y[i])
                x_split[j+1].append(x[i,keep_idx2])
        elif x[i,22] == 1:
            j = 2
            ranges = [(1,4), (7,12), (13,22), (23, 26)]
            keep_idx = build_idx(ranges)
            ranges2 = [(0,4), (7,12), (13,22), (23, 26)]
            keep_idx2 = build_idx(ranges2)
            if x[i,0] == -999:
                ids_split[j].append(ids[i])
                y_split[j].append(y[i])
                x_split[j].append(x[i,keep_idx])
            else :
                ids_split[j+1].append(ids[i])
                y_split[j+1].append(y[i])
                x_split[j+1].append(x[i,keep_idx2])
        elif x[i,22] == 2:
            j = 4
            ranges = [(1,22), (23,30)]
            keep_idx = build_idx(ranges)
            ranges2 = [(0,22), (23,30)]
            keep_idx2 = build_idx(ranges2)
            if x[i,0] == -999:
                ids_split[j].append(ids[i])
                y_split[j].append(y[i])
                x_split[j].append(x[i,keep_idx])
            else :
                ids_split[j+1].append(ids[i])
                y_split[j+1].append(y[i])
                x_split[j+1].append(x[i,keep_idx2])
        elif x[i,22] == 3:
            j = 6
            ranges = [(1,22), (23,30)]
            keep_idx = build_idx(ranges)
            ranges2 = [(0,22), (23,30)]
            keep_idx2 = build_idx(ranges2)
            if x[i,0] == -999:
                ids_split[j].append(ids[i])
                y_split[j].append(y[i])
                x_split[j].append(x[i,keep_idx])
            else :
                ids_split[j+1].append(ids[i])
                y_split[j+1].append(y[i])
                x_split[j+1].append(x[i,keep_idx2])
        
        # transform lists in arrays
    for i in range(len(ids_split)):
        ids_split[i] = np.array(ids_split[i])
        y_split[i] = np.array(y_split[i])
        x_split[i] = np.array(x_split[i])

    return ids_split, y_split, x_split

