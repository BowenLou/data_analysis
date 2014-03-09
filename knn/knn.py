# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
import math
import random

# when k is an even number, it is likely that we can't decide the class of a test tuple, then we skip the even number of k.

def calculate_distance(test_row, train_data):
    test_data_columns = test_row.index
    distance_list = []
    for i in train_data.index:
        distance = sum((test_row - train_data[test_data_columns].ix[i])**2)
        distance_list.append(distance)
    return distance_list

def predict(test_row, train_data, k):
    train_data.index = np.arange(0, len(train_data))
    distance_list = calculate_distance(test_row, train_data)
    min_k_distance = sorted(distance_list)[:k]
    min_k_distance_index = [distance_list.index(dist) for dist in min_k_distance]
    nn_class = [train_data.ix[distance_list.index(dist), 'class'] for dist in min_k_distance]
    return max(set(nn_class), key = nn_class.count) 
    
    
if __name__ == '__main__':
    analysis_data = pd.read_csv('./Wisconsin_breast_cancer/wdbc.data', header = None)

    base_names = ['radius', 'texture', 'perimeter', 'area', 'smooth', 'compact', 'concav', 'conpoints', 'symmetry', 'fracdim']

    names = ['m' + name for name in base_names]
    names += ['s' + name for name in base_names]
    names += ['e' + name for name in base_names]

    analysis_data.columns = ['id', 'class'] + names
    
    test_columns = ['earea', 'esmooth', 'mtexture']
    train_columns = test_columns + ['class']
    
    df_pre = analysis_data[test_columns]
    
    # normalized data
    df = (df_pre - df_pre.min(axis = 0)) / (df_pre.max(axis = 0) - df_pre.min(axis = 0))
    df['class'] = Series(analysis_data['class'], index = df.index)
    
    df_index_range = list(df.index)
    random.shuffle(df_index_range)
 
    fold_range_list = []
    fold_mark = 0
    fold_value_num = int(len(df_index_range) / 10) + 1
    
    while fold_mark < len(df_index_range):
        fold_mark_right = (fold_mark + fold_value_num)
        if fold_mark_right >= len(df_index_range):
            fold_mark_right = len(df_index_range)
        fold_range_list.append(df_index_range[fold_mark: fold_mark_right])
        fold_mark += fold_value_num
    
    k_list = np.arange(1, 12, 2)
    average_result = []

    for k in k_list:
        print 'k is: ', k
        knn_accuracy_list = []
        for test_data_index in fold_range_list:
            train_data_index = [x for x in df_index_range if x not in test_data_index]
            train_data = df.ix[train_data_index]
            train_data = train_data[train_columns]
            test_data = df.ix[test_data_index]
            test_data_class = test_data['class']
            test_data = test_data[test_columns]
            
            correct_num = 0
            all_num = len(test_data_index)
            for i in test_data_index:
                # print i, test_data.ix[i]
                if predict(test_data.ix[i], train_data, k) == test_data_class[i]:
                    correct_num += 1
            
            knn_accuracy = correct_num / all_num
            print 'accuracy:' ,  knn_accuracy
            knn_accuracy_list.append(knn_accuracy)
        
        knn_accuracy_series = Series(knn_accuracy_list)
        print 'k: ', k, 'knn average accuracy:', knn_accuracy_series.mean()
        average_result.append(knn_accuracy_series.mean())
    print 'all average accuracy result for each k: ', average_result
    
    fig, axes = plt.subplots(nrows = 2, ncols = 1)
    fig.suptitle('KNN Result')

    axes[0].set_title('simple line plot result')
    axes[0].plot(k_list, average_result)
    axes[0].set_xlabel('k')
    axes[0].set_ylabel('Average Accuracy')
    axes[0].tick_params(axis='x', labelsize=11)
    axes[0].tick_params(axis='y', labelsize=11)

    axes[1].set_title('scatter plot result')
    axes[1].scatter(k_list, average_result)
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('Average Accuracy')
    axes[1].tick_params(axis='x', labelsize=11)
    axes[1].tick_params(axis='y', labelsize=11)
    
    plt.subplots_adjust(left = 0.18, right = 0.92, hspace = 0.4)
    plt.savefig('knn_result_plot.pdf')



