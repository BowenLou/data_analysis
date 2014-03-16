# -*- coding: utf-8 -*-
from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame, Series

data = pd.read_csv('./1000-out2.csv', header = None)
data.index = data[0]

import itertools

# Fk−1 × Fk−1 method: Assumes F is in lexicographic order. Return candidates in lexicographic order.
# F is list of list, k is the length: length of the item in F plus 1

def generate_candidates(F, k):
    length_Fk_1 = len(F[0])
    if not all([len(Fk_1_list_item) == length_Fk_1 for Fk_1_list_item in F]):
        print 'length of the item in the Fk-1 is not the same.'
        raise Exception
    else:
        if k != len(F[0]) + 1:
            print 'k is not the length of the item in Fk-1 plus 1.'
            raise Exception
        else:
            Fk_2_list = [Fk_1_list_item[0: len(Fk_1_list_item) - 1] for Fk_1_list_item in F]
            Fk_2_list_set = list(set(tuple(Fk_2_list_item) for Fk_2_list_item in Fk_2_list if Fk_2_list.count(Fk_2_list_item) > 1))
            # print Fk_2_list_set
            
            Fk_2_remain_list = []
            for Fk_2_tuple_item in Fk_2_list_set:
                remain_list = []
                for Fk_1_list_item in F:
                    if Fk_1_list_item[0: len(Fk_1_list_item) - 1] == list(Fk_2_tuple_item):
                        remain_list.append(Fk_1_list_item[-1])
                Fk_2_remain_list.append(remain_list)
            # print Fk_2_remain_list
            
            candidate_result = []
            for list_item, remain_list in zip(Fk_2_list_set, Fk_2_remain_list):
                for remain_tuple in list(itertools.combinations(remain_list, 2)):
                    candidate_result.append(list(list_item) + (list(remain_tuple)))
            order_candidate_result = sorted(candidate_result)
    return order_candidate_result

def prune_candidates(F, C, k):
    prune_candidate_list = []
    for candidate_item in C:
        # print list(itertools.combinations(candidate_item, k - 1))
        for item in list(itertools.combinations(candidate_item, k - 1)):
            if list(item) not in F:
                prune_candidate_list.append(candidate_item)
                break
    # print prune_candidate_list
    return prune_candidate_list

def support_count(D, minsupport, C):
    result_minimum_support_list = []
    total_count = len(D.index)
    for candidate_list in C:
        count = 0
        # print candidate_list
        for index in D.index:
            if all([(D.ix[index, candidate_item] == 1) for candidate_item in candidate_list]):
                # print index
                count +=1
        if count/ total_count >= minsupport:
            result_minimum_support_list.append(candidate_list)
    
    # print result_minimum_support_list
    return result_minimum_support_list 

def find_frequent_itemsets(D, minsupport):
    F_1_list = []
    for F_1_list_item in D.columns:
        empty_list = []
        empty_list.append(F_1_list_item)
        F_1_list.append(empty_list)
    # print F_1_list
    F_1_frequent_list = support_count(D, minsupport, F_1_list)
    F_k = F_1_frequent_list
    
    frequent_itemset = F_k
    while(len(F_k) != 0):
        k = len(F_k[0])
        candidate_list = generate_candidates(F_k, k + 1)
        prune_candidate_list = prune_candidates(F_k, candidate_list, k + 1)
        current_result_candidate_list = [candidate_item for candidate_item in candidate_list if candidate_item not in prune_candidate_list]
        final_result_candidate_list = support_count(D, minsupport, current_result_candidate_list)
        frequent_itemset += final_result_candidate_list
        F_k = final_result_candidate_list
    
    print 'frequent itemsets:\n', frequent_itemset
    return frequent_itemset

# test
find_frequent_itemsets(data[data.columns[1:len(data.columns)]], 0.04)




