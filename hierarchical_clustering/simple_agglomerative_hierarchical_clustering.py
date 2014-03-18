# agglomerative hierarchical clustering with single-linkage approach
from pandas import DataFrame
import itertools;
import math;
import numpy as np

def calculate_dist(data, linkage, cluster_x, cluster_y):
    dist = []
    for item_x in cluster_x:
        for item_y in cluster_y:
            dist.append(math.sqrt(sum((data.ix[item_x] - data.ix[item_y]) ** 2)))
    
    if linkage == 'min':
        return min(dist)
    elif linkage == 'max':
        return max(dist)
    elif linkage == 'mean':
        mean_dist = math.sqrt(sum((np.mean(data.ix[list(cluster_x)]) - np.mean(data.ix[list(cluster_y)])) ** 2))
        return mean_dist
    elif linkage == 'average':
        return np.mean(dist)
    
def h_cluster(data, k, linkage):
    if k > len(data.index):
        print 'the number of cluster input should not be larger than the number of objects in the data!'
        raise Exception
    
    if not (linkage == 'min' or linkage == 'max' or linkage == 'mean' or linkage == 'average'):
        print 'you should input linkage method string: min, max, mean, average'
        raise Exception
    else:
        # normalize original data
        data = (data - data.min(axis = 0)) / (data.max(axis = 0) - data.min(axis = 0))
        initial_clusters = []
        for index in data.index:
            list_index = []
            list_index.append(index)
            initial_clusters.append(list_index)
        # print initial_clusters
        data_length = len(data.index)

        if data_length == k:
            print linkage + ' linkage ' + 'final clusters: ', initial_clusters
            return initial_clusters
        else:
            initial_tuple_clusters = []
            for cluster in initial_clusters:
                initial_tuple_clusters.append(tuple(cluster))

            clusters = initial_tuple_clusters
            print data_length, 'clusters: ', clusters
            for iterate in range(data_length - k):
                dist_dict = {}
                for cluster_tuple in list(itertools.combinations(clusters, 2)):
                    dist_dict[cluster_tuple] = calculate_dist(data, linkage, cluster_tuple[0], cluster_tuple[1])
                dist_dict_sort_list = sorted(dist_dict.items(), key = lambda x: x[1])
                clusters = [cluster for cluster in clusters if cluster != dist_dict_sort_list[0][0][0] and cluster != dist_dict_sort_list[0][0][1]]
                clusters.append(dist_dict_sort_list[0][0][0] + dist_dict_sort_list[0][0][1])
                print data_length - iterate - 1, 'clusters:', clusters

        final_result = []
        for cluster in clusters:
            list_cluster = list(cluster)
            final_result.append(list_cluster)
        
        print linkage + ' linkage ' + 'final clusters: ', final_result
        return final_result

# test
test_data = DataFrame(np.random.rand(7, 4) * 20)
print 'sample test data:\n', test_data

print 'note: sample data is a dataframe in which all values of all columns are real numbers. Each row corresponds to a single object and each column corresponds to a dimension.'

print 'linkage measure min distance: '
h_cluster(test_data, 6, 'min')
print 'linkage measure max distance: '
h_cluster(test_data, 2, 'max')
print 'linkage measure mean distance: '
h_cluster(test_data, 4, 'mean')
print 'linkage measure average distance: '
h_cluster(test_data, 5, 'average')


