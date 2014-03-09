from __future__ import division
import numpy as np;
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
import math
import random

analysis_data = pd.read_csv('./Wisconsin_breast_cancer/wdbc.data', header = None)

base_names = ['radius', 'texture', 'perimeter', 'area', 'smooth', 'compact', 'concav', 'conpoints', 'symmetry', 'fracdim']

names = ['m'+name for name in base_names]
names +=  ['s'+name for name in base_names]
names += ['e'+name for name in base_names]

analysis_data.columns = ['id', 'class'] + names

# print analysis_data.columns

class Node(object):
    def __init__(self):
        self.label = None
        self.node_type = None
        self.data = None
        self.majority = None
        self.successor = []

class DecisionTree(object):
    "Decision Tree Class"
    def __init__(self, data, class_column, splitting_rule, max_height):
        if (not isinstance(data, pd.DataFrame) or class_column not in data.columns or not(splitting_rule == 'infogain' or splitting_rule == 'gainratio' or splitting_rule == 'gini') or (max_height is not None and (max_height != int(max_height) or max_height < 0))
            ):
            raise Exception
        
        self.data = data
        self.class_column = class_column
        self.splitting_rule = splitting_rule
        self.max_height = max_height
        self.attr_list = list(self.data.columns)[:-1] # just remove 'class' attribute
        
    def buildTree(self, data, attr_list, height):
        node = Node()
        x = self.class_column
        self.attr_list = attr_list
        node.data = data
        
        # print node.data
        
        if (node.data[x] == 'B').all() or (node.data[x] == 'M').all():
            node.label = node.data[x].values[0]
            node.node_type = 'leaf'
            node.majority = None
            # print 'all:', 'node_label:', node.label, 'node_type:', node.node_type
            return node
        elif len(self.attr_list) == 0:
            if sum(node.data[x] == 'B') >= sum(node.data[x] == 'M'):
                node.label = 'B'
            else:
                node.label = 'M'
            node.node_type = 'leaf'
            node.majority = None
            # print 'no attr:', 'node_label:', node.label, 'node_type:', node.node_type
            return node
        else:
            node.node_type = 'internal'
            # print node.data
            if sum(node.data[x] == 'B') >= sum(node.data[x] == 'M'):
                node.majority = 'B'
            else:
                node.majority = 'M'
            node.label = self.attribute_selection_method(node)
            # print 'node_label:', node.label, 'node_type:', node.node_type
            
            for partition_data in self.partitions(node.label, node):
                if len(partition_data) == 0:
                    if len(node.data[x] == 'B') >= len(node.data[x] == 'M'):
                        node.label = 'B'
                    else:
                        node.label = 'M'
                    node.node_type = 'leaf'
                    node.majority = None
                else:
                    # print 'height', height
                    node.successor.append(self.buildTree(partition_data, self.attr_list, height + 1))
            # print 'node_label:', node.label, 'node_type:', node.node_type
            return node


    def attribute_selection_method(self, node):
        if self.splitting_rule == 'infogain':
            splitting_criterion = self.attribute_infogain(node)
        elif self.splitting_rule == 'gainratio':
            splitting_criterion = self.attribute_gainratio(node)
        elif self.splitting_rule == 'gini':
            splitting_criterion = self.attribute_gini(node)
        return splitting_criterion

    def attribute_infogain(self, node):
        info_attr_list = []
        mid_attr_list = []

        for m in self.attr_list:
            attr_min = node.data[m].min()
            attr_max = node.data[m].max()
            attr_mid = (attr_max + attr_min) / 2
            # print m, attr_min, attr_max, attr_mid
            mid_attr_list.append(attr_mid)
            
            left_attr_count = sum(node.data[m] <= attr_mid)
            right_attr_count = sum(node.data[m] > attr_mid)
            m_left_attr_count = sum(node.data[node.data[m] <= attr_mid][self.class_column] == 'M')
            b_left_attr_count = sum(node.data[node.data[m] <= attr_mid][self.class_column] == 'B')
            m_right_attr_count = sum(node.data[node.data[m] > attr_mid][self.class_column] == 'M')
            b_right_attr_count = sum(node.data[node.data[m] > attr_mid][self.class_column] == 'B')
            # print m, left_attr_count, m_left_attr_count, b_left_attr_count, right_attr_count, m_right_attr_count, b_right_attr_count
            
            left_attr_calculate_info = self.__calculate_info(m_left_attr_count, b_left_attr_count)
            right_attr_calculate_info = self.__calculate_info(m_right_attr_count, b_right_attr_count)
            info_attr = (left_attr_count / (left_attr_count + right_attr_count)) * left_attr_calculate_info + (right_attr_count / (left_attr_count + right_attr_count)) * right_attr_calculate_info

            # print info_attr
            info_attr_list.append(info_attr)
        
        # print min(info_attr_list)
        criterion_attr_index = info_attr_list.index(min(info_attr_list))
        
        criterion_tuple = (node.data.columns[criterion_attr_index], mid_attr_list[criterion_attr_index])
#        print criterion_tuple
        return criterion_tuple
            

    def attribute_gainratio(self, node):
        gainratio_attr_list = []
        mid_attr_list = []
        
        m_count = sum(node.data[self.class_column] == 'M')
        b_count = sum(node.data[self.class_column] == 'B')
        base_info = self.__calculate_info(m_count, b_count)
        # print 'base:', m_count, b_count, base_info

        for m in self.attr_list:
            attr_min = node.data[m].min()
            attr_max = node.data[m].max()
            attr_mid = (attr_max + attr_min) / 2
            # print m, attr_min, attr_max, attr_mid
            mid_attr_list.append(attr_mid)

            left_attr_count = sum(node.data[m] <= attr_mid)
            right_attr_count = sum(node.data[m] > attr_mid)

            left_attr_ratio = (left_attr_count / (left_attr_count + right_attr_count))
            right_attr_ratio = (right_attr_count / (left_attr_count + right_attr_count))
            
            attr_split_info = (- left_attr_ratio * math.log(left_attr_ratio, 2) - right_attr_ratio * math.log(right_attr_ratio, 2))
            # print 'attr:', m, 'attr_split_info:', attr_split_info
            
            m_left_attr_count = sum(node.data[node.data[m] <= attr_mid][self.class_column] == 'M')
            b_left_attr_count = sum(node.data[node.data[m] <= attr_mid][self.class_column] == 'B')
            m_right_attr_count = sum(node.data[node.data[m] > attr_mid][self.class_column] == 'M')
            b_right_attr_count = sum(node.data[node.data[m] > attr_mid][self.class_column] == 'B')
            # print m, left_attr_count, m_left_attr_count, b_left_attr_count, right_attr_count, m_right_attr_count, b_right_attr_count

            left_attr_calculate_info = self.__calculate_info(m_left_attr_count, b_left_attr_count)
            right_attr_calculate_info = self.__calculate_info(m_right_attr_count, b_right_attr_count)

            info_attr = left_attr_ratio * left_attr_calculate_info + right_attr_ratio * right_attr_calculate_info
            
            # print 'attr:', m, 'info_attr', info_attr

            info_gain_attr = base_info - info_attr
            # print 'attr:', m, 'info_gain', info_gain_attr
            
            gainratio_attr = info_gain_attr / attr_split_info
            # print 'attr:', m, 'gainratio_attr', gainratio_attr
            
            gainratio_attr_list.append(gainratio_attr)
        
        criterion_attr_index = gainratio_attr_list.index(max(gainratio_attr_list))
    
        criterion_tuple = (node.data.columns[criterion_attr_index], mid_attr_list[criterion_attr_index])
        return criterion_tuple

    def attribute_gini(self, node):
        gini_attr_list = []
        mid_attr_list = []
            
        for m in self.attr_list:
            attr_min = node.data[m].min()
            attr_max = node.data[m].max()
            attr_mid = (attr_max + attr_min) / 2
            # print m, attr_min, attr_max, attr_mid
            mid_attr_list.append(attr_mid)
            
            left_attr_count = sum(node.data[m] <= attr_mid)
            right_attr_count = sum(node.data[m] > attr_mid)
            
            m_left_attr_count = sum(node.data[node.data[m] <= attr_mid][self.class_column] == 'M')
            b_left_attr_count = sum(node.data[node.data[m] <= attr_mid][self.class_column] == 'B')
            m_right_attr_count = sum(node.data[node.data[m] > attr_mid][self.class_column] == 'M')
            b_right_attr_count = sum(node.data[node.data[m] > attr_mid][self.class_column] == 'B')
            # print m, left_attr_count, m_left_attr_count, b_left_attr_count, right_attr_count, m_right_attr_count, b_right_attr_count
                
            left_attr_gini = 1 - math.pow(m_left_attr_count / left_attr_count, 2) - math.pow(b_left_attr_count / left_attr_count , 2)
            right_attr_gini = 1 - math.pow( m_right_attr_count / right_attr_count, 2) - math.pow(b_right_attr_count / right_attr_count, 2)
            
            gini_attr = (left_attr_count / (left_attr_count + right_attr_count)) * left_attr_gini + (right_attr_count / (left_attr_count + right_attr_count)) * right_attr_gini
            
            # print m, gini_attr
            gini_attr_list.append(gini_attr)
        
        criterion_attr_index = gini_attr_list.index(min(gini_attr_list))
            
        criterion_tuple = (node.data.columns[criterion_attr_index], mid_attr_list[criterion_attr_index])
        # print criterion_tuple
        return criterion_tuple


    def __calculate_info(self, a, b):
        p_1 = a / (a + b)
        p_2 = b / (a + b)
        if p_1 == 1 or p_1 == 0:
            info = 0
        else:
            info = (- p_1 * math.log(p_1, 2) - p_2 * math.log(p_2, 2))
        return info

    def partitions(self, criterion_tuple, node):
        partition_attr = criterion_tuple[0]
        partition_point = criterion_tuple[1]
        self.attr_list.remove(partition_attr)
        left_data = node.data[node.data[partition_attr] <= partition_point][self.attr_list + ['class']]
        right_data = node.data[node.data[partition_attr] > partition_point][self.attr_list + ['class']]
        partition_data_tuple = (left_data, right_data)
        # print partition_data_tuple
        return partition_data_tuple
    
    def predict(self, row):
        if not isinstance(row, pd.Series):
            raise Exception
        
        build_tree = self.buildTree(self.data, self.attr_list, height = 0)
        node  = build_tree

        if self.max_height == 0:
            if node.successor != []:
                return node.majority
            else:
                return node.label
        else:
            if self.max_height > len(self.data.columns) - 1:
                self.max_height = len(self.data.columns) - 1
            height = 0
            while node.node_type != 'leaf' and height < self.max_height:
                if row[node.label[0]] > node.label[1]:
                    if node.successor[1].node_type != 'leaf':
                        node = node.successor[1]
                        if height + 1 == self.max_height:
                            return node.majority
                    else:
                       return node.successor[1].label
                else:
                    if node.successor[0].node_type != 'leaf':
                        node = node.successor[0]
                        if height + 1 == self.max_height:
                            return node.majority
                    else:
                        return node.successor[0].label
                height += 1


if __name__ == '__main__':
    class_column = 'class'
    
    df = analysis_data[['earea', 'esmooth', 'etexture', 'class']]
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

    average_result_infogain = []
    average_result_gainratio = []
    average_result_gini = []
    
    height_range = range(len(df.columns))

    for height in height_range:
        print 'height', height
        accuracy_list_infogain = []
        accuracy_list_gainratio = []
        accuracy_list_gini = []
        
        for test_data_index in fold_range_list:
            train_data_index = [x for x in df_index_range if x not in test_data_index]
            train_data = df.ix[train_data_index]
            train_data = train_data[['earea', 'esmooth', 'etexture', 'class']]
            test_data = df.ix[test_data_index]
            test_data_class = test_data['class']
            test_data = test_data[['earea', 'esmooth', 'etexture']]
        
            correct_num_infogain = 0
            correct_num_gainratio = 0
            correct_num_gini = 0
            all_num = len(test_data_index)
        
            for i in test_data_index:
                tree_infogain = DecisionTree(train_data, class_column, 'infogain', height)
                if tree_infogain.predict(test_data.ix[i]) == test_data_class[i]:
                    correct_num_infogain += 1
        
            for i in test_data_index:
                tree_gainratio = DecisionTree(train_data, class_column, 'gainratio', height)
                if tree_gainratio.predict(test_data.ix[i]) == test_data_class[i]:
                    correct_num_gainratio += 1
        
            for i in test_data_index:
                tree_gini = DecisionTree(train_data, class_column, 'gini', height)
                if tree_gini.predict(test_data.ix[i]) == test_data_class[i]:
                    correct_num_gini += 1

            accuracy_infogain = correct_num_infogain / all_num
            accuracy_gainratio = correct_num_gainratio / all_num
            accuracy_gini = correct_num_gini / all_num

#            print 'infogain accuracy', accuracy_infogain
#            print 'gainratio accuracy', accuracy_gainratio
#            print 'gini accuracy', accuracy_gini

            accuracy_list_infogain.append(accuracy_infogain)
            accuracy_list_gainratio.append(accuracy_gainratio)
            accuracy_list_gini.append(accuracy_gini)
        accuracy_series_infogain = Series(accuracy_list_infogain)
        accuracy_series_gainratio = Series(accuracy_list_gainratio)
        accuracy_series_gini = Series(accuracy_list_gini)

        print 'infogain average:', accuracy_series_infogain.mean()
        print 'gainratio average:', accuracy_series_gainratio.mean()
        print 'gini average:', accuracy_series_gini.mean()

        average_result_infogain.append(accuracy_series_infogain.mean())
        average_result_gainratio.append(accuracy_series_gainratio.mean())
        average_result_gini.append(accuracy_series_gini.mean())
    print average_result_infogain
    print average_result_gainratio
    print average_result_gini

    fig, axes = plt.subplots(nrows = 3, ncols = 1)

    fig.suptitle('Decision Tree Result')

    axes[0].set_title('info_gain result')
    axes[0].plot(height_range, average_result_infogain)
    axes[0].set_xlabel('Tree Height')
    axes[0].set_ylabel('Average Accuracy')
    axes[0].tick_params(axis='x', labelsize=6)
    axes[0].tick_params(axis='y', labelsize=6)

    axes[1].set_title('gain_ratio result')
    axes[1].plot(height_range, average_result_gainratio)
    axes[1].set_xlabel('Tree Height')
    axes[1].set_ylabel('Average Accuracy')
    axes[1].tick_params(axis='x', labelsize=6)
    axes[1].tick_params(axis='y', labelsize=6)


    axes[2].set_title('gini result')
    axes[2].plot(height_range, average_result_gini)
    axes[2].set_xlabel('Tree Height')
    axes[2].set_ylabel('Average Accuracy')
    axes[2].tick_params(axis='x', labelsize=6)
    axes[2].tick_params(axis='y', labelsize=6)
    
    plt.subplots_adjust(left = 0.18, right = 0.92, hspace = 1)
    plt.savefig('decision_tree_result_plot.pdf')












