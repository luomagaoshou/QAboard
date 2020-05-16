import numpy as np
import pandas as pd
import prettyprint
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
import math
from math import log
from enum import Enum


class DTreeType(Enum):
    ID3 = 1
    CART = 2
    C4_5 = 3


def create_data():
    '''
        xxx
    '''
    datasets = [['青年', '否', '否', '一般', '否'],
                ['青年', '否', '否', '好', '否'],
                ['青年', '是', '否', '好', '是'],
                ['青年', '是', '是', '一般', '是'],
                ['青年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '好', '否'],
                ['中年', '是', '是', '好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '好', '是'],
                ['老年', '是', '否', '好', '是'],
                ['老年', '是', '否', '非常好', '是'],
                ['老年', '否', '否', '一般', '否'],
                ]
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    # 返回数据集和每个维度的名称
    return datasets, labels


datasets, labels = create_data()
train_data = pd.DataFrame(datasets, columns=labels)


#
# # 计算信息熵
# # 根据当前数据集计算熵
# def calc_ent(datasets):
#     data_length = len(datasets)
#     label_count = {}
#     import math
#     for i in range(data_length):
#         label = datasets[i][-1]
#         # print(label)
#         if label not in label_count:
#             label_count[label] = 0
#         label_count[label] += 1
#     ent = -sum([(t / data_length) * math.log(t / data_length, 2) for t in label_count.values()])
#     return ent
#
#
# print(calc_ent(datasets))
#
#
# # 0.9709505944546686
# # 条件熵 id3的信息增益第二项
# # 条件熵  H(Y|X)用于找出熵最小  最稳定的特征作为第一个分类
# # sum((Di / D) * ent(Di))
# def cond_ent(datasets, axis=0):
#     data_length = len(datasets)
#     from collections import defaultdict
#
#     feature_sets = defaultdict(list)
#     for i in range(data_length):
#         feature = datasets[i][axis]
#         # 根据属性的不同进行分组
#         feature_sets[feature].append(datasets[i])
#
#     condition_ent = sum(
#         [len(sub_features) / data_length * calc_ent(sub_features) for sub_features in feature_sets.values()])
#     print(condition_ent)
#     return condition_ent
#
#
# # 总熵- 特征熵
# # 总熵不变，  特征熵越小越纯  所以求最大值
# def info_gain(ent, cond_ent):
#     return ent - cond_ent
#
#
# def info_gain_train(datasets):
#     # 最后一列是标签 不用比较
#     feature_count = len(datasets[0]) - 1
#     ent = calc_ent(datasets)
#     info_gains = []
#
#     for c in range(feature_count):
#         condition_ent = cond_ent(datasets, axis=c)
#         c_info_gain = info_gain(ent, condition_ent)
#         info_gains.append((c, c_info_gain))
#         print('特征: ({})  info_gain: {:.3f}'.format(labels[c], c_info_gain))
#     best_feature = max(info_gains, key=lambda x: x[-1])
#     return best_feature
#
#
# best_feature = info_gain_train(np.array(datasets))
# print("最优特征为：第%d列 %s %.3f" % (best_feature[0], labels[best_feature[0]], best_feature[1]))


# node->{value: node}
class Node:

    def __init__(self, is_leave=True, label=None, feature_name=None, feature_index=None):
        self.is_root = is_leave
        self.label = label
        self.feature_name = feature_name
        self.feature_index = feature_index
        # 用于放置子节点
        self.tree = {}
        self.result = {
            'label': self.label,
            'feature_index': self.feature_index,
            "tree": self.tree
        }

    def __repr__(self):
        return '{}'.format(self.result)

    # 添加节点
    def add_node(self, value, node):
        self.tree[value] = node

    def predict(self, features):
        if self.is_root is True:
            return self.label
        return self.tree[features[self.feature_index]].predict(features)


# tree -> {} ID3 使用信息增益
# C4.5  信息增益率
# CART 基尼系数
class DTree:
    def __init__(self, epsilon=0.1, tree_type=DTreeType.ID3):
        self.epsilon = epsilon
        self.tree = {}
        self.tree_type = tree_type



    # 计算熵
    def calc_ent(self, datasets, axis):

        data_length = len(datasets)
        label_counter = {}
        for i in range(data_length):
            # featues,...., label
            label = datasets[i][axis]
            if label not in label_counter:
                label_counter[label] = 0
            label_counter[label] += 1
        pis = [count / data_length for count in label_counter.values()]
        ent = -sum([pi * log(pi, 2) for pi in pis])
        return ent

    # ID3 -> 信息增益
    # C4.5 -> 信息增益率
    # CART -> 基尼系数
    def calc_cond_ent(self, datasets, axis):


        data_length = len(datasets)
        feature_counter = {}
        for i in range(data_length):
            feature = datasets[i][axis]
            if feature not in feature_counter:
                feature_counter[feature] = []
            feature_counter[feature].append(datasets[i])
        condition_ent = sum(
            [len(sub_features) / data_length * self.calc_ent(sub_features, -1) for sub_features in
             feature_counter.values()])

        return condition_ent





    def calc_info_gain(self, datasets, axis):
        ent = self.calc_ent(datasets, -1)
        condition_ent = self.calc_cond_ent(datasets, axis)
        gain = ent - condition_ent
        print(ent, condition_ent, gain)
        return gain

    def calc_intrinsic_value(self, datasets, axis):
        data_length = len(datasets)
        feature_counter = {}
        for i in range(data_length):
            feature = datasets[i][axis]
            if feature not in feature_counter:
                feature_counter[feature] = []
            feature_counter[feature].append(datasets[i])
        #     Dv / D
        Dv_Ds = [len(sub_features) / data_length for sub_features in feature_counter.values()]
        iv = -1 * sum([DV_D * log(DV_D, 2) for DV_D in Dv_Ds])
        return iv





    def info_gain_ratio(self, info_gain, intrinsic_value):
        ratio = info_gain / intrinsic_value
        return ratio

    # 基尼越小越集中
    def calc_gini(self, datasets, axis):
        data_length = len(datasets)
        feature_counter = {}
        for i in range(data_length):
            feature = datasets[i][axis]
            if feature not in feature_counter:
                feature_counter[feature] = []
            feature_counter[feature].append(datasets[i])
        pis = [(len(sub_features) / data_length) ** 2 for sub_features in feature_counter.values()]
        gini_value = 1 - sum(pis)
        return gini_value

    def calc_gini_index(self, datasets, axis):

        data_length = len(datasets)
        # gini_idnexes = []

        feature_counter = {}
        for i in range(data_length):
            feature = datasets[i][axis]
            if feature not in feature_counter:
                feature_counter[feature] = []
            feature_counter[feature].append(datasets[i])
    #     Dv / D
        Dv_Ds = [len(sub_features) / data_length for sub_features in feature_counter.values()]

        dv_ginis = [self.calc_gini(sub_features, -1) for sub_features in feature_counter.values()]
        gini_parts = [DV_D * gini_part for DV_D, gini_part in zip(Dv_Ds, dv_ginis)]
        gini_index = sum(gini_parts)
        return gini_index


    def gini_index_train(self, datasets):
        feature_count = len(datasets[0]) - 1

        gini_indexes = []
        for i in range(feature_count):
            gini_index = self.calc_gini_index(datasets, i)
            gini_indexes.append((i, gini_index))

        for i, gini_index in gini_indexes:
           print('特征: ({})  gini_index: {:.3f}'.format(labels[i], gini_index))

        index, best_gini_index= min(gini_indexes, key=lambda x: x[-1])
        return index, best_gini_index








    def info_gain_ratio_train(self, datasets):
        feature_count = len(datasets[0]) - 1

        info_gains = []
        for i in range(feature_count):
            info_gain_value = self.calc_info_gain(datasets, axis=i)
            info_gains.append((i, info_gain_value))
            print('特征: ({})  info_gain: {:.3f}'.format(labels[i], info_gain_value))

        intrinsic_values = []
        for i in range(feature_count):
            intrinsic_value = self.calc_intrinsic_value(datasets, i)
            intrinsic_values.append((i, intrinsic_value))
            print('特征: ({})  intrinsic_value: {:.3f}'.format(labels[i], intrinsic_value))


    #  熵不变的情况下   特征越集中信息增益越小 信息增益越大
    # 同时特征越集中，固有值越小 作为分母，结果越大
        gain_ratios = [(i, info_gain / intrinsic_value) for (i, info_gain), (j, intrinsic_value) in zip(info_gains, intrinsic_values)]
        for i, gain_ratio in gain_ratios:
            print('特征: ({})  gain_ratio: {:.3f}'.format(labels[i], gain_ratio))
        best_feature = max(gain_ratios, key=lambda x: x[-1])
        return best_feature




    def info_gain_train(self, datasets):
            feature_count = len(datasets[0]) - 1
            info_gains = []
            for i in range(feature_count):

                info_gain_value = self.calc_info_gain(datasets, axis=i)
                info_gains.append((i, info_gain_value))
                print('特征: ({})  info_gain: {:.3f}'.format(labels[i], info_gain_value))

            #  熵不变的情况下   特征越集中信息增益越小 信息增益越大
            best_feature = max(info_gains, key=lambda x: x[-1])
            return best_feature



    def train(self, train_data):
        # 除标签其他值， 所有标签，   特征名
        _, y_train, features = train_data.iloc[:, :-1], train_data.iloc[:, -1], train_data.columns[:-1]
        # 只有一个类别
        if len(y_train.value_counts()) == 1:
            return Node(is_leave=True, label=y_train.iloc[0])

        #         只有一个特征需要判断
        if len(features) == 0:
            return Node(

                is_leave=True,
                # 找出数量最多的为该节点子树
                label=y_train.value_counts().sort_values(ascending=False).index[0]

            )
        # 在给定数据中找到最大增益的特征
        best_feature_index, best_info_value, best_feature_name = None, None, None
        if self.tree_type == DTreeType.ID3:

            best_feature_index, best_info_value = self.info_gain_train(np.array(train_data))
            best_feature_name = features[best_feature_index]


        elif self.tree_type == DTreeType.C4_5:
            best_feature_index, best_info_value = self.info_gain_ratio_train(np.array(train_data))
            best_feature_name = features[best_feature_index]

        elif self.tree_type == DTreeType.CART:
            best_feature_index, best_info_value = self.gini_index_train(np.array(train_data))
            best_feature_name = features[best_feature_index]

        # 增益太小  即分类太混乱  达不到预期效果  不如不分
        if best_info_value < self.epsilon:
            return Node(
                is_leave=True,
                label=y_train.value_counts().sort_values(ascending=False).index[0]
            )



        # 正常情况建树 还有子节点的 没有label
        node_tree = Node(is_leave=False,
                         feature_name=best_feature_name, feature_index=best_feature_index)

        # 取出该分类下的特征
        features_list = train_data[best_feature_name].value_counts().index
        # 按特征值遍历 并以此为标准分类
        for f in features_list:
            # 取出当前特征最大增益的数据集
            #
            sub_train_df = train_data.loc[train_data[best_feature_name] == f].drop([best_feature_name], axis=1)
            # 训练当前特征值的数据集
            sub_tree = self.train(sub_train_df)
            node_tree.add_node(f, sub_tree)

        return node_tree

    def fit(self, train_data):
        self.tree = self.train(train_data)
        return self.tree

    def predict(self, X_test):
        return self.tree.predict(X_test)


print(DTree().info_gain_train(datasets))
# def train(self, train_data):

datasets, labels = create_data()
data_df = pd.DataFrame(datasets, columns=labels)

# decision_tree = DTree(tree_type=DTreeType.ID3)
# decision_tree = DTree(tree_type=DTreeType.C4_5)
decision_tree = DTree(tree_type=DTreeType.CART)

tree = decision_tree.fit(data_df)
print(tree)
# prettyprint(tree)

print(tree.predict(['老年', '否', '否', '一般']))
