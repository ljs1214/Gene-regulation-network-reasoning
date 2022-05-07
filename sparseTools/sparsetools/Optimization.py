################################算法函数区################################

import torch.nn.functional as F
import torch.nn as nn
import torch
import heapq
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
import math
import math
from tqdm import tqdm
import warnings
import pandas as pd


def take_AUC(network_predict_tf, network):
    a = network.reshape(len(network)*len(network[0]), 1).copy()
    b = network_predict_tf.reshape(
        len(network_predict_tf)*len(network_predict_tf[0]), 1).copy()
    a[a == 1] = 2
    a[a == 0] = 1
    b[b == 1] = 2
    b[b == 0] = 1
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve, auc
    # print(a)
    # print(b)
    auc_score4 = roc_auc_score(a, b)
    fpr3, tpr3, thersholds3 = roc_curve(a, b, pos_label=2)
    print(auc_score4)
    return auc_score4


# ista实现代码


def shrinkage(x, theta):
    return np.multiply(np.sign(x), np.maximum(np.abs(x) - theta, 0))


def ista(X, W_d, a, L, max_iter, eps):
    # print(X.shape)
    # print(W_d.shape)
    eig, eig_vector = np.linalg.eig(W_d.T * W_d)
    L = np.max(eig)
    del eig, eig_vector

    W_e = W_d.T / L
    t = 1
    x0 = 0
    recon_errors = []
    # print(W_d.shape[1])
    Z_old = np.zeros((W_d.shape[1], 1))
    # print(Z_old.shape)
    for i in range(max_iter):
        temp = W_d * Z_old - X
        #print(W_e * temp)
        Z_new = shrinkage(Z_old - W_e * temp, a / L)

        if np.sum(np.abs(Z_new - Z_old)) <= eps:
            break
        Z_old = Z_new
        recon_error = np.linalg.norm(X - W_d * Z_new, 2) ** 2
        recon_errors.append(recon_error)
    return Z_new, recon_errors


def cs_CoSaMP(y, D, K):
    S = K  # 稀疏度
    residual = y  # 初始化残差
    pos_last = np.array([], dtype=np.int64)
    result = np.zeros((D.shape[1]))
    for j in range(S):  # 迭代次数
        product = np.fabs(np.dot(D.T, residual))
        pos_temp = np.argsort(product)
        pos_temp = pos_temp[::-1]  # 反向，得到前面L个大的位置
        pos_temp = pos_temp[0:2*S]  # 对应步骤3
        pos = np.union1d(pos_temp, pos_last)

        result_temp = np.zeros(((D.shape[1])))
        result_temp[pos] = np.dot(np.linalg.pinv(D[:, pos]), y)

        pos_temp = np.argsort(np.fabs(result_temp))
        pos_temp = pos_temp[::-1]  # 反向，得到前面L个大的位置
        result[pos_temp[:S]] = result_temp[pos_temp[:S]]
        pos_last = pos_temp
        residual = y-np.dot(D, result)

    return result


def cs_IHT(y, D, K):
    K = K  # 稀疏度
    result_temp = np.zeros((D.shape[1]))  # 初始化重建信号
    u = 1  # 影响因子
    result = result_temp
    for j in range(K):  # 迭代次数
        x_increase = np.dot(D.T, (y-np.dot(D, result_temp)))  # x=D*(y-D*y0)
        result = result_temp+np.dot(x_increase, u)  # x(t+1)=x(t)+D*(y-D*y0)
        temp = np.fabs(result)
        pos = temp.argsort()
        pos = pos[::-1]  # 反向，得到前面L个大的位置
        result[pos[K:]] = 0
        result_temp = result
    return result


def cs_sp(y, D, K):
    K = K
    pos_last = np.array([], dtype=np.int64)
    result = np.zeros((D.shape[1]))

    product = np.fabs(np.dot(D.T, y))
    pos_temp = product.argsort()
    pos_temp = pos_temp[::-1]  # 反向，得到前面L个大的位置
    pos_current = pos_temp[0:K]  # 初始化索引集 对应初始化步骤1
    residual_current = y - \
        np.dot(D[:, pos_current], np.dot(np.linalg.pinv(
            D[:, pos_current]), y))  # 初始化残差 对应初始化步骤2

    while True:  # 迭代次数
        product = np.fabs(np.dot(D.T, residual_current))
        pos_temp = np.argsort(product)
        pos_temp = pos_temp[::-1]  # 反向，得到前面L个大的位置
        pos = np.union1d(pos_current, pos_temp[0:K])  # 对应步骤1
        pos_temp = np.argsort(
            np.fabs(np.dot(np.linalg.pinv(D[:, pos]), y)))  # 对应步骤2
        pos_temp = pos_temp[::-1]
        pos_last = pos_temp[0:K]  # 对应步骤3
        residual_last = y - \
            np.dot(D[:, pos_last], np.dot(
                np.linalg.pinv(D[:, pos_last]), y))  # 更新残差 #对应步骤4
        if np.linalg.norm(residual_last) >= np.linalg.norm(residual_current):  # 对应步骤5
            pos_last = pos_current
            break
        residual_current = residual_last
        pos_current = pos_last
    result[pos_last[0:K]] = np.dot(
        np.linalg.pinv(D[:, pos_last[0:K]]), y)  # 对应输出步骤
    return result


def cs_omp(y, D):
    L = math.floor(3*(y.shape[0])/4)
    residual = y  # 初始化残差
    index = np.zeros((L), dtype=int)
    for i in range(L):
        index[i] = -1
    result = np.zeros((D.shape[1]))
    for j in range(L):  # 迭代次数
        product = np.fabs(np.dot(D.T, residual))
        pos = np.argmax(product)  # 最大投影系数对应的位置
        index[j] = pos
        my = np.linalg.pinv(D[:])  # 最小二乘,看参考文献1
        a = np.dot(my, y)  # 最小二乘,看参考文献1
        residual = y-np.dot(D[:, index >= 0], a)
    result[index >= 0] = a
    return result

    ################################投票算法################################


def voting(expre, HGS, tf_names, gene_names):
    new_expre = expre.copy()
    tf_names_list = list(tf_names)
    name_list = list(gene_names["#ID"])
    warnings.filterwarnings("ignore")
    network_predict = []
    temp_list = np.zeros((len(name_list), len(tf_names_list)))
    predict_tf = np.zeros((len(name_list), len(tf_names_list)))
    temp_list1 = np.zeros((len(name_list), len(tf_names_list)))
    temp_list2 = np.zeros((len(name_list), len(tf_names_list)))
    temp_list3 = np.zeros((len(name_list), len(tf_names_list)))
    temp_list4 = np.zeros((len(name_list), len(tf_names_list)))
    temp_list4_real = np.zeros((len(name_list), len(tf_names_list)))
    temp_list5 = np.zeros((len(name_list), len(tf_names_list)))
    tf_exp = []
    for i in range(len(tf_names)):

        tf_exp.append(new_expre[tf_names.iloc[i].at[0]])
    tf_exp = pd.DataFrame(tf_exp)
    tf_exp = tf_exp.T

    for i in tqdm(range(len(new_expre.iloc[0]))):
        flag = False
        y = new_expre[gene_names["#ID"]].T.iloc[i]
        copy_new_expre = tf_exp.copy()

        try:
            del copy_new_expre[gene_names["#ID"][i]]
            # print(gene_names["#ID"][i])
            temp_index = tf_names_list.index(gene_names["#ID"][i])
            flag = True
            # print("T")  # 处理tf作为被调控因子的情况
        except:
            pass

        ###SP###

        predict_list = []
        # print(y.values.shape)
        # print(copy_new_expre.values.shape)
        par_lists = cs_sp(y.values, copy_new_expre.values,
                          int(len(tf_names_list)*0.35))
        if flag:
            # print(len(par_lists))
            par_lists = np.insert(par_lists, temp_index, 0)
            # print(len(par_lists))
        for j in range(len(temp_list[i])):
            temp_list[i][j] = par_lists[j]
        # print(par_lists)
        # network_predict.append(par_lists)
        #temp_index = list(gene_names[0]).index(tf_names[0][i])
        # par_lists.insert(temp_index,0)
        #network_predict_tf[temp_index] = par_lists
        # print((i/len(new_expre.iloc[0])*100),"%")

        ###IHT###
        par_lists3 = cs_IHT(y.values, copy_new_expre.values,
                            int(len(tf_names_list)*0.35))
        if flag:
            # print(len(par_lists))
            par_lists3 = np.insert(par_lists3, temp_index, 0)
            # print(len(par_lists))
        for j in range(len(temp_list3[i])):
            temp_list3[i][j] = par_lists3[j]

        ###CosaOMP###
        par_lists5 = cs_CoSaMP(
            y.values, copy_new_expre.values, int(len(tf_names_list)*0.35))
        if flag:
            # print(len(par_lists))
            par_lists5 = np.insert(par_lists5, temp_index, 0)
            # print(len(par_lists))
        for j in range(len(temp_list5[i])):
            temp_list5[i][j] = par_lists5[j]

        ###Lasso###
        model = LassoLarsCV()
        #model = OrthogonalMatchingPursuitCV()
        model.fit(copy_new_expre, y)
        predict_list1 = []
        par_lists1 = model.coef_
        if flag:
            # print(len(par_lists1))
            par_lists1 = np.insert(par_lists1, temp_index, 0)
            # print(len(par_lists))
        for j in range(len(temp_list1[i])):
            temp_list1[i][j] = par_lists1[j]

        ###OMP###
        model = OrthogonalMatchingPursuitCV()
        model.fit(copy_new_expre, y)
        predict_list2 = []
        par_lists2 = model.coef_
        if flag:
            # print(len(par_lists1))
            par_lists2 = np.insert(par_lists2, temp_index, 0)
            # print(len(par_lists))
        for j in range(len(temp_list2[i])):
            temp_list2[i][j] = par_lists2[j]

        ###ista###
        X = y
        W_d = copy_new_expre
        # print(X)
        # print(W_d)
        # print(X,W_d)
        Z_recon, recon_errors = ista(np.mat(X), np.mat(
            W_d.values), 0.35, 83000, 20, 0.00001)
        # print(recon_errors)
        par_lists4 = sum(Z_recon.T.A)
        if flag:
            # print(len(par_lists))
            par_lists4 = np.insert(par_lists4, temp_index, 0)
            # print(len(par_lists))
        for j in range(len(temp_list4[i])):
            temp_list4[i][j] = par_lists4[j]

        temp_list4_real = np.zeros((len(name_list), len(tf_names_list)))
        for i in range(len(temp_list4)):
            for j in range(len(temp_list4[i])):
                temp_list4[i][j] = abs(temp_list4[i][j])

        for i in range(len(temp_list4)):
            for k in temp_list4[i].argsort()[-120:][::-1]:
                temp_list4_real[i][k] = 1

        pd.DataFrame(temp_list).to_csv("SP.csv")
        pd.DataFrame(temp_list1).to_csv("IHT.csv")
        pd.DataFrame(temp_list2).to_csv("CosaMP.csv")
        pd.DataFrame(temp_list3).to_csv("Lasso.csv")
        pd.DataFrame(temp_list4_real).to_csv("OMP.csv")
        pd.DataFrame(temp_list5).to_csv("ISTA.csv")

        ans = np.zeros((len(name_list), len(tf_names_list)))
    for i in range(len(temp_list)):
        for j in range(len(temp_list[i])):
            k = 0
            if temp_list[i][j] != 0:
                k += 1
                temp_list[i][j] = 1
            if temp_list1[i][j] != 0:
                k += 1
                temp_list1[i][j] = 1
            if temp_list2[i][j] != 0:
                k += 1
                temp_list2[i][j] = 1

            if temp_list3[i][j] != 0:
                temp_list3[i][j] = 1
            else:
                temp_list3[i][j] = 1
                k += 1

            if temp_list4_real[i][j] != 0:
                k += 1

            if temp_list5[i][j] != 0:
                k += 1
                temp_list5[i][j] = 1
            if k >= 4:
                ans[i][j] = 1
    return ans
