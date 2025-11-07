import os
import glob
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import torchnet as tnt
from skimage import io
import cv2

def bi_matrix(bi_pred, bi_target):

    test_change_num = len(bi_target[bi_target == 1])
    test_unchange_num = len(bi_target[bi_target == 0])
    test_num = test_change_num + test_unchange_num

    TP = np.sum((np.multiply(bi_pred, bi_target)) == 1)
    TN = np.sum((np.multiply((1 - bi_pred), (1 - bi_target))) == 1)
    FN = test_change_num - TP
    FP = test_unchange_num - TN
    Matrix_Bi = [[TP, FP], [FN, TN]]

    return Matrix_Bi

def bi_accuracy(matrix):
    TP = matrix[0, 0]
    FP = matrix[0, 1]
    FN = matrix[1, 0]
    TN = matrix[1, 1]
    Matrix = [[TP, FP], [FN, TN]]
    test_num = TP + FP + FN + TN

    overall_accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1_score = 2 * precision * recall / (precision + recall)

    p1 = int(int(int(TP + FP) * int(TP + FN)) + int(int(FN + TN) * int(FP + TN)))
    Pe = float(p1 / test_num / test_num)
    kappa = (overall_accuracy - Pe) / (1 - Pe)

    Iou = TP / (FN + FP + TP)
    MIoU = (TP / (FN + FP + TP) + TN / (TN + FN + FP)) / 2

    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)
    BA = (TPR + TNR) / 2

    return  overall_accuracy, precision, recall, F1_score, kappa, Iou, MIoU, FPR, FNR

def conf_m(output, target_th):
    #作用是将数据转换为以为的向量
    # print('\noutput', output.shape)
    # print('target_th', target_th.shape)
    output_conf=output.data
    # print('output_conf',output_conf.shape)
    output_conf=(output_conf.contiguous()).view(output_conf.size(0)*output_conf.size(1)*output_conf.size(2))
    # print('output_conf',output_conf.shape)
    target_conf=target_th.data
    # print('target_conf',target_conf.shape)
    target_conf=(target_conf.contiguous()).view(target_conf.size(0)*target_conf.size(1)*target_conf.size(2))
    # print('target_conf',target_conf.shape)
    return output_conf, target_conf

def multi_matrix(multi_pred, multi_target):
    Maxtrix_Multi = np.zeros((9, 9))
    Maxtrix_Multi[0, 0] = np.sum((multi_target == 0) & (multi_pred == 0))
    Maxtrix_Multi[1, 0] = np.sum((multi_target == 1) & (multi_pred == 0))
    Maxtrix_Multi[2, 0] = np.sum((multi_target == 2) & (multi_pred == 0))
    Maxtrix_Multi[3, 0] = np.sum((multi_target == 3) & (multi_pred == 0))
    Maxtrix_Multi[4, 0] = np.sum((multi_target == 4) & (multi_pred == 0))
    Maxtrix_Multi[5, 0] = np.sum((multi_target == 5) & (multi_pred == 0))
    Maxtrix_Multi[6, 0] = np.sum((multi_target == 6) & (multi_pred == 0))
    Maxtrix_Multi[7, 0] = np.sum((multi_target == 7) & (multi_pred == 0))
    Maxtrix_Multi[8, 0] = np.sum((multi_target == 8) & (multi_pred == 0))

    Maxtrix_Multi[0, 1] = np.sum((multi_target == 0) & (multi_pred == 1))
    Maxtrix_Multi[1, 1] = np.sum((multi_target == 1) & (multi_pred == 1))
    Maxtrix_Multi[2, 1] = np.sum((multi_target == 2) & (multi_pred == 1))
    Maxtrix_Multi[3, 1] = np.sum((multi_target == 3) & (multi_pred == 1))
    Maxtrix_Multi[4, 1] = np.sum((multi_target == 4) & (multi_pred == 1))
    Maxtrix_Multi[5, 1] = np.sum((multi_target == 5) & (multi_pred == 1))
    Maxtrix_Multi[6, 1] = np.sum((multi_target == 6) & (multi_pred == 1))
    Maxtrix_Multi[7, 1] = np.sum((multi_target == 7) & (multi_pred == 1))
    Maxtrix_Multi[8, 1] = np.sum((multi_target == 8) & (multi_pred == 1))

    Maxtrix_Multi[0, 2] = np.sum((multi_target == 0) & (multi_pred == 2))
    Maxtrix_Multi[1, 2] = np.sum((multi_target == 1) & (multi_pred == 2))
    Maxtrix_Multi[2, 2] = np.sum((multi_target == 2) & (multi_pred == 2))
    Maxtrix_Multi[3, 2] = np.sum((multi_target == 3) & (multi_pred == 2))
    Maxtrix_Multi[4, 2] = np.sum((multi_target == 4) & (multi_pred == 2))
    Maxtrix_Multi[5, 2] = np.sum((multi_target == 5) & (multi_pred == 2))
    Maxtrix_Multi[6, 2] = np.sum((multi_target == 6) & (multi_pred == 2))
    Maxtrix_Multi[7, 2] = np.sum((multi_target == 7) & (multi_pred == 2))
    Maxtrix_Multi[8, 2] = np.sum((multi_target == 8) & (multi_pred == 2))

    Maxtrix_Multi[0, 3] = np.sum((multi_target == 0) & (multi_pred == 3))
    Maxtrix_Multi[1, 3] = np.sum((multi_target == 1) & (multi_pred == 3))
    Maxtrix_Multi[2, 3] = np.sum((multi_target == 2) & (multi_pred == 3))
    Maxtrix_Multi[3, 3] = np.sum((multi_target == 3) & (multi_pred == 3))
    Maxtrix_Multi[4, 3] = np.sum((multi_target == 4) & (multi_pred == 3))
    Maxtrix_Multi[5, 3] = np.sum((multi_target == 5) & (multi_pred == 3))
    Maxtrix_Multi[6, 3] = np.sum((multi_target == 6) & (multi_pred == 3))
    Maxtrix_Multi[7, 3] = np.sum((multi_target == 7) & (multi_pred == 3))
    Maxtrix_Multi[8, 3] = np.sum((multi_target == 8) & (multi_pred == 3))

    Maxtrix_Multi[0, 4] = np.sum((multi_target == 0) & (multi_pred == 4))
    Maxtrix_Multi[1, 4] = np.sum((multi_target == 1) & (multi_pred == 4))
    Maxtrix_Multi[2, 4] = np.sum((multi_target == 2) & (multi_pred == 4))
    Maxtrix_Multi[3, 4] = np.sum((multi_target == 3) & (multi_pred == 4))
    Maxtrix_Multi[4, 4] = np.sum((multi_target == 4) & (multi_pred == 4))
    Maxtrix_Multi[5, 4] = np.sum((multi_target == 5) & (multi_pred == 4))
    Maxtrix_Multi[6, 4] = np.sum((multi_target == 6) & (multi_pred == 4))
    Maxtrix_Multi[7, 4] = np.sum((multi_target == 7) & (multi_pred == 4))
    Maxtrix_Multi[8, 4] = np.sum((multi_target == 8) & (multi_pred == 4))

    Maxtrix_Multi[0, 5] = np.sum((multi_target == 0) & (multi_pred == 5))
    Maxtrix_Multi[1, 5] = np.sum((multi_target == 1) & (multi_pred == 5))
    Maxtrix_Multi[2, 5] = np.sum((multi_target == 2) & (multi_pred == 5))
    Maxtrix_Multi[3, 5] = np.sum((multi_target == 3) & (multi_pred == 5))
    Maxtrix_Multi[4, 5] = np.sum((multi_target == 4) & (multi_pred == 5))
    Maxtrix_Multi[5, 5] = np.sum((multi_target == 5) & (multi_pred == 5))
    Maxtrix_Multi[6, 5] = np.sum((multi_target == 6) & (multi_pred == 5))
    Maxtrix_Multi[7, 5] = np.sum((multi_target == 7) & (multi_pred == 5))
    Maxtrix_Multi[8, 5] = np.sum((multi_target == 8) & (multi_pred == 5))

    Maxtrix_Multi[0, 6] = np.sum((multi_target == 0) & (multi_pred == 6))
    Maxtrix_Multi[1, 6] = np.sum((multi_target == 1) & (multi_pred == 6))
    Maxtrix_Multi[2, 6] = np.sum((multi_target == 2) & (multi_pred == 6))
    Maxtrix_Multi[3, 6] = np.sum((multi_target == 3) & (multi_pred == 6))
    Maxtrix_Multi[4, 6] = np.sum((multi_target == 4) & (multi_pred == 6))
    Maxtrix_Multi[5, 6] = np.sum((multi_target == 5) & (multi_pred == 6))
    Maxtrix_Multi[6, 6] = np.sum((multi_target == 6) & (multi_pred == 6))
    Maxtrix_Multi[7, 6] = np.sum((multi_target == 7) & (multi_pred == 6))
    Maxtrix_Multi[8, 6] = np.sum((multi_target == 8) & (multi_pred == 6))

    Maxtrix_Multi[0, 7] = np.sum((multi_target == 0) & (multi_pred == 7))
    Maxtrix_Multi[1, 7] = np.sum((multi_target == 1) & (multi_pred == 7))
    Maxtrix_Multi[2, 7] = np.sum((multi_target == 2) & (multi_pred == 7))
    Maxtrix_Multi[3, 7] = np.sum((multi_target == 3) & (multi_pred == 7))
    Maxtrix_Multi[4, 7] = np.sum((multi_target == 4) & (multi_pred == 7))
    Maxtrix_Multi[5, 7] = np.sum((multi_target == 5) & (multi_pred == 7))
    Maxtrix_Multi[6, 7] = np.sum((multi_target == 6) & (multi_pred == 7))
    Maxtrix_Multi[7, 7] = np.sum((multi_target == 7) & (multi_pred == 7))
    Maxtrix_Multi[8, 7] = np.sum((multi_target == 8) & (multi_pred == 7))

    Maxtrix_Multi[0, 8] = np.sum((multi_target == 0) & (multi_pred == 8))
    Maxtrix_Multi[1, 8] = np.sum((multi_target == 1) & (multi_pred == 8))
    Maxtrix_Multi[2, 8] = np.sum((multi_target == 2) & (multi_pred == 8))
    Maxtrix_Multi[3, 8] = np.sum((multi_target == 3) & (multi_pred == 8))
    Maxtrix_Multi[4, 8] = np.sum((multi_target == 4) & (multi_pred == 8))
    Maxtrix_Multi[5, 8] = np.sum((multi_target == 5) & (multi_pred == 8))
    Maxtrix_Multi[6, 8] = np.sum((multi_target == 6) & (multi_pred == 8))
    Maxtrix_Multi[7, 8] = np.sum((multi_target == 7) & (multi_pred == 8))
    Maxtrix_Multi[8, 8] = np.sum((multi_target == 8) & (multi_pred == 8))

    return Maxtrix_Multi

def multi_accuracy(matrix):
    recall_0 = matrix[0, 0] / matrix[0].sum()
    recall_1 = matrix[1, 1] / matrix[1].sum()
    recall_2 = matrix[2, 2] / matrix[2].sum()
    recall_3 = matrix[3, 3] / matrix[3].sum()
    recall_4 = matrix[4, 4] / matrix[4].sum()
    recall_5 = matrix[5, 5] / matrix[5].sum()
    recall_6 = matrix[6, 6] / matrix[6].sum()
    recall_7 = matrix[7, 7] / matrix[7].sum()
    recall_8 = matrix[8, 8] / matrix[8].sum()

    precision_0 = matrix[0, 0] / matrix[:, 0].sum()
    precision_1 = matrix[1, 1] / matrix[:, 1].sum()
    precision_2 = matrix[2, 2] / matrix[:, 2].sum()
    precision_3 = matrix[3, 3] / matrix[:, 3].sum()
    precision_4 = matrix[4, 4] / matrix[:, 4].sum()
    precision_5 = matrix[5, 5] / matrix[:, 5].sum()
    precision_6 = matrix[6, 6] / matrix[:, 6].sum()
    precision_7 = matrix[7, 7] / matrix[:, 7].sum()
    precision_8 = matrix[8, 8] / matrix[:, 8].sum()

    FP_0 = matrix[:, 0].sum() - matrix[0, 0]
    FP_1 = matrix[:, 1].sum() - matrix[1, 1]
    FP_2 = matrix[:, 2].sum() - matrix[2, 2]
    FP_3 = matrix[:, 3].sum() - matrix[3, 3]
    FP_4 = matrix[:, 4].sum() - matrix[4, 4]
    FP_5 = matrix[:, 5].sum() - matrix[5, 5]
    FP_6 = matrix[:, 6].sum() - matrix[6, 6]
    FP_7 = matrix[:, 7].sum() - matrix[7, 7]
    FP_8 = matrix[:, 8].sum() - matrix[8, 8]

    TN_0 = matrix.sum() - matrix[:, 0] - matrix[0, :] + matrix[0, 0]
    TN_1 = matrix.sum() - matrix[:, 1] - matrix[1, :] + matrix[1, 1]
    TN_2 = matrix.sum() - matrix[:, 2] - matrix[2, :] + matrix[2, 2]
    TN_3 = matrix.sum() - matrix[:, 3] - matrix[3, :] + matrix[3, 3]
    TN_4 = matrix.sum() - matrix[:, 4] - matrix[4, :] + matrix[4, 4]
    TN_5 = matrix.sum() - matrix[:, 5] - matrix[5, :] + matrix[5, 5]
    TN_6 = matrix.sum() - matrix[:, 6] - matrix[6, :] + matrix[6, 6]
    TN_7 = matrix.sum() - matrix[:, 7] - matrix[7, :] + matrix[7, 7]
    TN_8 = matrix.sum() - matrix[:, 8] - matrix[8, :] + matrix[8, 8]

    FPR_0 = FP_0 / (FP_0 + TN_0)
    FPR_1 = FP_1 / (FP_1 + TN_1)
    FPR_2 = FP_2 / (FP_2 + TN_2)
    FPR_3 = FP_3 / (FP_3 + TN_3)
    FPR_4 = FP_4 / (FP_4 + TN_4)
    FPR_5 = FP_5 / (FP_5 + TN_5)
    FPR_6 = FP_6 / (FP_6 + TN_6)
    FPR_7 = FP_7 / (FP_7 + TN_7)
    FPR_8 = FP_8 / (FP_8 + TN_8)

    FNR_0 = 1 - recall_0
    FNR_1 = 1 - recall_1
    FNR_2 = 1 - recall_2
    FNR_3 = 1 - recall_3
    FNR_4 = 1 - recall_4
    FNR_5 = 1 - recall_5
    FNR_6 = 1 - recall_6
    FNR_7 = 1 - recall_7
    FNR_8 = 1 - recall_8

    precision = [precision_0, precision_1, precision_2, precision_3, precision_4, precision_5, precision_6, precision_7, precision_8]
    recall = [recall_0, recall_1, recall_2, recall_3, recall_4, recall_5, recall_6, recall_7, recall_8]
    FPR = [FPR_0, FPR_1, FPR_2, FPR_3, FPR_4, FPR_5, FPR_6, FPR_7, FPR_8]
    FNR = [FNR_0, FNR_1, FNR_2, FNR_3, FNR_4, FNR_5, FNR_6, FNR_7, FNR_8]
    precision = np.nan_to_num(precision, nan=0)
    recall = np.nan_to_num(recall, nan=0)
    FPR = np.nan_to_num(FPR, nan=0)
    FNR = np.nan_to_num(FNR, nan=0)
    F1_score = 2 * precision * recall / (precision + recall)
    F1_score = np.nan_to_num(F1_score, nan=0)

    overall_accuracy = (matrix[0, 0] + matrix[1, 1] + matrix[2, 2] + matrix[3, 3] + matrix[4, 4] + matrix[5, 5] + matrix[6, 6] + matrix[7, 7] + matrix[8, 8]) / matrix.sum()
    precision_average = precision.mean()
    recall_average = recall.mean()
    F1_score_average = F1_score.mean()
    FPR_average = FPR.mean()
    FNR_average = FNR.mean()

    pe = ((matrix[0].sum() * matrix[:, 0].sum()) + (matrix[1].sum() * matrix[:, 1].sum()) +  (matrix[2].sum() * matrix[:, 2].sum()) +
          (matrix[3].sum() * matrix[:, 3].sum()) + (matrix[4].sum() * matrix[:, 4].sum()) + (matrix[5].sum() * matrix[:, 5].sum()) +
          (matrix[6].sum() * matrix[:, 6].sum()) + (matrix[7].sum() * matrix[:, 7].sum()) + (matrix[8].sum() * matrix[:, 8].sum())) / \
         (matrix.sum() * matrix.sum())
    kappa = (overall_accuracy - pe) / (1 - pe)

    IoU_0 = matrix[0, 0] / (matrix[:, 0].sum() + matrix[0, :].sum() - matrix[0, 0])
    IoU_1 = matrix[1, 1] / (matrix[:, 1].sum() + matrix[1, :].sum() - matrix[1, 1])
    IoU_2 = matrix[2, 2] / (matrix[:, 2].sum() + matrix[2, :].sum() - matrix[2, 2])
    IoU_3 = matrix[3, 3] / (matrix[:, 3].sum() + matrix[3, :].sum() - matrix[3, 3])
    IoU_4 = matrix[4, 4] / (matrix[:, 4].sum() + matrix[4, :].sum() - matrix[4, 4])
    IoU_5 = matrix[5, 5] / (matrix[:, 5].sum() + matrix[5, :].sum() - matrix[5, 5])
    IoU_6 = matrix[6, 6] / (matrix[:, 6].sum() + matrix[6, :].sum() - matrix[6, 6])
    IoU_7 = matrix[7, 7] / (matrix[:, 7].sum() + matrix[7, :].sum() - matrix[7, 7])
    IoU_8 = matrix[8, 8] / (matrix[:, 8].sum() + matrix[8, :].sum() - matrix[8, 8])
    IoU = [IoU_1, IoU_2, IoU_3, IoU_4, IoU_5, IoU_6, IoU_7, IoU_8]
    MIoU = [IoU_0, IoU_1, IoU_2, IoU_3, IoU_4, IoU_5, IoU_6, IoU_7, IoU_8]
    IoU = np.nan_to_num(IoU, nan=0)
    MIoU = np.nan_to_num(MIoU, nan=0)
    IoU_average = np.mean(IoU)
    MIoU_average = np.mean(MIoU)

    return overall_accuracy, precision_average, recall_average, F1_score_average, kappa, IoU_average, MIoU_average, FPR_average, FNR_average






