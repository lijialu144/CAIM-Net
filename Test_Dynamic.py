import os                                                               #处理文件和目录
import os.path as osp                                                   #对应加载模块的别名
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import time
import matplotlib.pyplot as plt
import yaml
import pickle
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
import torch.nn.functional as F
import argparse
from sklearn import metrics
import random
from data_processing import Dynamic_custom
from torch.utils.data import DataLoader
from model import Dynamic_CAIMNet
from skimage import io
from datetime import datetime
import pandas as pd
import cv2
import get_matrix
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
parser = argparse.ArgumentParser()
parser.add_argument('--max_epoch', type=int, default=1, help='max epoch]')
parser.add_argument('--change_num1', type=int, default=2, help='number of change')
parser.add_argument('--TSIs_len', type=int, default=6, help='image number')
parser.add_argument('--Fsplit', type=str, default='Dynamic_Fsplit/', help='Fsplit folder')
parser.add_argument('--xys', type=str, default='Dynamic_xys_64_64/', help='xys folder')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--patch_size', type=int, default=64, help='patch_size')
parser.add_argument('--save_path', default='best_models', help='model save path')
args = parser.parse_args()
class Trainer(object):

    def __init__(self, cfig):
        self.model = Dynamic_CAIMNet.Model(batch=args.batch_size, input_channels=4, features=64,
                     change_nums=args.TSIs_len, times=args.TSIs_len, size=args.patch_size).cuda()

    # def adjust_learning_rate(self, optimizer, epoch, steps, gamma):
    #     if epoch == 0:
    #         self.lr = 0.001
    #     if epoch in steps:
    #         self.lr *= gamma
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] = self.lr
    #             print('After modify, the learning rate is', param_group['lr'])

    def change_map(self):
        print('Testing..................')
        best_model_pth = args.save_path+'/CAIMNet-37.67.pth'
        self.model.load_state_dict(torch.load(best_model_pth))
        Ftest = np.load(args.Fsplit + 'Ftest.npy').tolist()
        csv_test = args.xys + 'myxys_test.csv'

        test_dataset = Dynamic_custom.MyDataset(csv_test, Ftest, args.patch_size)
        test_sample = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
        Test_Bi_Pred_list = []
        Test_Multi_Pred_list = []

        for iter_train, batch in enumerate(test_sample):
            test_batch_image, test_batch_bilabel, test_batch_multilabel = batch
            test_batch_image = test_batch_image.permute(1, 0, 4, 2, 3)

            test_batch_image = torch.as_tensor(test_batch_image, dtype=torch.float32).cuda()
            test_batch_bilabel = torch.as_tensor(test_batch_bilabel, dtype=torch.long).cuda()
            test_batch_multilabel = torch.as_tensor(test_batch_multilabel, dtype=torch.long).cuda()

            pred_bi_list, pred_multi_list = self.change_map_epoch(test_batch_image)
            Test_Bi_Pred_list.extend(pred_bi_list)
            Test_Multi_Pred_list.extend(pred_multi_list)
        test_bi_pred_conf = np.asarray(Test_Bi_Pred_list)
        test_multi_pred_conf = np.asarray(Test_Multi_Pred_list)
        test_bi_pred_conf1 = test_bi_pred_conf[0:256]
        test_bi_pred_conf2 = test_bi_pred_conf[256:256*2]
        test_bi_pred_conf3 = test_bi_pred_conf[256*2:]
        test_multi_pred_conf1 = test_multi_pred_conf[0:256]
        test_multi_pred_conf2 = test_multi_pred_conf[256:256*2]
        test_multi_pred_conf3 = test_multi_pred_conf[256 * 2:]

        pred_bi1 = np.random.rand(1024, 1024)
        pred_bi2 = np.random.rand(1024, 1024)
        pred_bi3 = np.random.rand(1024, 1024)
        pred_multi1 = np.random.rand(1024, 1024)
        pred_multi2 = np.random.rand(1024, 1024)
        pred_multi3 = np.random.rand(1024, 1024)

        for i in range(16):
            for j in range(16):
                # 计算当前块在96x64x64矩阵中的索引
                index = (i * 16 // 16) * 16 + (j * 16 // 16)
                pred_bi1[i * 64: (i + 1) * 64, j * 64:(j + 1) * 64] = test_bi_pred_conf1[index]
                pred_bi2[i * 64: (i + 1) * 64, j * 64:(j + 1) * 64] = test_bi_pred_conf2[index]
                pred_bi3[i * 64: (i + 1) * 64, j * 64:(j + 1) * 64] = test_bi_pred_conf3[index]
                pred_multi1[i * 64: (i + 1) * 64, j * 64:(j + 1) * 64] = test_multi_pred_conf1[index]
                pred_multi2[i * 64: (i + 1) * 64, j * 64:(j + 1) * 64] = test_multi_pred_conf2[index]
                pred_multi3[i * 64: (i + 1) * 64, j * 64:(j + 1) * 64] = test_multi_pred_conf3[index]

        plt.imshow(pred_bi1)
        plt.show()
        plt.imshow(pred_bi2)
        plt.show()
        plt.imshow(pred_bi3)
        plt.show()
        plt.imshow(pred_multi1)
        plt.show()
        plt.imshow(pred_multi2)
        plt.show()
        plt.imshow(pred_multi3)
        plt.show()
        cv2.imwrite('CAIM-Net-bi1.tif', pred_bi1)
        cv2.imwrite('CAIM-Net-bi2.tif', pred_bi2)
        cv2.imwrite('CAIM-Net-bi3.tif', pred_bi3)
        cv2.imwrite('CAIM-Net-multi1.tif', pred_multi1)
        cv2.imwrite('CAIM-Net-multi2.tif', pred_multi2)
        cv2.imwrite('CAIM-Net-multi3.tif', pred_multi3)

    def change_map_epoch(self, test_image):
        self.model.eval()
        with torch.no_grad():
            final_area, final_moment, area1, moment1, area2, moment2, area3, moment3, area4, moment4 = self.model(test_image)
            pred_bi = final_area.detach().max(1)[1].view(args.batch_size, args.patch_size, args.patch_size)
            pred_multi = final_moment.detach().max(1)[1].view(args.batch_size, args.patch_size, args.patch_size)
            pred_bi_list = pred_bi.data.cpu().numpy().tolist()
            pred_multi_list = pred_multi.data.cpu().numpy().tolist()
        return pred_bi_list, pred_multi_list

    def test(self):
        print('Testing..................')
        for epoch in range(0, args.max_epoch):
            # print(epoch)
            best_model_pth = args.save_path+'/Dynamic-37.67.pth'
            # best_model_pth = args.save_path + '/model_{}.pth'.format(epoch)
            self.model.load_state_dict(torch.load(best_model_pth))

            Ftrain = np.load(args.Fsplit + 'Ftest.npy').tolist()
            csv_train = args.xys + 'myxys_test.csv'

            test_dataset = Dynamic_custom.MyDataset(csv_train, Ftrain, args.patch_size)
            test_sample = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
            Test_Bi_Pred_list = []
            Test_Bi_Target_list = []
            Test_Multi_Pred_list = []
            Test_Multi_Target_list = []
            for iter_train, batch in enumerate(test_sample):
                test_batch_image, test_batch_bilabel, test_batch_multilabel = batch
                test_batch_image = test_batch_image.permute(1, 0, 4, 2, 3)

                test_batch_image = torch.as_tensor(test_batch_image, dtype=torch.float32).cuda()
                test_batch_bilabel = torch.as_tensor(test_batch_bilabel, dtype=torch.long).cuda()
                test_batch_multilabel = torch.as_tensor(test_batch_multilabel, dtype=torch.long).cuda()

                pred_bi_list, target_bi_list, pred_multi_list, target_multi_list = self.test_epoch(test_batch_image, test_batch_bilabel, test_batch_multilabel)

                Test_Bi_Pred_list.extend(pred_bi_list)
                Test_Bi_Target_list.extend(target_bi_list)
                Test_Multi_Pred_list.extend(pred_multi_list)
                Test_Multi_Target_list.extend(target_multi_list)
            test_bi_pred_conf = np.asarray(Test_Bi_Pred_list)
            test_bi_target_conf = np.asarray(Test_Bi_Target_list)
            test_multi_pred_conf = np.asarray(Test_Multi_Pred_list)
            test_multi_target_conf = np.asarray(Test_Multi_Target_list)

            Test_Bi_Matrxi = get_matrix.bi_matrix(test_bi_pred_conf, test_bi_target_conf)
            Test_Multi_Matrix = get_matrix.multi_matrix(test_multi_pred_conf, test_multi_target_conf)

            Test_Bi_Matrxi = np.asarray(Test_Bi_Matrxi)
            Test_Multi_Matrix = np.asarray(Test_Multi_Matrix)

            Test_Bi_Accuracy, Test_Bi_Precision, Test_Bi_Recall, Test_Bi_F1, Test_Bi_Kappa, Test_Bi_IoU, \
            Test_Bi_MIoU, Test_Bi_FPR, Test_Bi_FNR = get_matrix.bi_accuracy(Test_Bi_Matrxi)

            Test_Multi_Accuracy, Test_Multi_Precision, Test_Multi_Recall, Test_Multi_F1, Test_Multi_kappa, Test_Multi_IoU, \
                Test_Multi_MIoU, Test_Multi_FPR, Test_Multi_FNR = get_matrix.multi_accuracy(Test_Multi_Matrix)

            np.set_printoptions(suppress=True, precision=0)
            # print(Test_Multi_Matrix)
            print("test_bi_accuracy：%.2f test_bi_precision：%.2f test_bi_recall：%.2f test_bi_f1：%.2f test_bi_kappa：%.2f "
                  "test_multi_accuracy：%.2f test_multi_precision：%.2f test_multi_recall：%.2f test_multi_f1：%.2f "
                  "test_multi_kappa：%.2f" % (Test_Bi_Accuracy * 100, Test_Bi_Precision * 100, Test_Bi_Recall * 100,
                  Test_Bi_F1 * 100, Test_Bi_Kappa * 100, Test_Multi_Accuracy * 100, Test_Multi_Precision * 100,
                  Test_Multi_Recall * 100, Test_Multi_F1 * 100, Test_Multi_kappa * 100))

    def test_epoch(self, test_image, test_bi_label, test_multi_label):
        self.model.eval()
        pred_list, target_list = [], []
        with torch.no_grad():
            final_area, final_moment, area1, moment1, area2, moment2, area3, moment3, area4, moment4 = self.model(test_image)
            # area, area1, moment1 = self.model(test_image)

            test_bi_label = test_bi_label.view(-1)
            test_multi_label = test_multi_label.view(-1)

            pred_bi_list = final_area.detach().max(1)[1].data.cpu().numpy().tolist()
            target_bi_list = test_bi_label.data.cpu().numpy().tolist()
            pred_multi_list = final_moment.detach().max(1)[1].data.cpu().numpy().tolist()
            target_multi_list = test_multi_label.view(-1).data.cpu().numpy().tolist()
        return pred_bi_list, target_bi_list, pred_multi_list, target_multi_list



if __name__ == '__main__':
    trainer = Trainer(args)
    trainer.test()
    # trainer.change_map()  # get the change map
