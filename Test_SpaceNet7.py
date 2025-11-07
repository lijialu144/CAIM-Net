import os
import os.path as osp
import imageio.v2
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import numpy as np
from data_processing import SpaceNet7_custom
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchnet as tnt
from tqdm import tqdm
import get_f1
import matplotlib.pyplot as plt
from model import SpaceNet7_CAIMNet
import warnings
import time
import cv2
warnings.filterwarnings("ignore")

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('--max_epoch', type=int, default=1, help='max epoch]')
parser.add_argument('--Fsplit', type=str, default='SpaceNet7_Fsplit/', help='Fsplit folder')
parser.add_argument('--xys', type=str, default='SpaceNet7_xys_64_63/', help='xys folder')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--patch_size', type=int, default=64, help='patch_size')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='initial learning')
parser.add_argument('--change_num1', type=int, default=2, help='number of change')
parser.add_argument('--TSIs_len', type=int, default=9, help='image number')
parser.add_argument('--save_path', default='best_models/', help='model save path')
args = parser.parse_args()


class Trainer(object):
    def __init__(self, cfig):
        self.model = SpaceNet7_CAIMNet.Model(batch=args.batch_size, input_channels=4, features=64,
                     change_nums=args.TSIs_len, times=args.TSIs_len, size=args.patch_size).cuda()

    def change_map(self):
        print('Testing..................')
        best_model_pth = args.save_path + '/SpaceNet7-51.49.pth'
        self.model.load_state_dict(torch.load(best_model_pth))
        Ftest = np.load(args.Fsplit + 'Ftest.npy').tolist()
        csv_test = args.xys + 'myxys_test.csv'

        test_dataset = SpaceNet7_custom.MyDataset(csv_test, Ftest, args.patch_size)
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
        test_bi_pred_conf = test_bi_pred_conf[:,0:63, 0:63]
        test_multi_pred_conf = test_multi_pred_conf[:, 0:63, 0:63]

        test_bi_pred_conf1 = test_bi_pred_conf[0:256]
        test_bi_pred_conf2 = test_bi_pred_conf[256:256 * 2]
        test_bi_pred_conf3 = test_bi_pred_conf[256 * 2:256 * 3]
        test_bi_pred_conf4 = test_bi_pred_conf[256 * 3:256 * 4]
        test_bi_pred_conf5 = test_bi_pred_conf[256 * 4:256 * 5]
        test_bi_pred_conf6 = test_bi_pred_conf[256 * 5:]
        test_multi_pred_conf1 = test_multi_pred_conf[0:256]
        test_multi_pred_conf2 = test_multi_pred_conf[256:256*2]
        test_multi_pred_conf3 = test_multi_pred_conf[256 * 2:256 * 3]
        test_multi_pred_conf4 = test_multi_pred_conf[256 * 3:256 * 4]
        test_multi_pred_conf5 = test_multi_pred_conf[256 * 4:256 * 5]
        test_multi_pred_conf6 = test_multi_pred_conf[256 * 5:]

        pred_bi1 = np.random.rand(1008, 1008)
        pred_bi2 = np.random.rand(1008, 1008)
        pred_bi3 = np.random.rand(1008, 1008)
        pred_bi4 = np.random.rand(1008, 1008)
        pred_bi5 = np.random.rand(1008, 1008)
        pred_bi6 = np.random.rand(1008, 1008)
        pred_multi1 = np.random.rand(1008, 1008)
        pred_multi2 = np.random.rand(1008, 1008)
        pred_multi3 = np.random.rand(1008, 1008)
        pred_multi4 = np.random.rand(1008, 1008)
        pred_multi5 = np.random.rand(1008, 1008)
        pred_multi6 = np.random.rand(1008, 1008)



        for i in range(16):
            for j in range(16):
                # 计算当前块在96x64x64矩阵中的索引
                index = (i * 16 // 16) * 16 + (j * 16 // 16)
                pred_bi1[i * 63: (i + 1) * 63, j * 63:(j + 1) * 63] = test_bi_pred_conf1[index]
                pred_bi2[i * 63: (i + 1) * 63, j * 63:(j + 1) * 63] = test_bi_pred_conf2[index]
                pred_bi3[i * 63: (i + 1) * 63, j * 63:(j + 1) * 63] = test_bi_pred_conf3[index]
                pred_bi4[i * 63: (i + 1) * 63, j * 63:(j + 1) * 63] = test_bi_pred_conf4[index]
                pred_bi5[i * 63: (i + 1) * 63, j * 63:(j + 1) * 63] = test_bi_pred_conf5[index]
                pred_bi6[i * 63: (i + 1) * 63, j * 63:(j + 1) * 63] = test_bi_pred_conf6[index]
                pred_multi1[i * 63: (i + 1) * 63, j * 63:(j + 1) * 63] = test_multi_pred_conf1[index]
                pred_multi2[i * 63: (i + 1) * 63, j * 63:(j + 1) * 63] = test_multi_pred_conf2[index]
                pred_multi3[i * 63: (i + 1) * 63, j * 63:(j + 1) * 63] = test_multi_pred_conf3[index]
                pred_multi4[i * 63: (i + 1) * 63, j * 63:(j + 1) * 63] = test_multi_pred_conf4[index]
                pred_multi5[i * 63: (i + 1) * 63, j * 63:(j + 1) * 63] = test_multi_pred_conf5[index]
                pred_multi6[i * 63: (i + 1) * 63, j * 63:(j + 1) * 63] = test_multi_pred_conf6[index]

        plt.imshow(pred_bi1)
        plt.show()
        plt.imshow(pred_bi2)
        plt.show()
        plt.imshow(pred_bi3)
        plt.show()
        plt.imshow(pred_bi4)
        plt.show()
        plt.imshow(pred_bi5)
        plt.show()
        plt.imshow(pred_bi6)
        plt.show()
        plt.imshow(pred_multi1)
        plt.show()
        plt.imshow(pred_multi2)
        plt.show()
        plt.imshow(pred_multi3)
        plt.show()
        plt.imshow(pred_multi4)
        plt.show()
        plt.imshow(pred_multi5)
        plt.show()
        plt.imshow(pred_multi6)
        plt.show()
        cv2.imwrite('SpaceNet7-Moment-add-bi1.tif', pred_bi1)
        cv2.imwrite('SpaceNet7-Moment-add-bi2.tif', pred_bi2)
        cv2.imwrite('SpaceNet7-Moment-add-bi3.tif', pred_bi3)
        cv2.imwrite('SpaceNet7-Moment-add-bi4.tif', pred_bi4)
        cv2.imwrite('SpaceNet7-Moment-add-bi5.tif', pred_bi5)
        cv2.imwrite('SpaceNet7-Moment-add-bi6.tif', pred_bi6)

        cv2.imwrite('SpaceNet7-Moment-add-multi1.tif', pred_multi1)
        cv2.imwrite('SpaceNet7-Moment-add-multi2.tif', pred_multi2)
        cv2.imwrite('SpaceNet7-Moment-add-multi3.tif', pred_multi3)
        cv2.imwrite('SpaceNet7-Moment-add-multi4.tif', pred_multi4)
        cv2.imwrite('SpaceNet7-Moment-add-multi5.tif', pred_multi5)
        cv2.imwrite('SpaceNet7-Moment-add-multi6.tif', pred_multi6)

    def change_map_epoch(self, test_image):
        self.model.eval()
        with torch.no_grad():
            final_area, final_moment = self.model(test_image)
            pred_bi = final_area.detach().max(1)[1].view(args.batch_size, args.patch_size, args.patch_size)
            pred_multi = final_moment.detach().max(1)[1].view(args.batch_size, args.patch_size, args.patch_size)
            pred_bi_list = pred_bi.data.cpu().numpy().tolist()
            pred_multi_list = pred_multi.data.cpu().numpy().tolist()
        return pred_bi_list, pred_multi_list

    def test(self):
        print('Testing..................')
        for epoch in range(0, args.max_epoch):
            print(epoch)
            best_model_pth = args.save_path+'/SpaceNet7-51.49.pth'
            # best_model_pth = args.save_path + '/model_{}.pth'.format(epoch)
            self.model.load_state_dict(torch.load(best_model_pth))

            Ftest= np.load(args.Fsplit + 'Ftest.npy').tolist()
            csv_test = args.xys + 'myxys_test.csv'

            test_dataset = SpaceNet7_custom.MyDataset(csv_test, Ftest, args.patch_size)
            test_sample = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
            Test_Bi_Pred_list = []
            Test_Bi_Target_list = []
            Test_Multi_Pred_list = []
            Test_Multi_Target_list = []
            for iter_test, batch in enumerate(test_sample):
                test_batch_image, test_batch_bilabel, test_batch_multilabel = batch
                test_batch_image = test_batch_image.permute(1, 0, 4, 2, 3)

                test_batch_image = torch.as_tensor(test_batch_image, dtype=torch.float32).cuda()
                test_batch_bilabel = torch.as_tensor(test_batch_bilabel, dtype=torch.long).cuda()
                test_batch_multilabel = torch.as_tensor(test_batch_multilabel, dtype=torch.long).cuda()

                pred_bi_list, target_bi_list, pred_multi_list, target_multi_list  = self.test_epoch(test_batch_image, test_batch_bilabel, test_batch_multilabel)

                Test_Bi_Pred_list.extend(pred_bi_list)
                Test_Bi_Target_list.extend(target_bi_list)
                Test_Multi_Pred_list.extend(pred_multi_list)
                Test_Multi_Target_list.extend(target_multi_list)
            test_bi_pred_conf = np.asarray(Test_Bi_Pred_list)
            test_bi_target_conf = np.asarray(Test_Bi_Target_list)
            test_multi_pred_conf = np.asarray(Test_Multi_Pred_list)
            test_multi_target_conf = np.asarray(Test_Multi_Target_list)

            Test_Bi_Matrxi = get_f1.bi_matrix(test_bi_pred_conf, test_bi_target_conf)
            Test_Multi_Matrix = get_f1.multi_matrix(test_multi_pred_conf, test_multi_target_conf)

            Test_Bi_Matrxi = np.asarray(Test_Bi_Matrxi)
            Test_Multi_Matrix = np.asarray(Test_Multi_Matrix)

            Test_Bi_Accuracy, Test_Bi_Precision, Test_Bi_Recall, Test_Bi_F1, Test_Bi_Kappa, Test_Bi_IoU, \
            Test_Bi_MIoU, Test_Bi_FPR, Test_Bi_FNR = get_f1.bi_accuracy(Test_Bi_Matrxi)

            Test_Multi_Accuracy, Test_Multi_Precision, Test_Multi_Recall, Test_Multi_F1, Test_Multi_kappa, Test_Multi_IoU, \
            Test_Multi_MIoU, Test_Multi_FPR, Test_Multi_FNR = get_f1.multi_accuracy(Test_Multi_Matrix)
            np.set_printoptions(suppress=True, precision=0)
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
