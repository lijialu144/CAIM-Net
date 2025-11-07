import os
import os.path as osp
import imageio.v2
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from tqdm import tqdm
import get_matrix
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
from data_processing import Dynamic_custom
from torch.utils.data import DataLoader
from model import Dynamic_CAIMNet
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('--change_num1', type=int, default=2, help='number of change')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='initial learning')
parser.add_argument('--TSIs_len', type=int, default=6, help='image number')
parser.add_argument('--Fsplit', type=str, default='Dynamic_Fsplit/', help='Fsplit folder')
parser.add_argument('--xys', type=str, default='Dynamic_xys_64_64/', help='xys folder')
parser.add_argument('--max_epoch', type=int, default=100, help='max epoch]')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--patch_size', type=int, default=64, help='patch_size')
parser.add_argument('--adjust_lr', type=bool, default=True, help='adjust learning rate')
parser.add_argument('--learning_rate_steps', type=list, default=[100], help='learning_rate_steps')
parser.add_argument('--learning_rate_gamma', type=float, default=0.5, help='learning_rate_gamma ')
parser.add_argument('--save_path', default='best_models/', help='model save path')
args = parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def Bi_Loss(bi_pred, bi_label):
    lossinput = bi_pred
    sum = bi_label.shape[0]

    label_0 = torch.nonzero(bi_label == 0).cuda()
    label_1 = torch.nonzero(bi_label == 1).cuda()

    ratio_0 = ((sum - label_0.shape[0]) / sum)
    ratio_1 = ((sum - label_1.shape[0]) / sum)
    ratio_0_f = torch.ones([sum, 1]) * ratio_0
    ratio_1_f = torch.ones([sum, 1]) * ratio_1
    ratio = torch.cat((ratio_0_f, ratio_1_f), dim=1).cuda()
    bi_target = torch.eye(2)[bi_label, :].cuda()

    L = - (ratio * bi_target * torch.log(lossinput + 1e-10) * torch.pow(input=(1 - lossinput), exponent=2))

    W = torch.ones(bi_target.shape).cuda()
    loss = torch.mean(L * W)
    return loss

def Multi_Loss(multi_pred, multi_label):
    sum = multi_label.shape[0]

    label_0 = torch.nonzero(multi_label == 0).cuda()
    label_1 = torch.nonzero(multi_label == 1).cuda()
    label_2 = torch.nonzero(multi_label == 2).cuda()
    label_3 = torch.nonzero(multi_label == 3).cuda()
    label_4 = torch.nonzero(multi_label == 4).cuda()
    label_5 = torch.nonzero(multi_label == 5).cuda()

    ratio_0 = ((sum - label_0.shape[0]) / sum)
    ratio_1 = ((sum - label_1.shape[0]) / sum)
    ratio_2 = ((sum - label_2.shape[0]) / sum)
    ratio_3 = ((sum - label_3.shape[0]) / sum)
    ratio_4 = ((sum - label_4.shape[0]) / sum)
    ratio_5 = ((sum - label_5.shape[0]) / sum)

    ratio_0_f = torch.ones([sum, 1]) * ratio_0
    ratio_1_f = torch.ones([sum, 1]) * ratio_1
    ratio_2_f = torch.ones([sum, 1]) * ratio_2
    ratio_3_f = torch.ones([sum, 1]) * ratio_3
    ratio_4_f = torch.ones([sum, 1]) * ratio_4
    ratio_5_f = torch.ones([sum, 1]) * ratio_5

    ratio = torch.cat((ratio_0_f, ratio_1_f, ratio_2_f, ratio_3_f, ratio_4_f, ratio_5_f), dim=1).cuda()
    multi_target = torch.eye(6)[multi_label, :].cuda()
    L = - (ratio * multi_target * torch.log(multi_pred + 1e-10) * torch.pow(input=(1 - multi_pred), exponent=2))
    W = torch.ones(multi_target.shape).cuda()
    loss = torch.mean(L * W)
    return loss

class Trainer(object):
    def __init__(self, ):
        self.lr = args.learning_rate
        print('======distance rnn========')
        self.model = Dynamic_CAIMNet.Model(batch=args.batch_size, input_channels=4, features=64,
                     change_nums=args.TSIs_len, times=args.TSIs_len, size = args.patch_size).cuda()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.1)

    def adjust_learning_rate(self, optimizer, epoch, steps, gamma):     #optim=adam,epoch=epoch, steps=[10,15,20,25,30,35,40,45], lr_gamma=0.5
        if epoch == 0:
            self.lr = args.learning_rate
        if epoch in steps:
            self.lr *= gamma
            for param_group in optimizer.param_groups:    #optimizer.param_groups包括['params', 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize','foreach','capturable']
                param_group['lr'] = self.lr
                print('After modify, the learning rate is', param_group['lr'])

    def train(self):
        print('Training..................')
        Ftrain = np.load(args.Fsplit + 'Ftrain.npy').tolist()
        Fval = np.load(args.Fsplit + 'Fval.npy').tolist()

        csv_train = args.xys + 'myxys_train.csv'
        csv_val = args.xys + 'myxys_val.csv'

        train_dataset = Dynamic_custom.MyDataset(csv_train, Ftrain, args.patch_size)
        train_sample = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
        val_dataste = Dynamic_custom.MyDataset(csv_val, Fval, args.patch_size)
        val_sample = DataLoader(val_dataste, batch_size=args.batch_size, shuffle=False, drop_last=True)

        Train_Loss_list = []
        Val_Loss_list = []
        # self.model.load_state_dict(torch.load(args.save_path+'/32-Add-Loss3-2,3,5,6-37.21.pth'))
        for epoch in tqdm(range(0, args.max_epoch)):
            print('self.lr', self.lr)
            train_ave_loss = 0
            val_ave_loss = 0
            Train_Bi_Pred_list = []
            Train_Bi_Target_list = []
            Train_Multi_Pred_list = []
            Train_Multi_Target_list = []
            Val_Bi_Pred_list = []
            Val_Bi_Target_list = []
            Val_Multi_Pred_list = []
            Val_MUlti_Traget_list = []

            if args.adjust_lr:
                self.adjust_learning_rate(self.optim, epoch, args.learning_rate_steps, args.learning_rate_gamma)
                self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))

            for iter_train, batch in enumerate(train_sample):
                train_batch_image, train_batch_bilabel, train_batch_multilabel = batch
                train_batch_image = train_batch_image.permute(1, 0, 4, 2, 3)

                train_batch_image = torch.as_tensor(train_batch_image, dtype=torch.float32).cuda()
                train_batch_bilabel = torch.as_tensor(train_batch_bilabel, dtype=torch.long).cuda()
                train_batch_multilabel = torch.as_tensor(train_batch_multilabel, dtype=torch.long).cuda()

                train_epoch_loss, train_bi_pred_list, train_bi_target_list, train_multi_pred_list, train_multi_target_list = self.train_epoch(train_batch_image, train_batch_bilabel, train_batch_multilabel)
                train_ave_loss += train_epoch_loss
                Train_Bi_Pred_list.extend(train_bi_pred_list)
                Train_Bi_Target_list.extend(train_bi_target_list)
                Train_Multi_Pred_list.extend(train_multi_pred_list)
                Train_Multi_Target_list.extend(train_multi_target_list)

            for iter_val, batch in enumerate(val_sample):
                val_batch_image, val_batch_bilabel, val_batch_multilabel = batch
                val_batch_image = val_batch_image.permute(1, 0, 4, 2, 3)

                val_batch_image = torch.as_tensor(val_batch_image, dtype=torch.float32).cuda()
                val_batch_bilabel = torch.as_tensor(val_batch_bilabel, dtype=torch.long).cuda()
                val_batch_multilabel = torch.as_tensor(val_batch_multilabel, dtype=torch.long).cuda()

                val_epoch_loss, val_bi_pred_list, val_bi_target_list, val_multi_pred_list, val_multi_target_list = self.val_epoch(val_batch_image, val_batch_bilabel, val_batch_multilabel)

                val_ave_loss += val_epoch_loss
                Val_Bi_Pred_list.extend(val_bi_pred_list)
                Val_Bi_Target_list.extend(val_bi_target_list)
                Val_Multi_Pred_list.extend(val_multi_pred_list)
                Val_MUlti_Traget_list.extend(val_multi_target_list)

            train_ave_loss /= len(train_sample)
            val_ave_loss /= len(val_sample)
            Train_Loss_list.append(train_ave_loss)
            Val_Loss_list.append(val_ave_loss)

            train_bi_pred_conf = np.asarray(Train_Bi_Pred_list)
            train_bi_target_conf = np.asarray(Train_Bi_Target_list)
            train_multi_pred_conf = np.asarray(Train_Multi_Pred_list)
            train_multi_target_conf = np.asarray(Train_Multi_Target_list)
            val_bi_pred_conf = np.asarray(Val_Bi_Pred_list)
            val_bi_target_conf = np.asarray(Val_Bi_Target_list)
            val_multi_pred_conf = np.asarray(Val_Multi_Pred_list)
            val_multi_target_conf = np.asarray(Val_MUlti_Traget_list)

            Train_Bi_Matrxi = get_matrix.bi_matrix(train_bi_pred_conf, train_bi_target_conf)
            Train_Multi_Matrix = get_matrix.multi_matrix(train_multi_pred_conf, train_multi_target_conf)
            Val_Bi_Matrix = get_matrix.bi_matrix(val_bi_pred_conf, val_bi_target_conf)
            Val_Multi_Matrix = get_matrix.multi_matrix(val_multi_pred_conf, val_multi_target_conf)

            Train_Bi_Matrxi = np.asarray(Train_Bi_Matrxi)
            Train_Multi_Matrix = np.asarray(Train_Multi_Matrix)
            Val_Bi_Matrix = np.asarray(Val_Bi_Matrix)
            Val_Multi_Matrix = np.asarray(Val_Multi_Matrix)

            Train_Bi_Accuracy, Train_Bi_Precision, Train_Bi_Recall, Train_Bi_F1, Train_Bi_Kappa, Train_Bi_IoU, \
            Train_Bi_MIoU, Train_Bi_FPR, Train_Bi_FNR = get_matrix.bi_accuracy(Train_Bi_Matrxi)

            Val_Bi_Accuracy, Val_Bi_Precision, Val_Bi_Recall, Val_Bi_F1, Val_Bi_Kappa, Val_Bi_IoU, Val_Bi_MIoU, \
            Val_Bi_FPR, Val_Bi_FNR = get_matrix.bi_accuracy(Val_Bi_Matrix)

            Train_Multi_Accuracy, Train_Multi_Precision, Train_Multi_Recall, Train_Multi_F1, Train_Multi_kappa, Train_Multi_IoU, \
            Train_Multi_MIoU, Train_Multi_FPR, Train_Multi_FNR = get_matrix.multi_accuracy(Train_Multi_Matrix)

            Val_Multi_Accuracy, Val_Multi_Precision, Val_Multi_Recall, Val_Multi_F1, Val_Multi_kappa, Val_Multi_IoU, \
            Val_Multi_MIoU, Val_Multi_FPR, Val_Multi_FNR = get_matrix.multi_accuracy(Val_Multi_Matrix)

            print("train_loss：%.4f train_bi_accuracy：%.2f train_bi_precision：%.2f train_bi_recall：%.2f "
                  "train_bi_f1：%.2f train_bi_kappa：%.2f" % (train_ave_loss, Train_Bi_Accuracy * 100,
                   Train_Bi_Precision * 100, Train_Bi_Recall * 100, Train_Bi_F1 * 100, Train_Bi_Kappa * 100))
            print("train_loss：%.4f train_multi_accuracy：%.2f train_multi_precision：%.2f train_multi_recall：%.2f "
                  "train_multi_f1：%.2f train_multi_kappa：%.2f" % (train_ave_loss, Train_Multi_Accuracy * 100,
                  Train_Multi_Precision * 100, Train_Multi_Recall * 100, Train_Multi_F1 * 100, Train_Multi_kappa * 100))
            print("val_loss：%.4f val_bi_accuracy：%.2f val_bi_precision：%.2f val_bi_recall：%.2f val_bi_f1：%.2f "
                  "val_bi_kappa：%.2f" % (val_ave_loss, Val_Bi_Accuracy * 100, Val_Bi_Precision * 100,
                   Val_Bi_Recall * 100, Val_Bi_F1 * 100, Val_Bi_Kappa * 100))
            print("val_loss：%.4f val_multi_accuracy：%.2f val_multi_precision：%.2f val_multi_recall：%.2f "
                  "val_multi_f1：%.2f val_multi_kappa：%.2f" % (val_ave_loss, Val_Multi_Accuracy * 100,
                   Val_Multi_Precision * 100, Val_Multi_Recall * 100, Val_Multi_F1 * 100, Val_Multi_kappa * 100))

            # if epoch == args.max_epoch-1:
            #     best_model = args.save_path + '/model_' + str(epoch) + '.pth'
            #     torch.save(self.model.state_dict(), best_model)



    def train_epoch(self, train_image, train_bi_label, train_multi_label):
        self.model.train()
        pred_bi_list, target_bi_list, pred_multi_list, target_multi_list = [], [], [], []
        self.optim.zero_grad()
        final_area, final_moment, area1, moment1, area2, moment2, area3, moment3, area4, moment4 = self.model(train_image)
        train_bi_label = train_bi_label.view(-1)
        train_multi_label = train_multi_label.view(-1)
        loss_bi = Bi_Loss(final_area, train_bi_label)
        loss_multi = Multi_Loss(final_moment, train_multi_label)
        loss_multi1 = Multi_Loss(moment1, train_multi_label)
        loss_multi2 = Multi_Loss(moment2, train_multi_label)
        loss_multi3 = Multi_Loss(moment3, train_multi_label)
        loss_multi4 = Multi_Loss(moment4, train_multi_label)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 4)
        loss = loss_multi + loss_bi + 0.25 * (loss_multi1 + loss_multi2 + loss_multi3 + loss_multi4)
        loss.backward()
        self.optim.step()
        train_epoch_loss = loss.item()

        pred_bi_list = final_area.detach().max(1)[1].data.cpu().numpy().tolist()
        target_bi_list = train_bi_label.data.cpu().numpy().tolist()
        pred_multi_list = final_moment.detach().max(1)[1].data.cpu().numpy().tolist()
        target_multi_list = train_multi_label.view(-1).data.cpu().numpy().tolist()
        return train_epoch_loss, pred_bi_list, target_bi_list, pred_multi_list, target_multi_list

    def val_epoch(self, val_image,  val_bi_label, val_multi_label):
        self.model.eval()
        pred_bi_list, target_bi_list, pred_multi_list, target_multi_list = [], [], [], []
        with torch.no_grad():
            final_area, final_moment, area1, moment1, area2, moment2, area3, moment3, area4, moment4 = self.model(val_image)
            val_bi_label = val_bi_label.view(-1)
            val_multi_label = val_multi_label.view(-1)
            loss_bi = Bi_Loss(final_area, val_bi_label)
            loss_multi = Multi_Loss(final_moment, val_multi_label)
            loss_multi1 = Multi_Loss(moment1, val_multi_label)
            loss_multi2 = Multi_Loss(moment2, val_multi_label)
            loss_multi3 = Multi_Loss(moment3, val_multi_label)
            loss_multi4 = Multi_Loss(moment4, val_multi_label)

            loss = loss_multi + loss_bi + 0.25 * (loss_multi1 + loss_multi2 + loss_multi3 + loss_multi4)
            val_epoch_loss = loss.item()

            pred_bi_list = final_area.detach().max(1)[1].data.cpu().numpy().tolist()
            target_bi_list = val_bi_label.data.cpu().numpy().tolist()
            pred_multi_list = final_moment.detach().max(1)[1].data.cpu().numpy().tolist()
            target_multi_list = val_multi_label.view(-1).data.cpu().numpy().tolist()
        return val_epoch_loss, pred_bi_list, target_bi_list, pred_multi_list, target_multi_list

if __name__ == '__main__':
    set_seed(1)
    trainer = Trainer()
    trainer.train()
