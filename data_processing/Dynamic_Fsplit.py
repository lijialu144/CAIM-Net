import argparse
import glob
import random
import os
import shutil
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--train_images_folder', type=str, default='..//Images_Dynamic/train/', help='folder of training')
parser.add_argument('--val_images_folder', type=str, default='..//Images_Dynamic/val/', help='folder of valing')
parser.add_argument('--test_images_folder', type=str, default='..//Images_Dynamic/test/', help='folder of testing')
args = parser.parse_args()

Ftrain = glob.glob(args.train_images_folder + '*Scene*') #give your '/train/' folder destination
Fval = glob.glob(args.val_images_folder + '*Scene*')
Ftest = glob.glob(args.test_images_folder + '*Scene*')


train_years = []
for iter_train in range(len(Ftrain)):
    train_years.append(Ftrain[iter_train][30:32])
ind = np.argsort(train_years)
sort_Ftrain = [Ftrain[i] for i in ind]

val_years = []
for iter_val in range(len(Fval)):
    val_years.append(Fval[iter_val][28:30])
ind = np.argsort(val_years)
sort_Fval = [Fval[i] for i in ind]

test_years = []
for iter_test in range(len(Ftest)):
    test_years.append(Ftest[iter_test][29:31])
ind = np.argsort(test_years)
sort_Ftest = [Ftest[i] for i in ind]




print(sort_Ftrain)
print(sort_Fval)
print(sort_Ftest)

#
# if os.path.exists('../Dynamic_Fsplit'):
#     shutil.rmtree('../Dynamic_Fsplit')
os.mkdir('../Dynamic_Fsplit')
np.save('../Dynamic_Fsplit/Ftrain.npy', sort_Ftrain)
np.save('../Dynamic_Fsplit/Fval.npy', sort_Fval)
np.save('../Dynamic_Fsplit/Ftest.npy', sort_Ftest)

print(len(sort_Ftrain), 'folders for training')
print(len(sort_Fval), 'folders for validation')
print(len(sort_Ftest), 'folders for testing')
