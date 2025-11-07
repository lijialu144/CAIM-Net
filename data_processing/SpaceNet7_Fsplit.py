import argparse
import glob
import random
import os
import shutil
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--train_images_folder', type=str, default='../Images_SpaceNet7/train/', help='train image folder')
parser.add_argument('--val_images_folder', type=str, default='../Images_SpaceNet7/val/', help='val image folder')
parser.add_argument('--test_images_folder', type=str, default='../Images_SpaceNet7/test/', help='test image folder')
args = parser.parse_args()

Ftrain = glob.glob(args.train_images_folder + '*L15*')
Fval = glob.glob(args.val_images_folder + '*L15*')
Ftest = glob.glob(args.test_images_folder + '*L15*')

train_years = []
for iter_train in range(len(Ftrain)):
    train_years.append(Ftrain[iter_train][30:34]+Ftrain[iter_train][36:40]+Ftrain[iter_train][42:46])
ind = np.argsort(train_years)
sort_Ftrain = [Ftrain[i] for i in ind]

val_years = []
for iter_val in range(len(Fval)):
    val_years.append(Fval[iter_val][28:32]+Fval[iter_val][34:38]+Fval[iter_val][40:44])
ind = np.argsort(val_years)
sort_Fval = [Fval[i] for i in ind]

test_years = []
for iter_test in range(len(Ftest)):
    test_years.append(Ftest[iter_test][29:33]+Ftest[iter_test][35:39]+Ftest[iter_test][41:45])
ind = np.argsort(test_years)
sort_Ftest = [Ftest[i] for i in ind]



print(sort_Ftrain)
print(sort_Fval)
print(sort_Ftest)

# if os.path.exists('SpaceNet7_Fsplit'):
#     shutil.rmtree('SpaceNet7_Fsplit')
os.mkdir('../SpaceNet7_Fsplit')
np.save('../SpaceNet7_Fsplit/Ftrain.npy', sort_Ftrain)
np.save('../SpaceNet7_Fsplit/Fval.npy', sort_Fval)
np.save('../SpaceNet7_Fsplit/Ftest.npy', sort_Ftest)



print(len(Ftrain), 'folders for training')
print(len(Fval), 'folders for validation')
print(len(Ftest), 'folders for testing')

