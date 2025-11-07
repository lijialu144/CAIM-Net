import argparse
import numpy as np
from skimage import io
from skimage.transform import rotate, resize
import os
import cv2
import pandas as pd
import glob
import random
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--Fsplit', type=str, default='../Dynamic_Fsplit/', help='Fsplit folder')
parser.add_argument('--patch_size', type=int, default=64, help='patch size')
parser.add_argument('--step', type=int, default=64, help='sliding step')
args = parser.parse_args()

patch_s = args.patch_size
step = args.step
Ftrain = np.load(args.Fsplit + 'Ftrain.npy').tolist()
Fval = np.load(args.Fsplit + 'Fval.npy').tolist()
Ftest = np.load(args.Fsplit + 'Ftest.npy').tolist()

print('Ftrain',len(Ftrain))
print('Fval',len(Fval))
print('Ftest',len(Ftest))

def sliding_window_train(i_city, labeled_areas, label, window_size, step):
    city=[]
    fpatches_labels=[]
    x=0
    for x in range(0, label.shape[0], step):
        for y in range(0, label.shape[1], step):
            if (not y + window_size > label.shape[1]) and (not x + window_size > label.shape[0]):
                line = np.array([x, y, labeled_areas.index(i_city)])
                city.append(line)
    return np.asarray(city)

# if os.path.exists('../Dynamic_xys_64'):
#     shutil.rmtree('../Dynamic_xys_64')
os.mkdir('../Dynamic_xys_64_64')


train_cities=[]
for i_city in Ftrain:
    print('train ', i_city)
    train_path = i_city + '/change/bi_change.tif'
    train_gt = io.imread(train_path)
    xy_city = sliding_window_train(i_city, Ftrain, train_gt, patch_s, step)
    train_cities.append(xy_city)
final_train_cities = np.concatenate(train_cities, axis=0)
train_df = pd.DataFrame({'X': list(final_train_cities[:,0]), 'Y': list(final_train_cities[:,1]), 'image_ID': list(final_train_cities[:,2]), })
train_df.to_csv('../Dynamic_xys_64_64/myxys_train.csv', index=False, columns=["X", "Y", "image_ID"])

val_cities=[]
for i_city in Fval:
    print('val ', i_city)
    path = i_city + '/change/bi_change.tif'
    val_gt = io.imread(path)
    xy_city = sliding_window_train(i_city, Fval, val_gt, patch_s, step)
    val_cities.append(xy_city)
final_val_cities = np.concatenate(val_cities, axis=0)
val_df = pd.DataFrame({'X': list(final_val_cities[:,0]), 'Y': list(final_val_cities[:,1]), 'image_ID': list(final_val_cities[:,2]),})
val_df.to_csv('../Dynamic_xys_64_64/myxys_val.csv', index=False, columns=["X", "Y", "image_ID"])

test_cities=[]
for i_city in Ftest:
    print('test ', i_city)
    path = i_city + '/change/bi_change.tif'
    train_gt = io.imread(path)
    xy_city = sliding_window_train(i_city, Ftest, train_gt, patch_s, step)
    test_cities.append(xy_city)
final_test_cities = np.concatenate(test_cities, axis=0)
test_df = pd.DataFrame({'X': list(final_test_cities[:,0]), 'Y': list(final_test_cities[:,1]), 'image_ID': list(final_test_cities[:,2]),})
test_df.to_csv('../Dynamic_xys_64_64/myxys_test.csv', index=False, columns=["X", "Y", "image_ID"])


