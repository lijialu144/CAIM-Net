from skimage import io
import numpy as np
import pandas as pd
import glob
from torch.utils.data.dataset import Dataset


class MyDataset(Dataset):
    def __init__(self, csv_path, image_ids, patch_size):
        self.data_info = pd.read_csv(csv_path)
        # print('self.data_info', self.data_info.shape[0])
        self.patch_size = patch_size
        self.all_imgs = []
        for fold in image_ids:
            all_tifs = glob.glob(fold[3:] + '/images_masked/*.tif*')
            years = []
            for j in range(len(all_tifs)):
                ff = all_tifs[j].find('monthly')
                years.append(all_tifs[j][ff + 8:ff + 12] + all_tifs[j][ff + 13:ff + 15])
            ind = np.argsort(years)
            sort_tifs = [all_tifs[i] for i in ind]
            img = []
            for nd in range(0, len(sort_tifs)):
                im = io.imread(sort_tifs[nd])
                img.append(im)
            self.all_imgs.append(np.asarray(img))

        self.bi_labels = []
        for fold in image_ids:
            bi_label = io.imread(fold[3:] + '/change/bi_change.tif')
            self.bi_labels.append(bi_label)

        self.multi_labels = []
        for fold in image_ids:
            multi_label = io.imread(fold[3:] + '/change/multi_change_last.tif')
            self.multi_labels.append(multi_label)

        self.data_len = self.data_info.shape[0]



    def __getitem__(self, index):
        x = int(self.data_info.iloc[:,0][index])
        y = int(self.data_info.iloc[:,1][index])
        image_id = int(self.data_info.iloc[:,2][index])
        find_patch = self.all_imgs[image_id][:, x : x + self.patch_size, y : y + self.patch_size, :]
        find_patch = find_patch / 255.0
        find_bi_labels = self.bi_labels[image_id][x : x + self.patch_size, y : y + self.patch_size]
        find_multi_labels = self.multi_labels[image_id][x : x + self.patch_size, y : y + self.patch_size]
        return np.ascontiguousarray(find_patch), np.ascontiguousarray(find_bi_labels), np.ascontiguousarray(find_multi_labels)

    def __len__(self):
        return self.data_len

