import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
from skimage import measure
import torch
from medpy import metric
from multiprocessing import Pool
from tqdm import tqdm
#
from evaluate import cvt2one_hot


def get_info(image_path):
    nii = sitk.ReadImage(image_path)
    return nii.GetSize(), nii.GetSpacing(), sitk.GetArrayFromImage(nii)


def main(case_):
    if case_ == 'inner':
        txt_template = '/mnt/sda/Users/zcb/Datasets/CT_from_zhaoyi/txts/public_endplate/new_label_fold_{}.txt'
        lines = []
        for fold in range(3):
            with open(txt_template.format(fold), 'r') as reader:
                lines += reader.readlines()
    elif case_ == 'easy':
        txt_path = '/mnt/sda/Users/zcb/Datasets/CT_from_zhaoyi/txts/public_PUTH/new_label_easy_False.txt'
        with open(txt_path, 'r') as reader:
            lines = reader.readlines()
    elif case_ == 'hard':
        txt_path = '/mnt/sda/Users/zcb/Datasets/CT_from_zhaoyi/txts/public_PUTH/new_label_hard_False.txt'
        with open(txt_path, 'r') as reader:
            lines = reader.readlines()
    # bmd_list = []
    sizes = []
    spacings = []
    for line in tqdm(lines):
        image_path, label_path = line.strip().split(',')
        image_name = os.path.basename(image_path)
        size, spacing, image = get_info(image_path)
        _, _, label = get_info(label_path)
        sizes.append(size)
        spacings.append(spacing)
        # inst_labels = cvt2one_hot(measure.label(label, connectivity=1))
        # assert len(inst_labels) in [9, 10, 11], repr(len(inst_labels)) + ', ' + line
        # bmd_list.append(np.mean(image[label > 0]))
    print(np.mean(np.array(sizes), axis=0))
    print(np.mean(np.array(spacings), axis=0))



if __name__ == '__main__':
    main('inner')
    main('easy')
    main('hard')
