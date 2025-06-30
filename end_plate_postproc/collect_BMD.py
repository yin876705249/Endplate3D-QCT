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
from evaluate import get_arr


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
        image_temp = '/mnt/sda/Users/zcb/Datasets/CT_from_zhaoyi/raw_data/public_ct/{}/CT.nii.gz'
        label_temp = '/mnt/sda/Users/zcb/Datasets/CT_from_zhaoyi/raw_data/public_ct/{0}/{0}_{1}_endplate.nii.gz'
    elif case_ == 'easy':
        txt_path = '/mnt/sda/Users/zcb/Datasets/CT_from_zhaoyi/txts/public_PUTH/new_label_easy_False.txt'
        with open(txt_path, 'r') as reader:
            lines = reader.readlines()
        image_temp = '/mnt/sdb/pku/Documents/Datasets/CT_from_zhaoyi/data_backup/PUTH_old/{}/CT.nii.gz'
        label_temp = '/mnt/sda/Users/zcb/Datasets/CT_from_zhaoyi/raw_data/PUTH/{0}/{0}_{1}_endplate.nii.gz'
    elif case_ == 'hard':
        txt_path = '/mnt/sda/Users/zcb/Datasets/CT_from_zhaoyi/txts/public_PUTH/new_label_hard_False.txt'
        with open(txt_path, 'r') as reader:
            lines = reader.readlines()
        image_temp = '/mnt/sdb/pku/Documents/Datasets/CT_from_zhaoyi/data_backup/PUTH_old/{}/CT.nii.gz'
        label_temp = '/mnt/sda/Users/zcb/Datasets/CT_from_zhaoyi/raw_data/PUTH/{0}/{0}_{1}_endplate.nii.gz'
    #
    print('-' * 100)
    print(case_)
    organs = ['L1', 'L2', 'L3', 'L4', 'L5', 'S1']
    bmds_list = [[] for _ in range(len(organs))]
    for line in tqdm(lines):
        image_name = line.strip().split(',')[0]
        image_tag = os.path.basename(image_name).split('.')[0].replace('image_', '')
        #
        if not os.path.exists(image_temp.format(image_tag)):
            print('Ignore', image_temp.format(image_tag))
            continue
        image_arr, _ = get_arr(image_temp.format(image_tag))
        for ind, organ in enumerate(organs):
            label_arr, _ = get_arr(label_temp.format(image_tag, organ))
            if organ == 'S1':
                bmds_list[ind].append(np.mean(image_arr[label_arr > 0]))
            else:
                inst_arr = measure.label(label_arr, connectivity=3)
                regions = measure.regionprops(inst_arr)
                # 按区域大小降序排序
                sorted_regions = sorted(regions, key=lambda region: region.area, reverse=True)
                if len(sorted_regions) > 2:
                    # 最大的区域
                    largest_inst_id = sorted_regions[0].label
                    # 剩余的区域合并
                    remaining_inst_ids = [region.label for region in sorted_regions[1:]]
                    # 计算 BMD，并分别存入 bmds_list
                    bmds_list[ind].append(np.mean(image_arr[inst_arr == largest_inst_id]))
                    bmds_list[ind].append(np.mean(image_arr[np.isin(inst_arr, remaining_inst_ids)]))
                else:
                    inst_ids = [region.label for region in sorted_regions]
                    for inst_id in inst_ids:
                        bmds_list[ind].append(np.mean(image_arr[inst_arr == inst_id]))
    return bmds_list


if __name__ == '__main__':
    organs = ['L1', 'L2', 'L3', 'L4', 'L5', 'S1']
    bmds_list = [[] for _ in range(len(organs))]
    #
    for case_ in ['easy', 'hard', 'inner'][:: -1]:
        new_bmds_list = main(case_)
        for organ, bmds in zip(organs, new_bmds_list):
            print(organ, len(bmds), '&', f'{np.mean(bmds):.1f} $\\pm$ {np.std(bmds, ddof=1):.1f} &')
        for ind in range(len(organs)):
            bmds_list[ind].extend(new_bmds_list[ind])
    for organ, bmds in zip(organs, bmds_list):
        print(organ, len(bmds), '&', f'{np.mean(bmds):.1f} $\\pm$ {np.std(bmds, ddof=1):.1f} &')
