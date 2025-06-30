import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
from skimage import measure
import torch
from medpy import metric
from multiprocessing import Pool


def resample_image(itk_image, out_spacing=(1., 1., 1.), out_size=None, is_label=True):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    if (
        ((out_spacing is None) and (out_size is None))
        or ((out_spacing is not None) and (tuple(out_spacing) == original_spacing))
        or ((out_size is not None) and (tuple(out_size) == original_size))
    ):
        return itk_image
    if out_size is None:
        for axis in range(3):
            if out_spacing[axis] == -1:
                out_spacing[axis] = original_spacing[axis]
        out_size = [
            round(original_size[0] * (original_spacing[0] / out_spacing[0])),
            round(original_size[1] * (original_spacing[1] / out_spacing[1])),
            round(original_size[2] * (original_spacing[2] / out_spacing[2]))
        ]
    else:
        assert out_spacing is None
        out_spacing = [
            original_size[0] * original_spacing[0] / out_size[0],
            original_size[1] * original_spacing[1] / out_size[1],
            original_size[2] * original_spacing[2] / out_size[2],
        ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    if is_label:
        itk_image = sitk.Cast(itk_image, sitk.sitkUInt8)
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
        resample.SetDefaultPixelValue(0)
    else:
        itk_image = sitk.Cast(itk_image, sitk.sitkFloat32)
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetDefaultPixelValue(float(np.min(sitk.GetArrayFromImage(itk_image))))
    return resample.Execute(itk_image)  # Not an inplace operation


def get_arr(image_path):
    nii = resample_image(sitk.ReadImage(image_path))
    return sitk.GetArrayFromImage(nii), nii.GetSpacing()


def get_scores(pred_mask, gt_mask):
    res = {'dice': 0, 'jc': 0, 'hd95': 100, 'asd': 100}
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    # res['dice'] = 2 * np.sum(pred_mask * gt_mask) / (np.sum(pred_mask) + np.sum(gt_mask) + 1e-6)
    try:
        res['dice'] = metric.binary.dc(pred_mask, gt_mask)
        res['jc'] = metric.binary.jc(pred_mask, gt_mask)
        res['hd95'] = metric.binary.hd95(pred_mask, gt_mask).item()
        res['asd'] = metric.binary.asd(pred_mask, gt_mask).item()
    except Exception as e:
        print(e)
    return res


def get_max_connect(binary_image):
    labeled_image = measure.label(binary_image, connectivity=1)
    regions = measure.regionprops(labeled_image)
    max_region = max(regions, key=lambda r: r.area)  # 按区域面积排序
    largest_region_mask = labeled_image == max_region.label
    return largest_region_mask


def cvt2one_hot(pred_arr, min_area=1000):
    regions = measure.regionprops(pred_arr)
    regions.sort(key=lambda x: x.centroid[2])  # sorted by the x-axis
    unique_values = np.array([region.label for region in regions if region.area >= min_area])
    pred_arr = (pred_arr[None] == unique_values[:, None, None, None])  # (#unique_values, D, H, W)
    return pred_arr


def match_score(pred_arr, label_arr):
    device = 'cpu'
    pred_arr = cvt2one_hot(measure.label(pred_arr, connectivity=1))
    pred_tensor = torch.as_tensor(pred_arr, dtype=torch.float, device=device)
    label_arr = cvt2one_hot(measure.label(label_arr, connectivity=1))
    label_tensor = torch.as_tensor(label_arr, dtype=torch.float, device=device)
    #
    inter = torch.sum(pred_tensor[None] * label_tensor[:, None], dim=(2, 3, 4))
    sum_ = torch.sum(pred_tensor[None], dim=(2, 3, 4)) + torch.sum(label_tensor[:, None], dim=(2, 3, 4))
    dice_matrix = (2 * inter / (sum_ + 1e-6)).cpu().numpy()  # (#pred_insts, #label_insts)
    if dice_matrix.shape[1] == 0:
        return {'dice': 0, 'jc': 0, 'hd95': 100, 'asd': 100, 'Prec': 0, 'Reca': 0}
    #
    assignments = np.argmax(dice_matrix, axis=1)  # (#label_insts, )
    #
    TP = 0
    inst_scores_dict = {'dice': [], 'jc': [], 'hd95': [], 'asd': []}
    for label_ind, pred_ind in enumerate(assignments):
        inst_score_dict = get_scores(pred_arr[pred_ind], label_arr[label_ind])
        if inst_score_dict['dice'] > 0.8:
            TP += 1
            for key in inst_score_dict.keys():
                inst_scores_dict[key].append(inst_score_dict[key])
    inst_scores_dict = {key : np.mean(value).item() for key, value in inst_scores_dict.items()}
    inst_scores_dict.update({'Prec': TP / pred_arr.shape[0], 'Reca': TP / label_arr.shape[0]})
    return inst_scores_dict


def evaluate(args):
    score_dict = {}
    #
    pred_path, line = args
    print(line)
    #
    image_path, label_path = line.strip().split(',')
    image_name = os.path.basename(image_path)
    label_arr, spacing = get_arr(label_path)
    pred_arr, _ = get_arr(pred_path)
    score_dict['id'] = image_name
    #
    label_arr = (label_arr > 0)
    pred_arr = (pred_arr > 0)
    # Global dice & jaccard
    score_dict['glob_dice'] = (2 * np.sum(label_arr & pred_arr) / (np.sum(label_arr) + np.sum(pred_arr))).item()
    score_dict['glob_jacc'] = (np.sum(label_arr & pred_arr) / np.sum(label_arr | pred_arr)).item()
    score_dict.update(match_score(pred_arr, label_arr))
    return score_dict


def main_inner_set(model_name):
    if model_name in ['unet', 'effunet']:
        txt_template = '/mnt/sda/Users/zcb/Datasets/CT_from_zhaoyi/txts/public_endplate/new_label_fold_{}_sag.txt'
        pred_template = (
            '/mnt/sda/Users/zcb/project_oral/general_segment/output/'
            f'endplate_newdata_newlabel_{model_name}_'
            'fold_{}_sag/infer_results/{}'
        )
    else:
        txt_template = '/mnt/sda/Users/zcb/Datasets/CT_from_zhaoyi/txts/public_endplate/new_label_fold_{}.txt'
        pred_template = (
            '/mnt/sda/Users/zcb/project_oral/general_segment/output/'
            f'endplate_newdata_newlabel_{model_name}_'
            'fold_{}/infer_results/{}'
        )
    #
    args_list = []
    for fold in range(3):
        with open(txt_template.format(fold), 'r') as reader:
            lines = reader.readlines()
        for line in lines:
            image_path, label_path = line.strip().split(',')
            image_name = os.path.basename(image_path)
            args_list.append((pred_template.format(fold, image_name), image_name))
    with Pool(36) as p:
        score_dicts = p.map(evaluate, args_list)
    #
    res = {}
    for score_dict in score_dicts:
        for key, value in score_dict.items():
            if not key in res:
                res[key] = [value, ]
            else:
                res[key].append(value)
    pd.DataFrame(res).sort_values(by='id').to_csv(f'scores/{model_name}.tsv', sep='\t', index=False)


def main_outer_set(model_name, case):
    save_dir = f'scores_{case}/'
    os.makedirs(save_dir, exist_ok=True)
    #
    if model_name in ['unet', 'effunet']:
        sag = True
    else:
        sag = False
    #
    txt_path = f'/mnt/sda/Users/zcb/Datasets/CT_from_zhaoyi/txts/public_PUTH/new_label_{case}_{sag}.txt'
    pred_template = (
        '/mnt/sda/Users/zcb/project_oral/general_segment/output/'
        f'endplate_newdata_newlabel_{model_name}_fold_all_PUTH_{case}_test/'
        'infer_results/{}'
    )
    #
    args_list = []
    with open(txt_path, 'r') as reader:
        lines = reader.readlines()
    for line in lines:
        image_path, label_path = line.strip().split(',')
        image_name = os.path.basename(image_path)
        args_list.append((pred_template.format(image_name), line))
    # evaluate(args_list[0])
    with Pool(36) as p:
        score_dicts = p.map(evaluate, args_list)
    #
    res = {}
    for score_dict in score_dicts:
        for key, value in score_dict.items():
            if not key in res:
                res[key] = [value, ]
            else:
                res[key].append(value)
    pd.DataFrame(res).sort_values(by='id').to_csv(f'{save_dir}/{model_name}.tsv', sep='\t', index=False)


if __name__ == '__main__':
    model_names = ['unet3d', 'vnet', 'unetr', 'swinunetr', 'unet', 'effunet']
    # for model_name in model_names:
    #     main_inner_set(model_name)
    for model_name in model_names:
        for case in ['easy', 'hard'][1 :]:
            main_outer_set(model_name, case)