import os
import SimpleITK as sitk
import numpy as np
from skimage import measure



def main():
    endplate_dir = '/mnt/sdb/pku/Documents/project_oral/general_segment/output/endplate_newdata_newlabel_vnet_fold_all_2025-03_test/infer_results'
    spine_dir = '/mnt/sdb/pku/Documents/project_oral/general_segment/output/spine1k_on_endplate_newdata_new_label_test/infer_results'
    save_dir = 'combined'
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取文件列表
    spine_files = sorted([f for f in os.listdir(spine_dir) if f.endswith('.nii.gz')])
    endplate_files = sorted([f for f in os.listdir(endplate_dir) if f.endswith('.nii.gz')])

    # 确保两个目录的文件数量一致
    if len(spine_files) != len(endplate_files):
        print("Error: spine_dir and endplate_dir have different numbers of files!")
        print(len(spine_files), len(endplate_files))
        return

    # 遍历文件
    for spine_file, endplate_file in zip(spine_files, endplate_files):
        # 读取 spine 文件
        spine_path = os.path.join(spine_dir, spine_file)
        spine_img = sitk.ReadImage(spine_path)
        spine_data = sitk.GetArrayFromImage(spine_img)

        # 读取 endplate 文件
        endplate_path = os.path.join(endplate_dir, endplate_file)
        endplate_img = sitk.ReadImage(endplate_path)
        endplate_data = sitk.GetArrayFromImage(endplate_img)

        # 检查数据形状是否一致
        if spine_data.shape != endplate_data.shape:
            print(f"Error: Shape mismatch between {spine_file} and {endplate_file}")
            continue

        # 将 endplate 中为 1 的区域覆盖到 spine 中，并赋值为 2
        spine_data[endplate_data == 1] = 2

        # 创建新的 SimpleITK 图像
        new_img = sitk.GetImageFromArray(spine_data)
        new_img.CopyInformation(spine_img)  # 保留原始图像的元信息（如空间坐标系）

        # 保存到 save_dir
        save_path = os.path.join(save_dir, spine_file)
        sitk.WriteImage(new_img, save_path)
        print(f"Saved: {save_path}")



if __name__ == '__main__':
    main()
