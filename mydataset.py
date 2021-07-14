import torch.utils.data as data
import glob
import torchvision.transforms.functional as TF
from PIL import Image


class create_dataset(data.Dataset):
    def __init__(self, main_dir='', sub_dir=''):
        super(create_dataset, self).__init__()

        self.input_path = ('%s/%s' % (main_dir, sub_dir))
        self.x_data_path = sorted(glob.glob(r'%s/RF_crop_D_*_Phi_*_ellipse_*_iter_*_aug_*' % self.input_path))
        self.y_data_path = sorted(glob.glob(r'%s/AC_median_D_*_Phi_*_ellipse_*_iter_*_aug_*' % self.input_path))
        self.roi_seg_info_path = sorted(glob.glob(r'%s/ROI_crop_info_D_*_Phi_*_ellipse_*_iter_*_aug_*' % self.input_path))

        print('number of data : ' + str(len(self.x_data_path)))
        print('number of data : ' + str(len(self.y_data_path)))
        print('number of data : ' + str(len(self.roi_seg_info_path)))

    def __getitem__(self, index):
        x_index_path = self.x_data_path[index]
        y_index_path = self.y_data_path[index]
        roi_seg_index_path = self.roi_seg_info_path[index]

        x_image = Image.open(x_index_path)
        y_image = Image.open(y_index_path)
        roi_seg_info = Image.open(roi_seg_index_path)

        return {'rf_data': TF.to_tensor(x_image), 'Q_value': TF.to_tensor(y_image), 'roi_seg_info':TF.to_tensor(roi_seg_info)}

    def __len__(self):
        return len(self.x_data_path)

