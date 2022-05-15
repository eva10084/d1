from xml.dom import minidom
import os

import pandas
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

import Others

image_size = 256


def Load_Locations(path='data/'):
    """
    读取(中心凹坐标, 黄斑的bounding box坐标)，保存为新的csv文件，并作为dict返回

    :param path: 文件夹根目录，应包括：data、train_location、fovea_localization_train_GT
    :return: dict(data, Fovea_X, Fovea_Y, Image_width, Image_height, BBox_xmin, BBox_xmax, BBox_ymin, BBox_ymax)
    """
    Image_width = []
    Image_height = []
    BBox_xmin = []
    BBox_xmax = []
    BBox_ymin = []
    BBox_ymax = []
    for root, dirs, files in os.walk(path + 'train_location'):
        Image_width.extend([minidom.parse(root + '/' + i).getElementsByTagName("width")[0].childNodes[0].data for i in files])
        Image_height.extend([minidom.parse(root + '/' + i).getElementsByTagName("height")[0].childNodes[0].data for i in files])
        BBox_xmin.extend([int(minidom.parse(root + '/' + i).getElementsByTagName("xmin")[0].childNodes[0].data) for i in files])
        BBox_xmax.extend([int(minidom.parse(root + '/' + i).getElementsByTagName("xmax")[0].childNodes[0].data) for i in files])
        BBox_ymin.extend([int(minidom.parse(root + '/' + i).getElementsByTagName("ymin")[0].childNodes[0].data) for i in files])
        BBox_ymax.extend([int(minidom.parse(root + '/' + i).getElementsByTagName("ymax")[0].childNodes[0].data) for i in files])
    # print(Image_width, Image_height, BBox_xmin, BBox_xmax, BBox_ymin, BBox_ymax, sep='\n')

    # 将bbox的4个坐标信息加入DataFrame
    Locations = pandas.read_csv(path + 'fovea_localization_train_GT.csv')
    Locations['Image_width'] = Image_width
    Locations['Image_height'] = Image_height
    Locations['BBox_xmin'] = BBox_xmin
    Locations['BBox_xmax'] = BBox_xmax
    Locations['BBox_ymin'] = BBox_ymin
    Locations['BBox_ymax'] = BBox_ymax
    # 将第一列改为完整的图片名称
    Locations['data'] = ['{:04d}.jpg'.format(i) for i in Locations['data']]
    # print(fovea_locations)
    Locations.to_csv('data/Locations.csv', index=False)
    return Locations.to_dict('list')


def _target_reshape(y):
    """
    按照图片的reshape，相应调整y，y = [Fovea_X, Fovea_Y, Image_width, Image_height, BBox_xmin, BBox_xmax, BBox_ymin, BBox_ymax]

    :param y: 修改前的target（Locations参数）
    :return: 修改后的target
    """
    scale_x = image_size / y[2]
    scale_y = image_size / y[3]
    # print(type(y))
    # print(y)
    y = [scale_x * y[i] if i in [0, 4, 5]
         else scale_y * y[i] if i in [1, 6, 7] else image_size for i in range(8)]
    # print(y)
    y[2:4] = []
    # print(y)
    # exit()
    return torch.Tensor(y[0:2])


class Custom_Dataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.img_labels = pandas.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                             transforms.ToTensor()])
        self.target_transform = transforms.Lambda(lambda y: _target_reshape(y))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = read_image(img_path)
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1:].tolist()
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


if __name__ == '__main__':
    Locations = Load_Locations()
    # print(Locations)
    # Others.ShowDistribution(Locations)
    # Others.ShowBoundingBoxes(Locations, 10, 20)
    Others.LoaderTest(Custom_Dataset('data/Locations.csv', 'data/train'))
