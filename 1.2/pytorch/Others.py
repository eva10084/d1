import matplotlib.pyplot as plt
import numpy
import cv2

from torch.utils.data import DataLoader


def Show2Image(a, b):
    """显示两张图片"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用中文
    plt.subplot(121)  # 布局（行列序号）
    plt.imshow(a)  # imshow()对图像进行处理，画出图像，show()进行图像显示
    plt.title('图像1')
    plt.axis('off')  # 不显示坐标轴

    plt.subplot(122)
    plt.imshow(b)
    plt.title('图像2')
    plt.axis('off')  # 不显示坐标轴
    plt.tight_layout()
    plt.show()


def ShowDistribution(Locations):
    """显示中心凹在bounding box的分布位置"""
    x, y = Locations['Fovea_X'], Locations['Fovea_Y']
    xmin, xmax, ymin, ymax = Locations['BBox_xmin'], Locations['BBox_xmax'], Locations['BBox_ymin'], Locations['BBox_ymax']
    # print(x), print(xmax)
    percent_x = [(x[i] - xmin[i]) / (xmax[i] - xmin[i]) for i in range(len(x))]
    percent_y = [(y[i] - ymin[i]) / (ymax[i] - ymin[i]) for i in range(len(y))]
    print('percent_x的波动范围：{}'.format(numpy.max(percent_x) - numpy.min(percent_x)))
    print('percent_y的波动范围：{}'.format(numpy.max(percent_y) - numpy.min(percent_y)))
    print('percent_x的中值：{}'.format((numpy.max(percent_x) + numpy.min(percent_x)) / 2))
    print('percent_y的中值：{}'.format((numpy.max(percent_y) + numpy.min(percent_y)) / 2))
    print('中央凹在标定框归一化分布如图')
    plt.scatter(percent_x, percent_y)
    plt.show()


def ShowBoundingBox(image, Fovea_X, Fovea_Y, BBox_xmin, BBox_xmax, BBox_ymin, BBox_ymax):
    Fovea_X, Fovea_Y = int(Fovea_X), int(Fovea_Y)
    cv2.rectangle(image, pt1=(BBox_xmin, BBox_ymin), pt2=(BBox_xmax, BBox_ymax), color=(255, 0, 0), thickness=1)
    cv2.line(image, pt1=(BBox_xmin, BBox_ymin), pt2=(BBox_xmax, BBox_ymax), color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.line(image, pt1=(BBox_xmin, BBox_ymax), pt2=(BBox_xmax, BBox_ymin), color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    # cv2.drawMarker(file,(Fovea_X,Fovea_Y),color=(0,0,255))
    cv2.circle(image, center=(Fovea_X, Fovea_Y), radius=1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.rectangle(image, pt1=(Fovea_X - 50, Fovea_Y - 50), pt2=(Fovea_X + 50, Fovea_Y + 50), color=(0, 255, 0), thickness=1)

    cv2.namedWindow('title', 0)
    # cv2.resizeWindow('title',500,500)
    cv2.imshow('title', image[BBox_ymin - 100:BBox_ymax + 100, BBox_xmin - 100:BBox_xmax + 100, :])
    cv2.waitKey()


def ShowBoundingBoxes(Locations, a=0, b=10):
    for i in range(a, b):
        # print('image/train/{:04d}.jpg'.format(i))
        image = cv2.imread('image/train/{:04d}.jpg'.format(i + 1))
        Fovea_X, Fovea_Y, BBox_xmin, BBox_xmax, BBox_ymin, BBox_ymax = Locations['Fovea_X'], Locations['Fovea_Y'], Locations[
            'BBox_xmin'], Locations['BBox_xmax'], Locations['BBox_ymin'], Locations['BBox_ymax']
        # print(Fovea_X, Fovea_Y, BBox_xmin, BBox_xmax, BBox_ymin, BBox_ymax)
        ShowBoundingBox(image, Fovea_X[i], Fovea_Y[i], BBox_xmin[i], BBox_xmax[i], BBox_ymin[i], BBox_ymax[i])


def LoaderTest(dataset):
    dataloader = DataLoader(dataset, 10, shuffle=False)
    train_features, train_labels = next(iter(dataloader))
    print(f"Labels形状: {train_labels.shape}")
    print(f"Labels: {train_labels}")
    print(f"Labels[0]形状: {train_labels[0].shape}")
    print(f"Labels[0]: {train_labels[0]}")
    print(f"Labels[0]片段截取: {train_labels[0][0:2]}")
    # Show2Image(train_features[0].numpy().transpose(1, 2, 0), train_features[9].numpy().transpose(1, 2, 0))
