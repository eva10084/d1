#########模型预测
import numpy as np
import pandas as pd
from PIL import Image
import paddle.vision.transforms as T
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.models import resnet50
tpsize = 256
split = 0.9
batch_size = 16
epochs = 5




res_model = 'resnet_50_save_models_256_0.9_16/final.pdparams'


def res(address):
    model = nn.Sequential(
        resnet50(pretrained=False),
        nn.LeakyReLU(),
        nn.Linear(1000, 2)
    )
    params = paddle.load(res_model)
    model.set_state_dict(params)
    model.eval()

    inf_transforms = T.Compose([
        T.Resize((tpsize, tpsize), interpolation='bicubic'),
        T.ToTensor()
    ])
    data_path = 'data/train/' + address  #路径后续需要调整
    data = np.asarray(Image.open(data_path).convert('RGB'))
    H, W, _ = data.shape
    data = inf_transforms(data)
    data = data.astype('float32').reshape([1, 3, tpsize, tpsize])
    pred = model(data)
    pre = [None] * 2
    # 还原坐标
    pre[0] = pred.numpy()[0][0] * W / tpsize
    pre[1] = pred.numpy()[0][1] * H / tpsize
    return pre[0], pre[1]





