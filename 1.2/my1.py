################查看结果
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

#显示图片上的点
path = 'data/train'
flrs = np.array(pd.read_csv('Fovea_Localization_Results.csv'))
for flr in flrs:
    img = np.array(Image.open(os.path.join(path, flr[0])))
    x, y = flr[1:]
    plt.imshow(img.astype('uint8'))
    plt.plot(x, y, 'or')
    plt.show()
    break