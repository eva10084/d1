import os
import pandas
from PIL import Image

import torch
from torchvision import transforms
from Train import epochs
from Load import image_size

save_path = 'data/outputs/'
file_path = 'data/train/'

model = torch.load(save_path + f'resnet50_{epochs}epoch.ckpt').to('cpu')

transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                transforms.ToTensor()])

predict_list = []
with torch.set_grad_enabled(False):
    for file in os.listdir(file_path):
        image = Image.open(os.path.join(file_path, file))
        inputs = transform(image).unsqueeze(0)
        outputs = model(inputs)
        predict = [None] * 2
        # 还原坐标
        print(image.width / image_size, image.height / image_size)
        predict[0] = outputs.numpy()[0][0] * image.width / image_size
        predict[1] = outputs.numpy()[0][1] * image.height / image_size
        print(file, predict)
        # exit()
        predict_list.append([file, predict[0], predict[1]])

df = pandas.DataFrame(predict_list, columns=['FileName', 'Fovea_X', 'Fovea_Y'])
df.to_csv(save_path + f'Results_{epochs}epoch.csv', index=False)
df.to_csv(f'data/Results_{epochs}epoch.csv', index=False)
