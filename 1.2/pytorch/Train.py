import torch
from torch import nn
from torch.nn import functional
import torchvision.models
from torch.utils.data import DataLoader
import Load

from tqdm import tqdm

epochs = 155
epoch_size = 40


# 自定义损失函数
class MyLoss(nn.Module):
    def __init__(self, weights=(0.5, 0.5)):
        super(MyLoss, self).__init__()
        self.weights = weights  # 损失权重

    def forward(self, predict, label):
        # MSE均方误差
        mse_loss_x = nn.functional.mse_loss(predict[:, 0], label[:, 0], reduction='mean')
        mse_loss_y = nn.functional.mse_loss(predict[:, 1], label[:, 1], reduction='mean')
        mse_loss = 0.5 * mse_loss_x + 0.5 * mse_loss_y

        # 欧氏距离
        distance_loss = torch.square(torch.subtract(predict, label))
        distance_loss = torch.sum(distance_loss, dim=-1)
        distance_loss = torch.sqrt(distance_loss)
        distance_loss = torch.sum(distance_loss, dim=0) / predict.shape[0]  # outputs.shape[0] == batch_size

        alpha1, alpha2 = self.weights
        all_loss = alpha1 * mse_loss + alpha2 * distance_loss

        return all_loss


# 数据、模型加载
############################################################################
Load.Load_Locations()
dataset = Load.Custom_Dataset('data/Locations.csv', 'data/train')
dataloader = DataLoader(dataset, epoch_size, False)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cuda:0'
model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(2048, 1000), nn.LeakyReLU(),
    nn.Linear(1000, 2)
)
model = model.to(device)
# print(model)

loss_fn = MyLoss((0.4, 0.6))
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=5e-6)  # 优化器
lr = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)  # 调度器


# 模型训练、使用
############################################################################
def Train_Model(total_epoch):
    for epoch in tqdm(range(total_epoch)):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            # print(y)
            # print(outputs)
            # exit()
            loss = loss_fn(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tqdm.write(f'loss={loss}')
        if loss < 0.7:
            torch.save(model, f"data/outputs/resnet50_loss{.5:loss}.ckpt")
    torch.save(model, f"data/outputs/resnet50_{epochs}epoch.ckpt")


if __name__ == '__main__':
    Train_Model(epochs)
