import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch

torch.manual_seed(0)  # 设置随机种子以确保结果可重复
torch.backends.cudnn.deterministic = False  # 允许某些非确定性操作，以提高训练速度
torch.backends.cudnn.benchmark = True  # 允许CUDNN自动寻找最优的算法，以提高训练速度

import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
import timm  # 提供了大量的预训练模型
import time

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm  # 用于显示训练和验证过程的进度条

# 读取训练和验证数据的标签文件
train_label = pd.read_csv('/root/lanyun-tmp/phase1/trainset_label.txt')
val_label = pd.read_csv('/root/lanyun-tmp/phase1/valset_label.txt')

# 生成训练和验证数据的路径
train_label['path'] = '/root/lanyun-tmp/phase1/trainset/' + train_label['img_name']
val_label['path'] = '/root/lanyun-tmp/phase1/valset/' + val_label['img_name']


# 定义一个计算并存储平均值和当前值的类
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


# 定义一个显示进度的类
class ProgressMeter(object):
    def __init__(self, num_batches, *meters):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = ""

    def pr2int(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# 定义验证过程
def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1)

    # 切换到评估模式
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # 计算输出
            output = model(input)
            loss = criterion(output, target)

            # 计算准确度并记录损失
            acc = (output.argmax(1).view(-1) == target.float().view(-1)).float().mean() * 100
            losses.update(loss.item(), input.size(0))
            top1.update(acc, input.size(0))
            # 计算耗时
            batch_time.update(time.time() - end)
            end = time.time()

        # TODO: 这也应该用ProgressMeter来做
        print(' * Acc@1 {top1.avg:.3f}'
              .format(top1=top1))
        return top1


# 定义预测过程
def predict(test_loader, model, tta=10):
    # 切换到评估模式
    model.eval()

    test_pred_tta = None
    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            end = time.time()
            for i, (input, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
                input = input.cuda()
                target = target.cuda()

                # 计算输出
                output = model(input)
                output = F.softmax(output, dim=1)
                output = output.data.cpu().numpy()

                test_pred.append(output)
        test_pred = np.vstack(test_pred)

        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta


# 定义训练过程
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, losses, top1)

    # 切换到训练模式
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # 计算输出
        output = model(input)
        loss = criterion(output, target)

        # 记录损失
        losses.update(loss.item(), input.size(0))

        # 计算准确度
        acc = (output.argmax(1).view(-1) == target.float().view(-1)).float().mean() * 100
        top1.update(acc, input.size(0))

        # 计算梯度并执行优化步骤
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算耗时
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            progress.pr2int(i)


# 定义自定义数据集类
class FFDIDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label

        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.from_numpy(np.array(self.img_label[index]))

    def __len__(self):
        return len(self.img_path)


def train_and_validate_with_randaugment(num_ops_values, train_label, val_label, model_name='resnet18', epoch_num=20,
                                        bs_value=256, lr=0.001):
    for num_ops in num_ops_values:
        # 使用timm库创建预训练的ResNet18模型
        model = timm.create_model(model_name, pretrained=True, num_classes=2)
        model = model.cuda()  # 将模型移到GPU上

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)

        # 定义训练数据加载器
        train_loader = torch.utils.data.DataLoader(
            FFDIDataset(train_label['path'][:10000], train_label['target'][:10000],
                        transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.RandAugment(num_ops=num_ops, magnitude=9),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                        ), batch_size=bs_value, shuffle=True, num_workers=8, pin_memory=True
        )

        # 定义验证数据加载器
        val_loader = torch.utils.data.DataLoader(
            FFDIDataset(val_label['path'], val_label['target'],
                        transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                        ), batch_size=bs_value, shuffle=False, num_workers=8, pin_memory=True
        )

        best_acc = 0.0  # 初始化最佳准确度
        best_model_path = None  # 初始化最佳模型路径

        # 训练和验证模型
        for epoch in range(epoch_num):
            print(f'Epoch: {epoch}, num_ops: {num_ops}')

            train(train_loader, model, criterion, optimizer, epoch)  # 训练模型
            scheduler.step()  # 更新学习率
            val_acc = validate(val_loader, model, criterion)  # 验证模型

            if val_acc.avg.item() > best_acc:
                best_acc = round(val_acc.avg.item(), 2)

                # 删除之前的最佳模型
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)

                # 保存新的最佳模型
                best_model_path = f'./model_{best_acc}_RandAugment{num_ops}_9_resnet18.pt'
                torch.save(model.state_dict(), best_model_path)

                print(f'New best model saved with accuracy: {best_acc}, num_ops: {num_ops}')


# 测试不同的num_ops参数
num_ops_values = [1, 2, 3, 4, 5]
train_and_validate_with_randaugment(num_ops_values, train_label, val_label)
train_and_validate_with_randaugment(num_ops_values, train_label, val_label)