# [全球Deepfake攻防挑战赛（图像赛道）](https://www.atecup.cn/deepfake)  DataWhale AI夏令营 Task1 学习笔记

# 1 什么是Deepfake？

## 1.1 Deepfake介绍

Deepfake技术是一种利用人工智能和机器学习算法创建高度逼真的合成音频、视频或图像内容的技术。其核心是使用深度学习，特别是生成对抗网络（GANs）、扩散模型（Stable Diffusion），来模仿和生成与真实数据几乎无法区分的虚假内容。以下是deepfake技术的一些关键方面：

1. **生成对抗网络（GANs）**：deepfake通常使用生成对抗网络。GANs包括两个部分：生成器和判别器。生成器负责创建虚假的图像或视频，而判别器则评估这些内容的真实性。两者在训练过程中相互竞争，生成器不断改进其输出，使得生成的内容越来越逼真。
2. **人脸交换**：最常见的deepfake应用是人脸交换，即将一个人的面部表情和动作映射到另一个人的视频中。这可以用于创建虚假的视频片段，使得看起来像是某人说了或做了某些事情。
3. **语音合成**：deepfake技术也可以用于语音合成，即模拟某人的声音。这可以用于制作仿冒语音记录，甚至是实时的语音模仿。
4. **图像和视频编辑**：deepfake技术不仅可以交换人脸，还可以编辑图像和视频的其他方面，比如改变背景、增加或删除物体等。

## 1.2 Deepfake的应用和挑战

Deepfake技术可以应用到各行各业，以下是简单举例：

- **教育和培训**：通过deepfake技术创建虚拟讲师，提供个性化教学或培训。
- **艺术创作**：艺术家可以利用deepfake技术创造新的艺术形式，如动态肖像和数字雕塑。音乐家可以用deepfake技术创造独特的音乐视频效果。
- **语言学习**：创建虚拟对话伙伴，帮助学习者练习发音和对话技巧。模拟特定文化环境，提供真实的语言和文化学习体验。

但是Deepfake技术也存在着风险和挑战：

- **误导和欺骗**：deepfake技术可能被用于创建虚假的新闻或伪造证据，误导公众和法律机构。
- **隐私和道德问题**：未经同意制作和传播他人的deepfake内容可能侵犯隐私和名誉权。
- **技术滥用**：deepfake可能被用于恶意目的，如网络欺凌、诈骗和政治操纵。

## 1.3 Deepfake人工分辨方法

人工识别Deepfake图片，可以通过一下逻辑步骤进行：

1. **观察图片细节：**真实图片的人物面部特征更加自然，重点观察**眼睛**、**嘴唇**、**牙齿**、**皮肤**等重要的特征区域。

   - **眼睛**：Deepfake生成的眼睛有时会看起来不自然，可能没有焦点或光线反射不一致。
   - **嘴唇和牙齿**：嘴唇和牙齿的运动可能不协调，尤其是在说话或微笑时。
   - **皮肤纹理**：皮肤的质感和细节可能不一致，有时会出现模糊或过于光滑的区域。

2. **光照和阴影**：检查图像中光线的方向和强度是否一致，Deepfake有时会出现不一致的光照效果。

3. **分析像素**：Deepfake图片中可能会存在局部模糊或者像素化的问题。

4. **检查背景和边缘细节：**

   - **背景与人物的融合**：Deepfake有时在人物与背景的过渡部分处理不好，可能会出现模糊或异常边缘。

   - **头发和边缘细节**：头发的细节尤其难以生成，可能会出现模糊或不自然的现象。

# 2 赛题介绍

## 2.1 赛题任务

随着人工智能技术的迅猛发展，深度伪造技术（Deepfake）正成为数字世界中的一把双刃剑。这项技术不仅为创意内容的生成提供了新的可能性，同时也对数字安全构成了前所未有的挑战。Deepfake技术可以通过人工智能算法生成高度逼真的图像、视频和音频内容，这些内容看起来与真实的毫无二致。然而，这也意味着虚假信息、欺诈行为和隐私侵害等问题变得更加严重和复杂。

在这个赛道中，比赛任务是判断一张人脸图像是否为Deepfake图像，并输出其为Deepfake图像的概率评分。参赛者需要开发和优化检测模型，以应对多样化的Deepfake生成技术和复杂的应用场景，从而提升Deepfake图像检测的准确性和鲁棒性。

## 2.2 赛题数据集

#### 第一阶段
在第一阶段，主办方将发布训练集和验证集。参赛者将使用训练集 (train_label.txt) 来训练模型，而验证集 (val_label.txt) 仅用于模型调优。文件的每一行包含两个部分，分别是图片文件名和标签值（label=1 表示Deepfake图像，label=0 表示真实人脸图像）。例如：

**train_label.txt**

```
img_name,target
3381ccbc4df9e7778b720d53a2987014.jpg,1
63fee8a89581307c0b4fd05a48e0ff79.jpg,0
7eb4553a58ab5a05ba59b40725c903fd.jpg,0
…
```

**val_label.txt**

```
img_name,target
cd0e3907b3312f6046b98187fc25f9c7.jpg,1
aa92be19d0adf91a641301cfcce71e8a.jpg,0
5413a0b706d33ed0208e2e4e2cacaa06.jpg,0
…
```

#### 第二阶段
在第一阶段结束后，主办方将发布测试集。在第二阶段，参赛者需要在系统中提交测试集的预测评分文件 (prediction.txt)，主办方将在线反馈测试评分结果。文件的每一行包含两个部分，分别是图片文件名和模型预测的Deepfake评分（即样本属于Deepfake图像的概率值）。例如：

**prediction.txt**

```
img_name,y_pred
cd0e3907b3312f6046b98187fc25f9c7.jpg,1
aa92be19d0adf91a641301cfcce71e8a.jpg,0.5
5413a0b706d33ed0208e2e4e2cacaa06.jpg,0.5
…
```

#### 第三阶段
在第二阶段结束后，前30名队伍将晋级到第三阶段。在这一阶段，参赛者需要提交代码docker和技术报告。Docker要求包括原始训练代码和测试API（函数输入为图像路径，输出为模型预测的Deepfake评分）。主办方将检查并重新运行算法代码，以重现训练过程和测试结果。


## 2.3 评价指标


#### 评估指标
比赛的性能评估主要使用ROC曲线下的AUC（Area under the ROC Curve）作为指标。AUC的取值范围通常在0.5到1之间。若AUC指标不能区分排名，则会使用TPR@FPR=1E-3作为辅助参考。

**相关公式：**

> 真阳性率 (TPR)：
>
> TPR = TP / (TP + FN)
>
> 假阳性率 (FPR)：
>
> FPR = FP / (FP + TN)
>
> 其中：
> - TP：攻击样本被正确识别为攻击；
> - TN：真实样本被正确识别为真实；
> - FP：真实样本被错误识别为攻击；
> - FN：攻击样本被错误识别为真实。

参考文献：[Aghajan, H., Augusto, J. C., & Delgado, R. L. C. (Eds.). (2009). Human-centric interfaces for ambient intelligence. Academic Press.](https://books.google.com/books?hl=zh-CN&lr=&id=64icBAAAQBAJ&oi=fnd&pg=PP1&dq=Human-centric+interfaces+for+ambient+intelligence&ots=mKNsJrymuK&sig=_ZrNLwqT9R6BDddTLy02FF1B3WE)

# 3 Code

## 3.1 导入库

```python
import torch
torch.manual_seed(0)  # 设置随机种子以确保结果可重复
torch.backends.cudnn.deterministic = False  # 允许某些非确定性操作，以提高训练速度
torch.backends.cudnn.benchmark = True  # 允许CUDNN自动寻找最优的算法，以提高训练速度

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import timm  # 提供了大量的预训练模型
import time

import pandas as pd
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm_notebook  # 用于显示训练和验证过程的进度条
```
我们的模型使用pytorch进行构建，因此需要导入相关的库。其中，`torch`是PyTorch的主要库，`torchvision`提供了一些视觉处理工具，`torch.optim`提供了优化算法，`torch.utils.data`提供了数据加载和处理的工具，`torch.autograd`提供了自动求导功能。`pandas`和`numpy`用于数据处理，`cv2`和`PIL`用于图像处理，`timm`提供了大量的预训练模型。`tqdm_notebook`用于显示训练和验证过程的进度条。

## 3.2 定义计算平均指标和显示进度的类
```python
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
```
`AverageMeter`类用于计算并存储平均值和当前值，`ProgressMeter`类用于显示训练和验证过程的进度。

## 3.3 定义数据加载类
```python
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
```
`FFDIDataset`类用于加载数据集。`__init__`方法用于初始化数据集，`__getitem__`方法用于获取数据集中的样本，`__len__`方法用于获取数据集的长度。
`__init__`方法接受`img_path`和`img_label`作为输入，`transform`参数用于对图像进行预处理。

## 3.4 定义训练、验证和预测函数
这一部分定义了训练、验证和预测的函数，包括`train`、`validate`和`predict`函数。这些函数用于训练模型、验证模型和预测测试集的结果。
```python
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
        for i, (input, target) in tqdm_notebook(enumerate(val_loader), total=len(val_loader)):
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
            for i, (input, target) in tqdm_notebook(enumerate(test_loader), total=len(test_loader)):
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
```

## 3.5 定义模型
```python
import timm
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
model = model.cuda()
```
使用`timm`库提供的`create_model`函数创建一个预训练的EfficientNet模型，并将其移动到GPU上。`timm`是一个提供了大量预训练模型的库，可以方便地使用各种模型进行训练和预测。更多信息可以访问[timm GitHub](https://github.com/huggingface/pytorch-image-models)。

## 3.6 定义数据加载和预处理
```python
# 读取训练和验证数据的标签文件
train_label = pd.read_csv('/kaggle/input/deepfake/phase1/trainset_label.txt')
val_label = pd.read_csv('/kaggle/input/deepfake/phase1/valset_label.txt')

# 生成训练和验证数据的路径
train_label['path'] = '/kaggle/input/deepfake/phase1/trainset/' + train_label['img_name']
val_label['path'] = '/kaggle/input/deepfake/phase1/valset/' + val_label['img_name']

bs_value = 32  # 设置批次大小

# 定义训练数据加载器
train_loader = torch.utils.data.DataLoader(
    FFDIDataset(train_label['path'], train_label['target'],
                transforms.Compose([
                    transforms.Resize((256, 256)),  # 调整图像大小
                    transforms.RandAugment(num_ops=2, magnitude=9),  # 随机增强
                    transforms.ToTensor(),  # 转换为张量
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
                ])
                ), batch_size=bs_value, shuffle=True, num_workers=4, pin_memory=True
)

# 定义验证数据加载器
val_loader = torch.utils.data.DataLoader(
    FFDIDataset(val_label['path'], val_label['target'],
                transforms.Compose([
                    transforms.Resize((256, 256)),  # 调整图像大小
                    transforms.ToTensor(),  # 转换为张量
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
                ])
                ), batch_size=bs_value, shuffle=False, num_workers=4, pin_memory=True
)
```
这一部分定义了数据加载和预处理的过程。首先，读取训练和验证数据的标签文件，然后生成训练和验证数据的路径。接着，定义了训练和验证数据加载器，使用`torch.utils.data.DataLoader`加载数据集，并对图像进行预处理。

## 3.7 定义损失函数和优化器
```python

epoch_num = 10  # 设置训练的总轮数


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), 0.005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.85)  # 学习率调度器
best_acc = 0.0  # 初始化最佳准确度
```
这一部分定义了损失函数、优化器和学习率调度器。`nn.CrossEntropyLoss`用于计算交叉熵损失，`torch.optim.Adam`用于执行Adam优化算法，`optim.lr_scheduler.StepLR`用于调整学习率。

## 3.8 训练模型
```python
# 训练和验证模型
for epoch in range(epoch_num):
    scheduler.step()  # 更新学习率
    print('Epoch: ', epoch)

    train(train_loader, model, criterion, optimizer, epoch)  # 训练模型
    val_acc = validate(val_loader, model, criterion)  #
    # 验证模型
    if val_acc.avg.item() > best_acc:
        best_acc = round(val_acc.avg.item(), 2)
        torch.save(model.state_dict(), f'./model_{best_acc}.pt')  # 保存表现更好的模型
```
## 3.9 预测测试集
```python
# 准备测试数据加载器
test_loader = torch.utils.data.DataLoader(
    FFDIDataset(val_label['path'], val_label['target'],
            transforms.Compose([
                        transforms.Resize((256, 256)),  # 调整图像大小
                        transforms.ToTensor(),  # 转换为张量
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
        ])
    ), batch_size=bs_value, shuffle=False, num_workers=4, pin_memory=True
)

# 进行预测
val_label['y_pred'] = predict(test_loader, model, 1)[:, 1]

# 将预测结果保存为CSV文件
val_label[['img_name', 'y_pred']].to_csv('submit.csv', index=None)
```
这一部分定义了预测测试集的过程。首先，准备测试数据加载器，然后使用`predict`函数对测试集进行预测。最后，将预测结果保存为CSV文件。

# 4 总结
本文介绍了全球Deepfake攻防挑战赛（图像赛道）Task1的学习笔记，包括Deepfake技术的介绍、赛题介绍、评价指标、代码实现等内容。通过学习和实践，我们可以更好地理解Deepfake技术的原理和应用，提高对Deepfake图像的识别和检测能力。希望本文对大家有所帮助，欢迎大家参加全球Deepfake攻防挑战赛，共同探索Deepfake技术的发展和应用。
## 4.1 可以优化的点
- 可以尝试使用更多的预训练模型，如ViT、Swin Transformer等，以提高模型的性能和泛化能力。
- 可以尝试使用更多的数据增强技术，如Mixup、CutMix等，以提高模型的鲁棒性和泛化能力。
- 训练数据中数据不平衡，可以尝试使用类别平衡技术，如过采样、欠采样等，以提高模型的性能和泛化能力。

## 4.2 参考资料
- [全球Deepfake攻防挑战赛（图像赛道）](https://www.atecup.cn/deepfake)
- [Kaggle[九月0.98]Deepfake-FFDI-ways to defeat 0.8696](https://www.kaggle.com/code/chg0901/0-98-deepfake-ffdi-ways-to-defeat-0-8696)

