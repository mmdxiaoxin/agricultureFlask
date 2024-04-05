import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

'''
    RGB病害分类，ResNet18
'''
class ResNet(nn.Module):
    def __init__(self,class_num):
        super(ResNet, self).__init__()
        self.net = models.resnet18(pretrained=False)
        self.net.fc = nn.Linear(in_features=512, out_features=class_num, bias=True)

    def forward(self,x):
        x = self.net(x)
        x = F.softmax(x, dim=1)
        return x

'''  ==加载==  '''
# 指定.pt文件路径
model_path = './Res_RGB.pt'
# 类别名
class_names = ('褐斑病' ,'斑点落叶病','花叶病','健康','锈病')
# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载模型
model = ResNet(5)  # 实例化模型对象
model.load_state_dict(torch.load(model_path, map_location=device)) # 加载模型参数
model.eval()  # 设置模型为评估模式
img = torch.randn(1, 3, 64, 64) # 第一个1是batch_size，这里随机生成了一个数据
result = model(img) #传入图像返回类别序号
print(class_names[result.argmax(dim=1)])


'''
    高光谱病害分类
'''
class InceptionResBlock(nn.Module):
    def __init__(self,in_channels):
        super(InceptionResBlock,self).__init__()
        self.branch1x1 = nn.Conv3d(1, in_channels, kernel_size=1, stride=(1, 1, 1), padding=0)
        self.branch2_1 = nn.Conv3d(1, 1, kernel_size=1, stride=(1, 1, 1), padding=0)
        self.branch3x3 = nn.Conv3d(1, in_channels, kernel_size=3, stride=(1, 1, 1), padding=1)
        self.branch3_1 = nn.Conv3d(1, 1, kernel_size=1, stride=(1, 1, 1), padding=0)
        self.branch5x5 = nn.Conv3d(1, in_channels, kernel_size=5, stride=(1, 1, 1), padding=2)
        self.conv1x1 = nn.Conv3d(in_channels*3, 1, kernel_size=1, stride=(1, 1, 1), padding=0)
        self.bn = nn.BatchNorm3d(1)
        self.scale = 0.1
        self.init_weights()

    def forward(self,x):
        x1 = self.branch1x1(x)
        x2 = self.branch3x3(self.branch2_1(x))
        x3 = self.branch5x5(self.branch3_1(x))
        out = torch.cat((x1,x2,x3),dim=1)
        out = self.bn(self.conv1x1(out))
        out = x + self.scale * out
        out = F.leaky_relu(out)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class InceptionResBlock_SE(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(InceptionResBlock_SE, self).__init__()
        self.branch1x1 = nn.Conv3d(1, in_channels, kernel_size=1, stride=(1, 1, 1), padding=0)
        self.branch2_1 = nn.Conv3d(1, 1, kernel_size=1, stride=(1, 1, 1), padding=0)
        self.branch3x3 = nn.Conv3d(1, in_channels, kernel_size=3, stride=(1, 1, 1), padding=1)
        self.branch3_1 = nn.Conv3d(1, 1, kernel_size=1, stride=(1, 1, 1), padding=0)
        self.branch5x5 = nn.Conv3d(1, in_channels, kernel_size=5, stride=(1, 1, 1), padding=2)
        self.conv1x1 = nn.Conv3d(in_channels * 3, 1, kernel_size=1, stride=(1, 1, 1), padding=0)
        self.bn = nn.BatchNorm3d(1)
        self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
                nn.LeakyReLU(inplace=True)
            )
        self.scale = 0.1
        self.init_weights()

    def forward(self, x):
        n, _, b, w, h = x.shape
        se_weight = self.se(x.view(n,b,w,h))
        x = x*se_weight.view(n,-1,b,1,1)
        x1 = self.branch1x1(x)
        x2 = self.branch3x3(self.branch2_1(x))
        x3 = self.branch5x5(self.branch3_1(x))
        out = torch.cat((x1, x2, x3), dim=1)
        out = self.bn(self.conv1x1(out))
        out = x + self.scale * out
        out = F.leaky_relu(out)
        return out

class Net2(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net2, self).__init__()
        self.num_classes = num_classes
        self.block1 = InceptionResBlock_SE(in_channels)
        self.block2 = InceptionResBlock_SE(in_channels)
        self.block3 = InceptionResBlock(in_channels)
        self.block4 = InceptionResBlock(in_channels)
        self.pool = nn.AdaptiveAvgPool3d((in_channels,1,1))
        self.fc = nn.Conv2d(in_channels,num_classes,(1,1))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        n,_,b,w,h = x.shape
        x = x.reshape(n,b,w,h)
        x = F.leaky_relu(self.fc(x))
        x = x.view(x.size(0), -1)
        x = F.softmax(x, dim=1)
        return x

'''  ==加载==  '''
# 指定.pt文件路径
model_path = './Net2_69.pt'
# 类别名
class_names = ('花叶病','健康','锈病')
# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载模型
model = Net2(125,3)  # 实例化模型对象
model.load_state_dict(torch.load(model_path, map_location=device)) # 加载模型参数
model.eval()  # 设置模型为评估模式
img = torch.randn(1, 1, 125, 64, 64) # 第一个1是batch_size，这里随机生成了一个数据
result = model(img) #传入图像返回类别序号
print(class_names[result.argmax(dim=1)])