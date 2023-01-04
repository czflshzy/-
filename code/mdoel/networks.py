import torch.nn as nn
from torchvision import transforms, models

class classifer(nn.Module):
    def __init__(self,model_type="resnet18"):
        super(classifer, self).__init__()
        self.model_type = model_type
        if model_type == "resnet18":
            self.model = models.resnet18(pretrained=True)
            self.input_dim = 512
        elif model_type == "mobilenet_v2":
            self.model = models.mobilenet_v2(pretrained=True)
            self.input_dim = 1280
        elif model_type == "resnet50":
            self.model = models.resnet50(pretrained=True)
            self.input_dim = 2048
        elif model_type == "regnet_x_400mf":
            self.model = models.regnet_x_400mf()
            self.input_dim = 400
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # 去掉网络的最后原始分类头
        # 当前任务的分类头 替换掉原始分类头
        self.fc_head = nn.Linear(self.input_dim,2)
        # 分类头参数初始化
        nn.init.xavier_uniform_(self.fc_head.weight)

    def forward(self,images):
        features = self.model(images) # CNN提取特征
        features = features.view(features.size(0), -1)  # 以resnet为例 [B,512,1,1] -> [B,512]
        x = self.fc_head(features) # 分类头进行预测
        return x

