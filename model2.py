#################所有融合：超网络+情感+颜色########
import torch as torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import math
import torch.utils.model_zoo as model_zoo
from numpy.lib.function_base import place
import torchvision as tv
import torchvision.models as models
import os
import numpy as np
import networks
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class HyperNet(nn.Module):
    """
    Hyper network for learning perceptual rules.

    Args:
        lda_out_channels: local distortion aware module output size.
        hyper_in_channels: input feature channels for hyper network.
        target_in_size: input vector size for target network.
        target_fc(i)_size: fully connection layer size of target network.
        feature_size: input feature map width/height for hyper network.

    Note:
        For size match, input args must satisfy: 'target_fc(i)_size * target_fc(i+1)_size' is divisible by 'feature_size ^ 2'.

    """
    def __init__(self, lda_out_channels, hyper_in_channels, target_in_size, target_fc1_size, target_fc2_size, target_fc3_size, target_fc4_size, feature_size):
        super(HyperNet, self).__init__()
        # best_state = torch.load('/myDockerShare/zys/aes-emotion/experiment/resnet-emotion.pth')
        self.hyperInChn = hyper_in_channels
        self.target_in_size = target_in_size
        self.f1 = target_fc1_size
        self.f2 = target_fc2_size
        self.f3 = target_fc3_size
        self.f4 = target_fc4_size
        self.feature_size = feature_size

        self.res = resnet50_backbone(lda_out_channels, target_in_size, pretrained=True)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Conv layers for resnet output features
        self.conv1 = nn.Sequential(
            nn.Conv2d(2048, 1024, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, self.hyperInChn, 1, padding=(0, 0)),
            nn.ReLU(inplace=True)
        )

        # Hyper network part, conv for generating target fc weights, fc for generating target fc biases
        self.fc1w_conv = nn.Conv2d(self.hyperInChn, int(self.target_in_size * self.f1 / feature_size ** 2), 3,  padding=(1, 1))
        self.fc1b_fc = nn.Linear(self.hyperInChn, self.f1)

        self.fc2w_conv = nn.Conv2d(self.hyperInChn, int(self.f1 * self.f2 / feature_size ** 2), 3, padding=(1, 1))
        self.fc2b_fc = nn.Linear(self.hyperInChn, self.f2)

        self.fc3w_conv = nn.Conv2d(self.hyperInChn, int(self.f2 * self.f3 / feature_size ** 2), 3, padding=(1, 1))
        self.fc3b_fc = nn.Linear(self.hyperInChn, self.f3)

        self.fc4w_conv = nn.Conv2d(self.hyperInChn, int(self.f3 * self.f4 / feature_size ** 2), 3, padding=(1, 1))
        self.fc4b_fc = nn.Linear(self.hyperInChn, self.f4)

        self.fc5w_fc = nn.Linear(self.hyperInChn, self.f4)
        self.fc5b_fc = nn.Linear(self.hyperInChn, 1)

        # initialize
        # for i, m_name in enumerate(self._modules):
        #     if i > 2:
        #         nn.init.kaiming_normal_(self._modules[m_name].weight.data)

    def forward(self, img, x1):
        feature_size = self.feature_size

        res_out = self.res(img,x1)
        # emotion = self.emotion(img)
        # input vector for target net
        target_in_vec = res_out['target_in_vec'].view(-1, self.target_in_size, 1, 1)
        # print('target_in_vec',target_in_vec.shape)

        # input features for hyper net
        hyper_in_feat = self.conv1(res_out['hyper_in_feat']).view(-1, self.hyperInChn, feature_size, feature_size)
        #hyper_in_feat:  torch.Size([32, 112, 7, 7])

        # generating target net weights & biases
        target_fc1w = self.fc1w_conv(hyper_in_feat).view(-1, self.f1, self.target_in_size, 1, 1)
        target_fc1b = self.fc1b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f1)

        target_fc2w = self.fc2w_conv(hyper_in_feat).view(-1, self.f2, self.f1, 1, 1)
        target_fc2b = self.fc2b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f2)

        target_fc3w = self.fc3w_conv(hyper_in_feat).view(-1, self.f3, self.f2, 1, 1)
        target_fc3b = self.fc3b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f3)

        target_fc4w = self.fc4w_conv(hyper_in_feat).view(-1, self.f4, self.f3, 1, 1)
        target_fc4b = self.fc4b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f4)

        target_fc5w = self.fc5w_fc(self.pool(hyper_in_feat).squeeze()).view(-1, 1, self.f4, 1, 1)
        target_fc5b = self.fc5b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, 1)
    

        out = {}
        out['target_in_vec'] = target_in_vec       # target_in_vec torch.Size([32, 224, 1, 1])
        out['target_fc1w'] = target_fc1w
        # print('target_fc1w',out['target_fc1w'].shape) #target_fc1w torch.Size([32, 112, 224, 1, 1])
        out['target_fc1b'] = target_fc1b 
        # print('target_fc1b',out['target_fc1b'].shape)  #target_fc1b torch.Size([32, 112])
        out['target_fc2w'] = target_fc2w
        # print('target_fc2w',out['target_fc2w'].shape)  #target_fc2w torch.Size([32, 56, 112, 1, 1])
        out['target_fc2b'] = target_fc2b
        # print('target_fc2b',out['target_fc2b'].shape)  #target_fc2b torch.Size([32, 56])
        out['target_fc3w'] = target_fc3w
        # print('target_fc3w',out['target_fc3w'].shape)
        out['target_fc3b'] = target_fc3b
        # print('target_fc3b',out['target_fc3b'].shape)
        out['target_fc4w'] = target_fc4w
        # print('target_fc4w',out['target_fc4w'].shape)
        out['target_fc4b'] = target_fc4b
        # print('target_fc4b',out['target_fc4b'].shape)
        out['target_fc5w'] = target_fc5w
        # print('target_fc5w',out['target_fc5w'].shape)
        out['target_fc5b'] = target_fc5b
        # print('target_fc5b',out['target_fc5b'].shape)

        return out
# target_in_vec torch.Size([32, 224, 1, 1])
# target_fc1w torch.Size([32, 112, 224, 1, 1])
# target_fc1b torch.Size([32, 112])
# target_fc2w torch.Size([32, 56, 112, 1, 1])
# target_fc2b torch.Size([32, 56])
# target_fc3w torch.Size([32, 28, 56, 1, 1])
# target_fc3b torch.Size([32, 28])
# target_fc4w torch.Size([32, 14, 28, 1, 1])
# target_fc4b torch.Size([32, 14])
# target_fc5w torch.Size([32, 1, 14, 1, 1])
# target_fc5b torch.Size([32, 1])

class TargetNet(nn.Module):
    """
    Target network for quality prediction.
    """
    def __init__(self, paras):
        super(TargetNet, self).__init__()
        self.l1 = nn.Sequential(
            TargetFC(paras['target_fc1w'], paras['target_fc1b']),
            nn.Sigmoid(),
        )
        self.l2 = nn.Sequential(
            TargetFC(paras['target_fc2w'], paras['target_fc2b']),
            nn.Sigmoid(),
        )

        self.l3 = nn.Sequential(
            TargetFC(paras['target_fc3w'], paras['target_fc3b']),
            nn.Sigmoid(),
        )

        self.l4 = nn.Sequential(
            TargetFC(paras['target_fc4w'], paras['target_fc4b']),
            nn.Sigmoid(),
            TargetFC(paras['target_fc5w'], paras['target_fc5b']),
        )

    def forward(self, x):
        q = self.l1(x)
        # print('f1:',q.shape)                #f1: torch.Size([32, 112, 1, 1])
        # q = F.dropout(q)                          
        q = self.l2(q)
        # print('f2:',q.shape)                   #f2: torch.Size([32, 56, 1, 1])
        q = self.l3(q)
        # print('f3:',q.shape)                   #f3: torch.Size([32, 28, 1, 1])
        q = self.l4(q).squeeze()
        # print('f4:',q.shape)                   #f4: torch.Size([32])
        return q                                        
                                           
      


class TargetFC(nn.Module):
    """
    Fully connection operations for target net

    Note:
        Weights & biases are different for different images in a batch,
        thus here we use group convolution for calculating images in a batch with individual weights & biases.
    """
    def __init__(self, weight, bias):
        super(TargetFC, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, input_):

        input_re = input_.view(-1, input_.shape[0] * input_.shape[1], input_.shape[2], input_.shape[3])
        weight_re = self.weight.view(self.weight.shape[0] * self.weight.shape[1], self.weight.shape[2], self.weight.shape[3], self.weight.shape[4])
        bias_re = self.bias.view(self.bias.shape[0] * self.bias.shape[1])
        out = F.conv2d(input=input_re, weight=weight_re, bias=bias_re, groups=self.weight.shape[0])

        return out.view(input_.shape[0], self.weight.shape[1], input_.shape[2], input_.shape[3])


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBackbone(nn.Module):

    def __init__(self, lda_out_channels, in_chn, block, layers, num_classes=1000):
        super(ResNetBackbone, self).__init__()
        predict_model = torch.load('/myDockerShare/zys/aes-emotion/experiment/color3.pth')
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # local distortion aware module
        self.lda1_pool = nn.Sequential(
            nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda1_fc = nn.Linear(16 * 64, lda_out_channels)

        self.lda2_pool = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda2_fc = nn.Linear(32 * 16, lda_out_channels)

        self.lda3_pool = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda3_fc = nn.Linear(64 * 4, lda_out_channels)

        self.lda4_pool = nn.AvgPool2d(7, stride=7)
        self.lda4_fc = nn.Linear(2048, in_chn - lda_out_channels * 3)
        self.fc_5 = nn.Linear(2528, 224)
        self.emotion=create_model('resnet50',0.75)
        emotion_dict = self.emotion.state_dict()
        state_dict = {k: v for k, v in predict_model["state_dict"].items() if k in emotion_dict.keys()}
        emotion_dict.update(state_dict) 
        self.emotion.load_state_dict(emotion_dict)
        # self.conv=nn.Conv2d(4096,2048, kernel_size=1, padding=0, bias=False)  
        for param in self.emotion.parameters():
                param.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # initialize
        nn.init.kaiming_normal_(self.lda1_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lda2_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lda3_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lda1_fc.weight.data)
        nn.init.kaiming_normal_(self.lda2_fc.weight.data)
        nn.init.kaiming_normal_(self.lda3_fc.weight.data)
        nn.init.kaiming_normal_(self.lda4_fc.weight.data)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x , x1):
        x1=self.emotion(x,x1)        #x1: torch.Size([32, 2112])
        # print('x1:',x1.shape)
        # print('x1',x1.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        # the same effect as lda operation in the paper, but save much more memory
        lda_1 = self.lda1_fc(self.lda1_pool(x).view(x.size(0), -1))
        x = self.layer2(x)
        lda_2 = self.lda2_fc(self.lda2_pool(x).view(x.size(0), -1))
        x = self.layer3(x)
        lda_3 = self.lda3_fc(self.lda3_pool(x).view(x.size(0), -1))
        x = self.layer4(x)          #x: torch.Size([32, 2048, 7, 7])
        # x= self.conv(x)
        # print('x:',x.shape)
        # print('hyper:',x.shape)
        lda_4 = self.lda4_fc(self.lda4_pool(x).view(x.size(0), -1))  #lda_4: torch.Size([32, 176])
        # print('lda_4:',lda_4.shape)
        # x1= self.lda4_fc(self.lda4_pool(x1).view(x1.size(0), -1))
        vec = torch.cat((lda_1, lda_2, lda_3, lda_4, x1), 1)       #vec:vec: torch.Size([32, 2336])
        # print('vec:',vec.shape)
        vec =self.fc_5(vec)
        # print('vec:',vec.shape)
        # print('target_in::',vec.shape)
        out = {}
        out['hyper_in_feat'] = x
        out['target_in_vec'] = vec

        return out


def resnet50_backbone(lda_out_channels, in_chn, pretrained=True, **kwargs):
    """Constructs a ResNet-50 model_hyper.

    Args:
        pretrained (bool): If True, returns a model_hyper pre-trained on ImageNet
    """
    model = ResNetBackbone(lda_out_channels, in_chn, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        save_model = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    else:
        model.apply(weights_init_xavier)
    return model


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    # if isinstance(m, nn.Conv2d):
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)




MODELS = {
    "resnet18": (tv.models.resnet18, 512),
    "resnet34": (tv.models.resnet34, 512),
    "resnet50": (tv.models.resnet50, 2048),
    "resnet101": (tv.models.resnet101, 2048),
    "resnet152": (tv.models.resnet152, 2048),
}


class NIMA(nn.Module):
    def __init__(self, base_model: nn.Module, input_features: int, drop_out: float):
        super(NIMA, self).__init__()

        self.base_model = base_model

        self.head = nn.Sequential(
            nn.ReLU(inplace=True), nn.Dropout(p=drop_out), nn.Linear(2304, 8), nn.Softmax(dim=1)
        )
        self.hist_l = 8
        self.hist_ab = 64
        self.img_type = 'lab'
        self.pad = 30
        self.reppad = nn.ReplicationPad2d(self.pad)
        self.hen=self.hen=HEN((self.hist_l+1), 256)

  
    def forward(self, x, x1):
      
        x = self.base_model(x)          #torch.Size([32, 2048, 1, 1])
        # print('x:',x.shape)
        x = x.view(x.size(0), -1)       #torch.Size([32, 2048])
        # print('x:',x.shape)
        hist_inp_ab=getHistogram2d_np(x1,self.hist_ab)
        hist_inp_l=getHistogram1d_np(x1, self.hist_l).repeat(1,1,self.hist_ab,self.hist_ab)
        hist_inp = torch.cat((hist_inp_ab,hist_inp_l),1)
        # print('lab:',hist_inp.shape)
        x1=  self.hen(hist_inp)
        # print('x1',x1.shape)
        # x1 = x1.view(x1.size(0), -1)
        # print('x1',x1.shape)
        x=torch.cat((x,x1), 1)
        # print('cat',x.shape)
        output = self.head(x)
        return x
 
def HEN(input_nc, output_nc):
    HEN = None

    HEN = ConditionNetwork2(input_nc, output_nc)
    

    return HEN


class ConditionNetwork2(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(ConditionNetwork2, self).__init__()
        self.input_nc = input_nc # 10
        self.output_nc = output_nc # 32
  
  

        model = [
                
                nn.Conv2d(self.input_nc, 128, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                nn.ReLU(inplace=True),

                nn.Conv2d(128, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1,1),bias=False),
                nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(output_size=(1, 1))  
        ]

        self.model = nn.Sequential(*model)

        self.model2 = nn.Sequential(nn.Linear(1024,output_nc))


    def forward(self, input):

        # Ver.2
        a1 = self.model(input)
        a2 = a1.view(a1.size(0),-1)
        a3 = self.model2(a2)
    

        return a3
def getHistogram2d_np( img_torch, num_bin):
        # arr = img_torch.detach().cpu().numpy()
        arr=img_torch.cpu().numpy()
        H_=[]
        # print('shape:',arr.shape[0])
        for i in range(arr.shape[0]):
            a=arr[i]
            # print(a)
            arr_ = a[np.newaxis, :]
            # print('arr',arr_.shape)
        # Exclude Zeros and Make value 0 ~ 1
            arr1 = ( arr_[0][1].ravel()[np.flatnonzero(arr_[0][1])] + 1 ) /2 
            arr2 = ( arr_[0][2].ravel()[np.flatnonzero(arr_[0][2])] + 1 ) /2 
            if (arr1.shape[0] != arr2.shape[0]):
                if arr2.shape[0] < arr1.shape[0]:
                    arr2 = np.concatenate([arr2, np.array([0])])
                else:
                    arr1 = np.concatenate([arr1, np.array([0])])
        # # AB space
            arr_new = [arr1, arr2]

            H,edges = np.histogramdd(arr_new, bins = [num_bin, num_bin], range = ((0,1),(0,1)))
            # print('H:',H)
            H = np.rot90(H)
            H = np.flip(H,0)

            H_torch = torch.from_numpy(H).float().cuda()
            H_torch = H_torch.unsqueeze(0).unsqueeze(0)
       
            # Normalize
            total_num = sum(sum(H_torch.squeeze(0).squeeze(0))) # 256 * 256 => same value as arr[0][0].ravel()[np.flatnonzero(arr[0][0])].shape
            H_torch = H_torch / total_num
            H_torch= H_torch .squeeze(axis=0)
            # print('H_torch:',H_torch)
            H_.append(H_torch)
        H_ab=torch.stack(H_,0)
        return   H_ab

def getHistogram1d_np(img_torch, num_bin): # L space # Idon't know why but they(np, conv) are not exactly same
        # Preprocess
        # arr = img_torch.detach().cpu().numpy()
        arr=img_torch.cpu().numpy()
        H_1=[]
        for i in range(arr.shape[0]):
            a=arr[i]
            arr_ = a[np.newaxis, :]
            arr0 = ( arr_[0][0].ravel()[np.flatnonzero(arr_[0][0])] + 1 ) / 2 
            arr1 = np.zeros(arr0.size)

            arr_new = [arr0, arr1]
            H, edges = np.histogramdd(arr_new, bins = [num_bin, 1], range =((0,1),(-1,2)))
            # H_torch = torch.from_numpy(H).float().cuda() #10/224/224
            H_torch = torch.from_numpy(H).float().cuda()
            H_torch = H_torch.unsqueeze(0).unsqueeze(0).permute(0,2,1,3)

            total_num = sum(sum(H_torch.squeeze(0).squeeze(0))) # 256 * 256 => same value as arr[0][0].ravel()[np.flatnonzero(arr[0][0])].shape
            H_torch = H_torch / total_num
            H_torch= H_torch .squeeze(axis=0)
            # print('H_l:',H_torch.shape)
            H_1.append(H_torch)
            H_l=torch.stack(H_1,0)
            # print('H_l:',H_l.shape)
        return H_l
 

def create_model(model_type: str, drop_out: float) -> NIMA:
    create_function, input_features = MODELS[model_type]
    base_model = create_function(pretrained=True)
    base_model = nn.Sequential(*list(base_model.children())[:-1])
    return NIMA(base_model=base_model, input_features=input_features, drop_out=drop_out)

if __name__=='__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    x=torch.rand(32,3,224,224)
    x1=torch.rand(32,3,224,224)
    model_hyper=HyperNet(16, 112, 224, 112, 56, 28, 14, 7)
    paras =model_hyper(x,x1)  
    model_target =TargetNet(paras)
    for param in model_target.parameters():
        param.requires_grad = False
                # Quality prediction
    pred = model_target(paras['target_in_vec']) 
    print('ok')