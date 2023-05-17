# ##可视化1
# import cv2
# import numpy as np
# from sympy import im
# import torch
# from torch.autograd import Variable
# from torchvision import models
# import os

# # 该函数创建保存特征图的文件目录,以网络层号命名文件夹，如feature\\1\\..文件夹中保存的是模型第二层的输出特征图
# def mkdir(path):

#     isExists = os.path.exists(path) # 判断路径是否存在，若存在则返回True，若不存在则返回False
#     if not isExists: # 如果不存在则创建目录
#         os.makedirs(path)
#         return True
#     else:
#         return False

# # 图像预处理函数，将图像转换成[224,224]大小,并进行Normalize，返回[1,3,224,224]的四维张量
# def preprocess_image(cv2im, resize_im=True):

#     # 在ImageNet100万张图像上计算得到的图像的均值和标准差，它会使图像像素值大小在[-2.7,2.1]之间，但是整体图像像素值的分布会是标准正态分布（均值为0，方差为1）
#     # 之所以使用这种方法，是因为这是基于ImageNet的预训练VGG16对输入图像的要求
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]

#     # 改变图像大小并进行Normalize
#     if resize_im:
#         cv2im = cv2.resize(cv2im, dsize=(224,224),interpolation=cv2.INTER_CUBIC)
#     im_as_arr = np.float32(cv2im)
#     im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
#     im_as_arr = im_as_arr.transpose(2, 0, 1)  # 将[W,H,C]的次序改变为[C,W,H]

#     for channel, _ in enumerate(im_as_arr): # 进行在ImageNet上预训练的VGG16要求的ImageNet输入图像的Normalize
#         im_as_arr[channel] /= 255
#         im_as_arr[channel] -= mean[channel]
#         im_as_arr[channel] /= std[channel]

#     # 转变为三维Tensor,[C,W,H]
#     im_as_ten = torch.from_numpy(im_as_arr).float()
#     im_as_ten = im_as_ten.unsqueeze_(0) # 扩充为四维Tensor,变为[1,C,W,H]

#     return im_as_ten # 返回处理好的[1,3,224,224]四维Tensor


# class FeatureVisualization():

#     def __init__(self,img_path,selected_layer):
#         '''
#         :param img_path:  输入图像的路径
#         :param selected_layer: 待可视化的网络层的序号
#         '''
#         self.img_path = img_path
#         self.selected_layer = selected_layer
#         self.pretrained_model = models.vgg16(pretrained=True).features # 调用预训练好的vgg16模型

#     def process_image(self):
#         img = cv2.imread(self.img_path)
#         img = preprocess_image(img)
#         return img

#     def get_feature(self):

#         input=self.process_image() # 读取输入图像
#         # 以下是关键代码：根据给定的层序号，返回该层的输出
#         x = input
#         for index, layer in enumerate(self.pretrained_model):
#             x = layer(x) # 将输入给到模型各层，注意第一层的输出要作为第二层的输入，所以才会复用x

#             if (index == self.selected_layer): # 如果模型各层的索引序号等于期望可视化的选定层号
#                 return x # 返回模型当前层的输出四维特征图

#     def get_single_feature(self):
#         features = self.get_feature() # 得到期望模型层的输出四维特征图
#         return features

#     def save_feature_to_img(self):

#         features=self.get_single_feature() # 返回一个指定层输出的特征图,属于四维张量[batch,channel,width,height]
#         for i in range(features.shape[1]):
#             feature = features[:, i, :, :] # 在channel维度上，每个channel代表了一个卷积核的输出特征图，所以对每个channel的图像分别进行处理和保存
#             feature = feature.data.numpy()
#             heatmap = np.maximum(feature, 0)        # heatmap与0比较
#             heatmap = np.mean(heatmap, axis=0)  # 多通道时，取均值
#             heatmap /= np.max(heatmap) 
     
#         return heatmap

# if __name__=='__main__':
#     img1='/myDockerShare/zys/ava/emotion_data/pic/2.jpg'
#     img=cv2.imread(img1)
    
#     for k in range(1): # k代表选定的可视化的层的序号
#         myClass = FeatureVisualization(img1, 10) # 实例化类
#         print (myClass.pretrained_model)
#         heatmap=myClass.save_feature_to_img() # 开始可视化，并将特征图保存成图像
#     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 特征图的大小调整为与原始图像相同
#     heatmap = np.uint8(255 * heatmap)  # 将特征图转换为uint8格式
#     heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)  # 将特征图转为伪彩色图
#     heat_img = cv2.addWeighted(img, 1, heatmap, 0.5, 0)     # 将伪彩色图与原始图片融合
#     #heat_img = heatmap * 0.5 + img 　　　　　　		  　　　 # 也可以用这种方式融合
#     cv2.imwrite('/myDockerShare/zys/aes-emotion/cam/heat10.jpg', heat_img)# 将图像保存





# #cam可视化
# #CAM相关方法：Grad-CAM: https://arxiv.org/pdf/1610.02391.pdf、Grad-CAM++: https://arxiv.org/pdf/1610.02391.pdf
# import os
# import torch
# import torchvision.models as models
# import torchvision.transforms as transforms
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# import cv2
# import model_cam
# os.environ["KMP_DUPLICATE_LIB_OK"]="True"
 
# def draw_cam(model, img_path, save_path, transform=None, visheadmap=False):
#     img = Image.open(img_path).convert('RGB')
#     if transform is not None:
#         img = transform(img)
#     img = img.unsqueeze(0)
#     model.eval()
#     x = model.conv1(img)
#     x = model.bn1(x)
#     x = model.relu(x)
#     x = model.maxpool(x)
#     x = model.layer1(x)
#     x = model.layer2(x)
#     x = model.layer3(x)
#     x = model.layer4(x)
#     features = x                #1x2048x7x7
#     print(features.shape)
#     output = model.avgpool(x)   #1x2048x1x1
#     print(output.shape)
#     output = output.view(output.size(0), -1)
#     print(output.shape)         #1x2048
#     output = model.fc(output)   #1x1000
#     print(output.shape)
#     def extract(g):
#         global feature_grad
#         feature_grad = g
#     pred = torch.argmax(output).item()
#     pred_class = output[:, pred]
#     features.register_hook(extract)
#     pred_class.backward()
#     greds = feature_grad
#     pooled_grads = torch.nn.functional.adaptive_avg_pool2d(greds, (1, 1))
#     pooled_grads = pooled_grads[0]
#     features = features[0]
#     for i in range(2048):
#         features[i, ...] *= pooled_grads[i, ...]
#     headmap = features.detach().numpy()
#     headmap = np.mean(headmap, axis=0)
#     headmap /= np.max(headmap)
 
#     if visheadmap:
#         plt.matshow(headmap)
#         # plt.savefig(headmap, './headmap.png')
#         plt.show()
 
#     img = cv2.imread(img_path)
#     headmap = cv2.resize(headmap, (img.shape[1], img.shape[0]))
#     headmap = np.uint8(255*headmap)
#     headmap = cv2.applyColorMap(headmap, cv2.COLORMAP_JET)
#     superimposed_img = headmap*0.4 + img
#     cv2.imwrite(save_path, superimposed_img)
 
# if __name__ == '__main__':
#     model = models.resnet50(pretrained=True)
#     transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
#     draw_cam(model, '/myDockerShare/zys/ava/emotion_data/pic/17407.jpg', '/myDockerShare/zys/aes-emotion/cam/cam_0.png', transform=transform, visheadmap=True)





# 我们的模型可视化
# CAM相关方法：Grad-CAM: https://arxiv.org/pdf/1610.02391.pdf、Grad-CAM++: https://arxiv.org/pdf/1610.02391.pdf
import os
import torch
# import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import model_cam
os.environ["KMP_DUPLICATE_LIB_OK"]="True"
 
def draw_cam(model, img_path, save_path, transform=None, visheadmap=False):
    img = Image.open(img_path).convert('RGB')
    if transform is not None:
        img = transform(img)
    img = img.unsqueeze(0)
    model.eval()
    x=model(img)
    features = x                #1x2048x7x7
    for i in range(features.shape[1]):
        feature = features[:, i, :, :]
    headmap = feature.detach().numpy()
    headmap = np.mean(headmap, axis=0)
    headmap /= np.max(headmap)
 
    if visheadmap:
        plt.matshow(headmap)
        # plt.savefig(headmap, './headmap.png')
        plt.show()
 
    img = cv2.imread(img_path)
    headmap = cv2.resize(headmap, (img.shape[1], img.shape[0]))
    headmap = np.uint8(255*headmap)
    headmap = cv2.applyColorMap(headmap, cv2.COLORMAP_JET)
    superimposed_img = headmap*0.4 + img
    cv2.imwrite(save_path, superimposed_img)
 
if __name__ == '__main__':
    #  model = models.resnet50(pretrained=True)
    model = model_cam.HyperNet(16, 112, 224, 112, 56, 49, 2, 7)
    best_state = torch.load('/myDockerShare/zys/aes-emotion/experiment/fusion4.pth')
    net_dict = model.state_dict()
    state_dict = {k: v for k, v in best_state.items() if k in net_dict.keys()}
    net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
    model.load_state_dict(net_dict)
    for name, param in model.named_parameters():
        param.requires_grad=False

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    draw_cam(model, '/myDockerShare/zys/ava/emotion_data/pic/17407.jpg', '/myDockerShare/zys/aes-emotion/cam/cam_1.png', transform=transform, visheadmap=True)
    print('ok')