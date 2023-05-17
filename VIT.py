import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.models as models

from torch.autograd import Variable


# 2D CNN encoder using ResNet-152 pretrained

class ResCNNEncoder(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        resnet = models.resnet18(pretrained=False)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

        
    def forward(self, x_3d):
        cnn_embed_seq = []
        # ResNet CNN
        x_3d = torch.squeeze(x_3d, 0)
        # with torch.no_grad():
        #     x = self.resnet(x_3d[:, :, :, :])  # ResNet
        #     x = x.view(x.size(0), -1)             # flatten output of conv
        x = self.resnet(x_3d)  # ResNet
        x = x.view(x.size(0), -1)  
        # FC layers
        x = self.bn1(self.fc1(x))
        x = F.relu(x)
        x = self.bn2(self.fc2(x))
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)

        cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)

        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


class TransformerModel(nn.Module):
    # TODO PYTORCH提供的transformer可以使用订制的encoder和decoder
    def __init__(self, CNN_embed_dim=300, h_Transformer_layers=3, h_Transformer=256, h_FC_dim=128, drop_p=0.3, num_classes=10):
        super(TransformerModel, self).__init__()

        self.Transformer_input_size = CNN_embed_dim
        self.h_Transformer_layers = h_Transformer_layers   
        self.h_Transformer = h_Transformer                 
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes


        self.transformer = nn.Transformer(
            d_model = self.Transformer_input_size,
            # num_encoder_layers = 0,
            num_decoder_layers = 0,
            dim_feedforward = self.h_Transformer,
            dropout = self.drop_p
        )

        self.fc1 = nn.Linear(self.h_Transformer, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_Transformer):
        
        Transformer_out = self.transformer(x_Transformer, x_Transformer) 
        # print(Transformer_out.shape)
        # x=Transformer_out.squeeze(dim=1)
        # FC layers 
        x = self.fc1(Transformer_out[:, -1, :])  
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        # x = self.fc2(x)
        return x