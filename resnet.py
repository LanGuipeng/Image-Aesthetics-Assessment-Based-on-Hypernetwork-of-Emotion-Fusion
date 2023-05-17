import torch.nn as nn
import torchvision as tv
import torch

MODELS = {
    "resnet18": (tv.models.resnet18, 512),
    "resnet34": (tv.models.resnet34, 512),
    "resnet50": (tv.models.resnet50, 2048),
    "resnet101": (tv.models.resnet101, 2048),
    "resnet152": (tv.models.resnet152, 2048),
    "vgg": (tv.models.vgg16, 4096),
    "mobilnet": (tv.models.mobilenet_v2, 1280),

}


class NIMA(nn.Module):
    def __init__(self, base_model: nn.Module, input_features: int, drop_out: float):
        super(NIMA, self).__init__()
        self.base_model = base_model
        # self.fc1=nn.Linear(512 * 7 * 7, 4096)
        # self.fc2=nn.Linear(4096, 4096)
        self.gvp=nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.ReLU(inplace=True), nn.Dropout(p=drop_out), nn.Linear(input_features, 10), nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


def create_model(model_type: str, drop_out: float) -> NIMA:
    create_function, input_features = MODELS[model_type]
    base_model = create_function(pretrained=True)
    base_model = nn.Sequential(*list(base_model.children())[:-1])
    return NIMA(base_model=base_model, input_features=input_features, drop_out=drop_out)

if __name__=='__main__':
    model_type='resnet50'
    x1 = torch.rand(32,3,224,224)
    model=create_model(model_type,0.75)
    # print(model)
    y=model(x1)
    # print(y.shape)
    print('ok')

