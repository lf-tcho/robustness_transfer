import numpy as np
# from robustbench.model_zoo.architectures.wide_resnet import WideResNet
import torch.nn.functional as F
from torchvision.models import ResNet
import torch
import torch.nn as nn

class Model_dsprites(nn.Module):
    def __init__(self, feat_size=84, input_dim=(1, 64, 64)):
        """
        Initializes the core encoder network.
        Optional args:
        - feat_size (int): size of the final features layer (default: 84)
        - input_dim (tuple): input image dimensions (channels, width, height) 
            (default: (1, 64, 64))
        """

        super().__init__()

        # check input dimensions provided
        self.input_dim = tuple(input_dim)
        if len(self.input_dim) == 2:
            self.input_dim = (1, *input_dim)            
        elif len(self.input_dim) != 3:
            raise ValueError("input_dim should have length 2 (wid x hei) or "
                f"3 (ch x wid x hei), but has length ({len(self.input_dim)}).")
        self.input_ch = self.input_dim[0]

        # convolutional component of the feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_ch, out_channels=6, kernel_size=5, 
                stride=1
                ),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.BatchNorm2d(6, affine=False),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.BatchNorm2d(16, affine=False)
        )

        # calculate size of the convolutional feature extractor output
        self.feat_extr_output_size = \
            self._get_feat_extr_output_size(self.input_dim)
        self.feat_size = feat_size

        # linear component of the feature extractor
        self.linear_projections = nn.Sequential(
            nn.Linear(self.feat_extr_output_size, 120),
            nn.ReLU(),
            nn.BatchNorm1d(120, affine=False),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.BatchNorm1d(84, affine=False),
        )

        self.linear_projections_output = nn.Sequential(
            nn.Linear(84, self.feat_size),
            nn.ReLU(),
            nn.BatchNorm1d(self.feat_size, affine=False)
        )

        self.fc = nn.Linear(feat_size, 1)
        
    def _get_feat_extr_output_size(self, input_dim):
        dummy_tensor = torch.ones(1, *input_dim)
        reset_training = self.training
        self.eval()
        with torch.no_grad():   
            output_dim = self.feature_extractor(dummy_tensor).shape
        if reset_training:
            self.train()
        return np.product(output_dim)

    def forward(self, X):
        feats_extr = self.feature_extractor(X)
        feats_flat = torch.flatten(feats_extr, 1)
        feats_proj = self.linear_projections(feats_flat)
        feats = self.linear_projections_output(feats_proj)
        feats = self.fc(feats.flatten(start_dim=1))
        return feats

    def get_features(self, X):
        with torch.no_grad():
            feats_extr = self.feature_extractor(X)
            feats_flat = torch.flatten(feats_extr, 1)
            feats_proj = self.linear_projections(feats_flat)
            feats = self.linear_projections_output(feats_proj)
        return feats

# class WideResNetForLwF(WideResNet):

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.block1(out)
#         out = self.block2(out)
#         out = self.block3(out)
#         out = self.relu(self.bn1(out))
#         out = F.avg_pool2d(out, 8)
#         out = out.view(-1, self.nChannels)
#         return self.fc(out), out


# class WideResNetForAdvRep(WideResNet):

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.block1(out)
#         out = self.block2(out)
#         out = self.block3(out)
#         out = self.relu(self.bn1(out))
#         out = F.avg_pool2d(out, 8)
#         out = out.view(-1, self.nChannels)
#         return out


# class WideResNetForLwFImageNet(ResNet):

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)

#         return self.fc(x), x


# class WideResNetForAdvRepImageNet(ResNet):

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)

#         return x
