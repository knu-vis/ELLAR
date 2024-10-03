import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead

@HEADS.register_module()
class I3DHead(BaseHead):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels=1024,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 freeze_classification_head=True,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)
        self.fc_cls_15 = nn.Linear(self.in_channels, self.num_classes)
        self.fc_cls_20 = nn.Linear(self.in_channels, self.num_classes)
        self.fc_cls_25 = nn.Linear(self.in_channels, self.num_classes)
        self.fc_cls_30 = nn.Linear(self.in_channels, self.num_classes)

        
        # self.init_weights() # turn off when load pre-trained model

        if self.spatial_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None


        # freeze the classification head
        if freeze_classification_head:
            for fc in [self.fc_cls, self.fc_cls_15, self.fc_cls_20, self.fc_cls_25, self.fc_cls_30]:
                for param in fc.parameters():
                    param.requires_grad = False

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)
        normal_init(self.fc_cls_15, std=self.init_std)
        normal_init(self.fc_cls_20, std=self.init_std)
        normal_init(self.fc_cls_25, std=self.init_std)
        normal_init(self.fc_cls_30, std=self.init_std)
        

     def forward(self, x, max_lambda_idx):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
            max_lambda_idx (torch.Tensor): The max_lambda indices.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """

        # Ensure max_lambda_idx is an integer tensor on the same device as x
        max_lambda_idx = max_lambda_idx.long().to(x.device) # [0,0,0,0,0]
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x[0])
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
       
        # Create an empty tensor to store the classification scores
        cls_score = torch.zeros((x.size(0), self.num_classes), device=x.device)
  
        temp = []
        for i, (sample, max_lambda) in enumerate(zip(x, max_lambda_idx)):
            sample = sample.unsqueeze(0)  # keep the batch dim
            if max_lambda == 0:
                sample_cls_score = self.fc_cls(sample)
                # print(f"fc_cls(GIC=1) are activated")

            elif max_lambda == 1:
                sample_cls_score = self.fc_cls_15(sample)
                # print(f"fc_cls(GIC=1.5) are activated")

            elif max_lambda == 2:
                sample_cls_score = self.fc_cls_20(sample)
                # print(f"fc_cls(GIC=2.0) are activated")

            elif max_lambda == 3:
                sample_cls_score = self.fc_cls_25(sample)
                # print(f"fc_cls(GIC=2.5) are activated")

            elif max_lambda == 4:
                sample_cls_score = self.fc_cls_30(sample)
                # print(f"fc_cls(GIC=3.0) are activated")
            else:
                print("Error: max_lambda index is out of range")
                import sys
                sys.exit()

            temp.append(sample_cls_score.squeeze(0))
        temp = torch.stack(temp, dim=0)
        return temp
        
