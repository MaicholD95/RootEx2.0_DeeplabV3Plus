
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import torch.nn as nn
from DeepLabV3.network._deeplab import DeepLabHeadV3Plus, ASPP
from DeepLabV3.network.modeling import _load_model
from torchvision.models._utils import IntermediateLayerGetter

class MultiHeadDeeplabV3Plus(nn.Module):
    def __init__(self, backbone='resnet50', output_stride=8, pretrained_backbone_path=None):
        super(MultiHeadDeeplabV3Plus, self).__init__()
        
        backbone_model = _load_model(
            arch_type='deeplabv3plus',
            backbone=backbone,  
            num_classes=1,
            output_stride=output_stride,
            pretrained_backbone=pretrained_backbone_path
        )

        # Extract intermediate layers
        self.backbone = IntermediateLayerGetter(
            backbone_model.backbone,
            return_layers={'layer1': 'low_level', 'layer4': 'out'}
        )
        
        # Define separate heads for each class
        self.head_root = DeepLabHeadV3Plus(
            in_channels=2048,        
            low_level_channels=256,  
            num_classes=1,
            aspp_dilate=[12, 24, 36]
        )
        self.head_tip = DeepLabHeadV3Plus(
            in_channels=2048,
            low_level_channels=256,
            num_classes=1,
            aspp_dilate=[12, 24, 36]
        )
        self.head_source = DeepLabHeadV3Plus(
            in_channels=2048,
            low_level_channels=256,
            num_classes=1,
            aspp_dilate=[12, 24, 36]
        )

    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Pass features to each head
        root_output = self.head_root(features)
        tip_output = self.head_tip(features)
        source_output = self.head_source(features)
        
        # Upsample outputs to match input size
        root_output = nn.functional.interpolate(root_output, size=(512, 512), mode='bilinear', align_corners=False)
        tip_output = nn.functional.interpolate(tip_output, size=(512, 512), mode='bilinear', align_corners=False)
        source_output = nn.functional.interpolate(source_output, size=(512, 512), mode='bilinear', align_corners=False)
        
        return {'roots': root_output, 'tips': tip_output, 'sources': source_output}

    def get_shared_parameters(self):
        return list(self.backbone.parameters())

    def get_root_parameters(self):
        return list(self.head_root.parameters())

    def get_tip_parameters(self):
        return list(self.head_tip.parameters())

    def get_source_parameters(self):
        return list(self.head_source.parameters())
