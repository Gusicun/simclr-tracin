import torch.nn as nn
import torchvision.models as models
import VGGFace
import VGGFace2.utils
from exceptions.exceptions import InvalidBackboneError


class VGGSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(VGGSimCLR, self).__init__()
        self.face_dict = {"VGG-Pre":VGGFace.baseModel(),
                          "VGG-FT":VGGFace.loadModel()}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)