import torch
import torch.nn as nn
import timm
from timm import create_model as creat
from torch.nn import init
from einops import rearrange, repeat
import os
from collections import OrderedDict
def load_model(model, check):
    model_state = model.state_dict()
    tempState = OrderedDict()
    for i in range(len(check.keys())):
        tempState[list(model_state.keys())[i]] = check[list(check.keys())[i]]
    model.load_state_dict(tempState)
    return model
class VisisionTransformer(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', Test=False, requires_grad=False):
        super(VisisionTransformer, self).__init__()
        if Test:
            print('test')
            print(model_name)
            model = creat(model_name, pretrained=False, num_classes=2)
        else:
            print('train')
            print(model_name)
            model = creat(model_name, pretrained=True, num_classes=2)

        self.model = model
        
    def forward(self, x):
        B = x.shape[0]
        x = self.model.patch_embed(x)
        x = x + self.model.pos_embed[:, 1:, :]
        cls_tokens = self.model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.model.pos_drop(x)
        for blk in self.model.blocks:
            x = blk(x)
        return self.model.head(x)