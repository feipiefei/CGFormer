import numpy as np
from PIL import Image
import sys, os
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
from model import VisisionTransformer
import argparse
import networks
import MyDataset
from tqdm import tqdm
import torch.nn.functional as F
from timm import create_model as creat
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
# Testing settings
parser = argparse.ArgumentParser(description='PyTorch NI vs CG')
parser.add_argument('--dataroot', type=str, help='path to images')
parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--img_mode', type=str, default='RGB', help='chooses how image are loaded')
parser.add_argument('--test-batch-size', type=int, default=5, metavar='N',help='input batch size for testing (default: 5)')
parser.add_argument('--no-cuda', action='store_true', default=False,help='disables CUDA training')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    cudnn.benchmark = True
kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}


dataroot = './.../test'

## CG image num
kCgNum  = 1111111  # 


normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

args.data_root = dataroot 
test_loader = torch.utils.data.DataLoader(
    MyDataset.MyDataset(args,
                    transforms.Compose([
                        transforms.Resize(512),
                        transforms.TenCrop(224),
                        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
                   ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)
print('The number of test data:{}'.format(len(test_loader.dataset)))
from collections import OrderedDict
def load_model(model, check):
    model_state = model.state_dict()
    tempState = OrderedDict()
    for i in range(len(check.keys())):
        tempState[list(model_state.keys())[i]] = check[list(check.keys())[i]]
    model.load_state_dict(tempState)
    return model
# instantiate model and initialize weights
model = VisisionTransformer('vit_base_patch16_224', Test=True, requires_grad=False)

## checkpoint
checkpoint = torch.load('..../.pth')
load_model(model, checkpoint['state_dict'])
model = torch.nn.DataParallel(model)
model.cuda()
model.eval()
pbar = tqdm(enumerate(test_loader))
class_cls = []
class_token = []
class_cls_token = []
ImageLabel = []
weight = 0.7
with torch.no_grad():
    for batch_idx, (data, target) in pbar:
        data, target= data.cuda(), target.cuda()
        ImageLabel.extend(target.data.cpu().numpy())
        b,ncrop,c,h,w = data.shape
        input = data.view(b*ncrop,c,h,w)
        out = model(input)
        cls_out, token_out = out[:,0,:] , out[:,1:,:]
        
        cls_out = cls_out.view(b,ncrop,2)
        token_out = token_out.view(b,ncrop,196,2)
        
        cls_score = F.softmax(cls_out, dim=-1) #(b,10,2)
        cls_score = cls_score.mean(dim=1) #(b,2)


        
        token_score = F.softmax(token_out, dim=-1) #(b,10,196,2)
        token_score = token_score.mean(dim=2) #(b,10,2)
        token_score = token_score.mean(dim=1)#(b,2)

       
        pred_cls = cls_score.data.max(dim=1)[1]
        class_cls.extend(pred_cls.data.cpu().numpy())

        
        pred_token = token_score.data.max(dim=1)[1]
        class_token.extend(pred_token.data.cpu().numpy())

        
        box = torch.ones_like(cls_score)
        box[:,0] = (weight)*(cls_score[:,0]) + (1-weight)*(token_score[:,0])
        box[:,1] = (weight)*(cls_score[:,1]) + (1-weight)*(token_score[:,1])
        pred_cls_token = box.data.max(dim=1)[1]
        class_cls_token.extend(pred_cls_token.data.cpu().numpy())

class_cls_result = np.array(ImageLabel) == np.array(class_cls)
class_token_result = np.array(ImageLabel) == np.array(class_token)
class_cls_token_result = np.array(ImageLabel) == np.array(class_cls_token)



cls_cg_result = class_cls_result[:kCgNum]
cls_ni_result = class_cls_result[kCgNum:]
ni_acc1 = cls_ni_result.sum() * 100.0 / len(cls_ni_result)
cg_acc1 = cls_cg_result.sum() * 100.0 / len(cls_cg_result)
hter1 = ((len(cls_cg_result) - cls_cg_result.sum()) * 100.0 / len(cls_cg_result) + (
            len(cls_ni_result) - cls_ni_result.sum()) * 100.0 / len(cls_ni_result)) / 2
print('cls__NI accuracy is: ', ni_acc1)
print('cls__CG accuracy is: ', cg_acc1)
print('cls__HTER: ', hter1)



token_cg_result = class_token_result[:kCgNum]
token_ni_result = class_token_result[kCgNum:]
ni_acc2 = token_ni_result.sum() * 100.0 / len(token_ni_result)
cg_acc2 = token_cg_result.sum() * 100.0 / len(token_cg_result)
hter2 = ((len(token_cg_result) - token_cg_result.sum()) * 100.0 / len(token_cg_result) + (
            len(token_ni_result) - token_ni_result.sum()) * 100.0 / len(token_ni_result)) / 2
print('token__NI accuracy is: ', ni_acc2)
print('token__CG accuracy is: ', cg_acc2)
print('token__HTER: ', hter2)



cls_token_cg_result = class_cls_token_result[:kCgNum]
cls_token_ni_result = class_cls_token_result[kCgNum:]
ni_acc3 = cls_token_ni_result.sum() * 100.0 / len(cls_token_ni_result)
cg_acc3 = cls_token_cg_result.sum() * 100.0 / len(cls_token_cg_result)
hter3 = ((len(cls_token_cg_result) - cls_token_cg_result.sum()) * 100.0 / len(cls_token_cg_result) + (
            len(cls_token_ni_result) - cls_token_ni_result.sum()) * 100.0 / len(cls_token_ni_result)) / 2
print('cls_token__NI accuracy is: ', ni_acc3)
print('cls_token__CG accuracy is: ', cg_acc3)
print('cls_token__HTER: ', hter3)