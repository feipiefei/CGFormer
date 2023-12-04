from __future__ import print_function
import argparse

import torch.cuda
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import MyDataset
from model import VisisionTransformer
import torch.backends.cudnn as cudnn
import os
import numpy as np
from einops import rearrange, repeat
from tqdm import tqdm
import networks
import torch.nn.functional as F
from timm import create_model as creat
from timm.loss import LabelSmoothingCrossEntropy


parser = argparse.ArgumentParser(description='ViT NI vs CG')
parser.add_argument('--data_root', type=str, help='path to images')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--img_mode', type=str, default='RGB', help='choose how image are loaded')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training')
parser.add_argument('--lr', type=int, default=0.003, metavar='LR', help='Learing rate')
parser.add_argument('--epochs', type=int, default=150, help='Epochs')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--wd', type=float, default=1e-4, metavar='W', help='weight decay')
parser.add_argument('--optimizer', type=str, default='sgd', metavar='OPT', help='The optimizer to use')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--log-interval', type=int, default='10', metavar='N', help='how many epoch to save model')
args = parser.parse_args(args=[])

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device("cuda")
LOG_DIR = './checkpoint'
os.maskdirs(LOG_DIR, exist_ok=True)
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    cudnn.benchmark = True
kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
data_root = './dataset/....'
normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
args.data_root = os.path.join(data_root, 'train')
train_loader = MyDataset.DataLoaderHalf(
    MyDataset.MyDataset(
        args,
        transforms.Compose([
            transforms.Resize(512),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])), batch_size=args.batch_size,
    shuffle=True, half_constraint=True,
    sampler_type='RandomBalancedSampler', **kwargs,
    drop_last=True
)
print('the number of train data:{}'.format(len(train_loader.dataset)))

def main():
    model = VisisionTransformer('vit_base_patch16_224',Test=False)
    networks.print_network(model)
    if args.cuda:
        model =torch.nn.DataParallel(model)
        model.cuda()
    from collections import OrderedDict
    def load_model(model, check):
        model_state = model.state_dict()
        tempState = OrderedDict()

        for i in range(len(check.keys())):
            tempState[list(model_state.keys())[i]] = check[list(check.keys())[i]]
        model.load_state_dict(tempState)
        return model
    criterion = LabelSmoothingCrossEntropy(smoothing=0.05).cuda()
    criterion1 = nn.CrossEntropyLoss().cuda()
    L1_criterion = nn.L1Loss(reduction='sum').cuda()
    optimizer = create_optimizer(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-7)
    for epoch in range(1, args.epochs + 1):
        train(train_loader, model, optimizer, criterion,criterion1 ,L1_criterion, epoch, scheduler)

def train(train_loader, model, optimizer, criterion,criterion1, L1_criterion, epoch,scheduler):
    oriImageLabel = []
    oriTestLabel = []
    scheduler.step()
    model.train()

    pbar = tqdm(enumerate(train_loader))
    for batch_idx, (data, target) in pbar:
        if args.cuda:
            data_var, target_var = data.cuda(), target.cuda()
        # zero = torch.zeros_like(label_maps_var)
        # ones  = torch.ones_like(a)
        bs, c, h, w = data_var.size()
        model_out = model(data_var)
        cls, token = model_out[:,0,:], model_out[:,1:,:]
        cls_score = F.softmax(cls, dim=-1)
        token = torch.softmax(token, dim=-1)  # (B, 196, 2)
        x = torch.zeros_like(token).cuda()
        y = torch.ones_like(token).cuda()
        token = torch.where(token < 0.5, x + 0.01, token)
        token = torch.where(token > 0.5, y - 0.01, token)
        # print(token.requires_grad)
        token_target = target_var.unsqueeze(dim=-1).unsqueeze(dim=-1)
        logprobs = torch.log(token)
        nll_loss = -logprobs.gather(dim=-1, index=token_target)
        token_loss = nll_loss.mean()
        pred = cls_score.data.max(1, keepdim=True)[1]
        oriTestLabel.extend(pred.squeeze().cpu().numpy())
        oriImageLabel.extend(target.data.cpu().numpy())
        L1_loss = 0
        for name, param in model.named_parameters():
            if name.find('bias') == -1:
                L1_loss += L1_criterion(param, torch.zeros_like(param))
        loss_cls = criterion1(cls, target_var)  # + args.wd + L1_loss
        loss = (loss_cls + token_loss)/2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]. Loss:{:.8f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                    loss.item()
                ))
    if epoch % 30 == 0:
        torch.save({
            'epoch': epoch,
            'state_dict': model.module.state_dict()},
            '{}/checkpoint_{}.pth'.format(LOG_DIR, epoch)
        )
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate"""
    lr = args.lr * (0.1 ** ((epoch - 1) // TRAIN_STEP))
    print('lr: {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def create_optimizer(params, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=new_lr,
                              momentum=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(params, lr=new_lr,
                               weight_decay=args.wd, betas=(args.beta1, 0.999))
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(params,
                                  lr=new_lr,
                                  lr_decay=args.lr_decay,
                                  weight_decay=args.wd)
    return optimizer


if __name__ == '__main__':
    main()