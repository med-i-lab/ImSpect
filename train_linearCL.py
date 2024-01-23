# python train_linearCL.py --exp-dir path_to_save_results  --learning_rate 0.5 --epochs 300 --model resnet18 --cosine

from __future__ import print_function

import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
# from main_ce import set_loader
from utils.util import AverageMeter
from utils.util import adjust_learning_rate, warmup_learning_rate, accuracy
from utils.util import set_optimizer
import models
from dataloader_reims import REIMS_dataset  # Need to write your dataloader
import numpy as np
from sklearn.metrics import roc_auc_score
import wandb
import os


def fix_random_seed(seed):
    """Ensure reproducible results"""
    import torch
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--ce', type=float, default=1.0,
                        help='ce multiplier')
    parser.add_argument('--rce', type=float, default=1.0,
                        help='rce multiplier')
    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--data', type=str, default='path')
    parser.add_argument('--fold', type=str, default='4')
    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--exp-dir', type=str, default='/home/Codes/REIMSVission/exp1/',
                        help='path to results folder')
    parser.add_argument('--ckpt', type=str, default='/home/Codes/REIMSVission/ssl/',
                    help='path to pre-trained model')
    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = '/home/Codes/REIMSVission/Data/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_lr_{}_decay_{}_bsz_{}'.\
        format( opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)
    
    if not(os.path.exists(opt.exp_dir)):
        os.mkdir(opt.exp_dir)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

   
    opt.n_cls = 2
   

    return opt

def make_weights_for_balanced_classes(images, nclasses):      
    count = [0] * nclasses                                                      
    for item in images:   
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                
    return weight      

def set_loader(opt):
    # construct data loader
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = REIMS_dataset(root=opt.data_folder, transform=train_transform, mode='all_sup', fold=opt.fold)
    eval_dataset = REIMS_dataset(root=opt.data_folder, transform=val_transform, mode='val', fold=opt.fold)
    test_dataset = REIMS_dataset(root=opt.data_folder, transform=val_transform, mode='test')
    
    weights = make_weights_for_balanced_classes(train_dataset, 2)
    weights = torch.Tensor(weights)
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    
    # train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=8, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=8, pin_memory=True)


    return train_loader, eval_loader, test_loader

def set_model(opt):
    opt.arch = opt.model
    model = models.encoder.EncodeProject(opt)  

    criterion = torch.nn.CrossEntropyLoss().cuda()

    classifier = models.resnet.LinearClassifier(name=opt.arch, num_classes=opt.n_cls)

    # Load saved SSL model
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])


    if torch.cuda.is_available(): 
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, classifier, criterion

def compute_accuracy(gt_all, y_pred_all, prob_all):
    auc_ = roc_auc_score(gt_all, prob_all[:,1])
    andlabels = np.logical_and(y_pred_all, gt_all)
    norLabels = len(np.where(y_pred_all + gt_all == 0)[0])
    Acc_test = (np.sum(andlabels) + norLabels) / len(gt_all)
    Sen_test_ = np.sum(andlabels) / np.sum(gt_all)
    Spe_test_ = norLabels / (len(gt_all) - np.sum(gt_all))
    balanced_acc = (Spe_test_+Sen_test_)/2
    return balanced_acc, Acc_test, auc_, Sen_test_, Spe_test_

def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    gt_train = []
    pred_train = []
    prob_all = []
    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model(images, out='h')
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)

        gt_train.append(labels.cpu().numpy())
        pred = torch.argmax(torch.softmax(output, dim=1), dim=1)
        pred_train.append(pred.cpu().numpy())
        prob = torch.softmax(output, dim=1).detach()
        prob_all.append(prob.cpu().numpy())
       
        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    gt_train = np.concatenate(gt_train)
    pred_train = np.concatenate(pred_train)
    prob_all = np.concatenate(prob_all)
    
    return losses.avg

def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    gt_train = []
    pred_train = []
    prob_all = []
    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model(images, out='h'))
            loss = criterion(output, labels)

            gt_train.append(labels.cpu().numpy())
            pred = torch.argmax(torch.softmax(output, dim=1), dim=1)
            pred_train.append(pred.cpu().numpy())
            prob = torch.softmax(output, dim=1)
            prob_all.append(prob.cpu().numpy())
            # update metric
            losses.update(loss.item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    gt_train = np.concatenate(gt_train)
    pred_train = np.concatenate(pred_train)
    prob_all = np.concatenate(prob_all)

    balanced_acc, Acc_test, auc_, Sen_test_, Spe_test_ = compute_accuracy(gt_train, pred_train, prob_all)

    return losses.avg, balanced_acc, Acc_test, auc_,  Sen_test_, Spe_test_ 


    
def main():
    
    wandb.init(project="REIMSSMCLR", entity="user")

    seed = 1234
    fix_random_seed(seed)
    best_acc = 0
    acc_test = 0

    opt = parse_option()
  
    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build data loader
    train_loader, eval_loader, test_loader = set_loader(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()

        # eval for one epoch
        loss,  balanced_acc, Acc, auc, Sen, Spe = validate(train_loader, model, classifier, criterion, opt)

        wandb.log({"TrainLoss": loss, 'custom_step': epoch})
        wandb.log({"TrainAcc-B": balanced_acc, 'custom_step': epoch})
        wandb.log({"TrainAcc": Acc, 'custom_step': epoch})
        wandb.log({"TrainAUC": auc, 'custom_step': epoch})
        wandb.log({"TrainSen": Sen, 'custom_step': epoch})
        wandb.log({"TrainSpe": Spe, 'custom_step': epoch})
        
        loss,  balanced_acc_val, Acc, auc, Sen, Spe = validate(eval_loader, model, classifier, criterion, opt)
       
        wandb.log({"ValLoss": loss, 'custom_step': epoch})
        wandb.log({"ValAcc-B": balanced_acc_val, 'custom_step': epoch})
        wandb.log({"ValAcc": Acc, 'custom_step': epoch})
        wandb.log({"ValAUC": auc, 'custom_step': epoch})
        wandb.log({"ValSen": Sen, 'custom_step': epoch})
        wandb.log({"ValSpe": Spe, 'custom_step': epoch})

        loss,  balanced_acc_test, Acc, auc, Sen, Spe = validate(test_loader, model, classifier, criterion, opt)
       
        wandb.log({"TestLoss": loss, 'custom_step': epoch})
        wandb.log({"TestAcc-B": balanced_acc_test, 'custom_step': epoch})
        wandb.log({"TestAcc": Acc, 'custom_step': epoch})
        wandb.log({"TestAUC": auc, 'custom_step': epoch})
        wandb.log({"TestSen": Sen, 'custom_step': epoch})
        wandb.log({"TestSpe": Spe, 'custom_step': epoch})
      
        if balanced_acc_val >= acc_test:
            acc_test = balanced_acc_val
            torch.save(classifier.state_dict(), opt.exp_dir +"Best-val-classifier-"+str(seed)+'-'+str(epoch)+".pth")



if __name__ == '__main__':
    main()
    