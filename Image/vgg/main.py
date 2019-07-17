import argparse
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.parallel

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import vgg

model_names = sorted(name for name in vgg.__dict__
                     if name.islower() and not name.startswith("__")
                     and name.startswith("vgg") and callable(vgg.__dict__[name]))

parser = argparse.ArgumentParser(description="Pytorch ImageNet Training")

parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: vgg16) ')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N',
                    help='mini-batch-size')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')

# for save checkpoint
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# train & validate

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

# utils
# half precision : floating point format
parser.add_argument('--half', dest='half', action='store_true',  # action : option이 지정되면 true 반환
                    help='use half-precision(16-bit) ')
parser.add_argument('--cpu', dest='cpu', action='store_true',
                    help='use cpu')
parser.add_argument('--save_dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--print-freq', '-p', default=20,
                    type=int, metavar='N', help='print frequency')

#
best_prec1 = 0

#
args = parser.parse_args()

# change learning rate


def adjust_learning_rate(optimizer, epoch):
    """ set learning rate decayed 2 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 30))

    # lr을 optimizer group 바꿔줌.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    global args, best_prec1
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    model = vgg.__dict__[args.arch]()  # model architecture
    model.features = torch.nn.DataParallel(model.features)

    if args.cpu:
        model.cpu()
    else:
        model.cuda()

    # optionally resume from a checkpoint
    if args.resume:  # resume : file name of checkpoint
        if os.path.isfile(args.resume):
            print("==>loading the checkpoint {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("==> no check point found at {}".format(args.resume))

    cudnn.benchmark = True

# normalize
 # chnnel => n : 3
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # CIFAR datset
    """ cifar 데이터의 경우 (H x W x C) range가 [0, 255] 이기 때문에
        계산을 위해서 torch.FloatTensor [0, 1] range로 바꿔줄 필요가 있다."""

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize
        ]), download=False),  # already exists
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([  # transform
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


# define loss function
    criterion = nn.CrossEntropyLoss()
    if args.cpu:
        criterion = criterion.cpu()
    else:
        criterion = criterion.cuda()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # if args.evaluate:
    #     validate(val_loader, model, criterion)
    #     return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        if epoch % 49 == 0:
            torch.save(model, os.path.join(args.save_dir,
                'chkpoint_{}.tar'.format(epoch)))

        # for save validate checkpoint
        # is_best = prec1 > best_prec1
        # best_prec1 = max(prec1, best_prec1)

        # if epoch % 40 == 0:
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': model.state_dict(),
        #         'best_prec1': best_prec1,
        #     }, is_best, file_name=os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(epoch)))

# loss funtion


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, criterion, optimizer, epoch):

    # 변수관리 - 여기서는 객체로 관리하는 점이 특이 (한번에 하려는 것으로 보임)
    losses = AverageMeter()
    top1 = AverageMeter()

    # swich the train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):

        if args.cpu == False:
            input = input.cuda(async=True)
            target = target.cuda(async=True)
        if args.half:
            input = input.half()

        # compute output
        output = model(input)
        loss=criterion(output, target)

        # compute gradient and SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output=output.float()
        loss=loss.float()

        # measure accuracy and record loss
        prec1=accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), loss=losses, top1=top1
                  )
                  )


def accuracy(output, target, topk=(1,)):
    """ Computes the precision for the specified values of k """
    maxk=max(topk)
    batch_size=target.size(0)

    _, pred=output.topk(maxk, 1, True, True)  # largest=True, sorted=True
    pred=pred.t()
    correct=pred.eq(target.view(1, -1).expand_as(pred))

    res=[]
    for k in topk:
        correct_k=correct[:k].view(-1).float().sum(0)
        # batch size 개수에서 얼만큼 맞췄는지
        res.append(correct_k.mul_(100.0 / batch_size))
                                                        # 100. to make persent %
    return res

# 일반 모델 저장일 경우, torch.save(model)
 # torch.save(state, _, filename)


def save_checkpoint(model, is_best, file_name='checkpoint.pth.tar'):
    torch.save(model, file_name)

def load_chcekpoint(file):
    return torch.load(file)

if __name__ == '__main__':
    main()
