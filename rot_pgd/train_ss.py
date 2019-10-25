# -*- coding: utf-8 -*-
import numpy as np
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from tqdm import tqdm
from models.wrn_with_pen import WideResNet
from pgd_attack import PGD 
import torchvision

parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/tmp', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='./snapshots/tmp', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(sorted(state.items()))

torch.manual_seed(1)
np.random.seed(1)



train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor()])
test_transform = trn.Compose([trn.ToTensor()])


trainset = torchvision.datasets.CIFAR10(root='./dataset/', train=True,
                                        download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=8 )

testset = torchvision.datasets.CIFAR10(root='./dataset/', train=False,
                                       download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_bs,
                                         shuffle=False, num_workers=8)

# Create model
net = WideResNet(args.layers, 10 , args.widen_factor, dropRate=args.droprate)
net.rot_pred = nn.Linear(128, 4)

#create PGD adversary
adversary = PGD( epsilon=8./255., num_steps=10, step_size=2./255.)
adversary_test = PGD( epsilon=8./255., num_steps=20, step_size=1./255., attack_rotations=False)


start_epoch = 0

# Restore model if desired
if args.load != '':
    for i in range(1000 - 1, -1, -1):
        model_name = os.path.join(args.load, 'cifar10' + '_' + 'wrn' +
                                  '_baseline_epoch_' + str(i) + '.pt')
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            start_epoch = i + 1
            break
    if start_epoch == 0:
        assert False, "could not resume"

data_parallel = False

# if args.ngpu > 1:
#     net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
#     data_parallel = True

if args.ngpu > 0:
    net.cuda()
    torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

optimizer = torch.optim.SGD(
    net.parameters(), state['learning_rate'], momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))



# /////////////// Training ///////////////


def train(epoch):
    net.train()  # enter train mode
    loss_avg = 0.0
    steps = 0 
    total_steps = len(train_loader)
    for bx, by in train_loader:
        curr_batch_size = bx.size(0)
        by_prime = torch.cat((torch.zeros(bx.size(0)), torch.ones(bx.size(0)),
                              2*torch.ones(bx.size(0)), 3*torch.ones(bx.size(0))), 0).long()
        bx = bx.numpy()
        bx = np.concatenate((bx, bx, np.rot90(bx, 1, axes=(2, 3)),
                             np.rot90(bx, 2, axes=(2, 3)), np.rot90(bx, 3, axes=(2, 3))), 0)
        bx = torch.FloatTensor(bx)
        bx, by, by_prime = bx.cuda(), by.cuda(), by_prime.cuda()

        adv_bx = adversary(net, bx, by, by_prime, curr_batch_size , data_parallel)

        # forward
        logits, pen = net(adv_bx * 2 - 1)

        # backward
        scheduler.step()
        optimizer.zero_grad()
        loss = F.cross_entropy(logits[:curr_batch_size], by)
        if data_parallel:
            loss += 0.5 * F.cross_entropy(net.module.rot_pred(pen[curr_batch_size:]), by_prime)
        else:
            loss += 0.5 * F.cross_entropy(net.rot_pred(pen[curr_batch_size:]), by_prime)
        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.9 + float(loss) * 0.1
        if steps %20 ==0:
            print("Epoch" , epoch , "steps" , steps ,'/' ,total_steps, "train_loss" , loss_avg )

        steps +=1

    print("lr is ",optimizer.state_dict()['param_groups'][0]['lr'])
    state['train_loss'] = loss_avg


# test function
def test(epoch):
    net.eval()
    loss_avg = 0.0
    correct = 0
    correct_adv = 0.
    eval_loader = test_loader
    with torch.no_grad():
        for data, target in eval_loader:
            data, target = data.cuda(), target.cuda()
            curr_batch_size = data.size(0)
            # forward
            output, _ = net(data * 2 - 1)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)
            
            if epoch %10 ==0:
                adv_data = adversary_test(net, data, target, None , curr_batch_size , data_parallel)
                output_adv, _ = net(adv_data * 2 - 1)
                pred_adv = output_adv.data.max(1)[1]
                correct_adv += pred_adv.eq(target.data).sum().item()

                
    print("test" , epoch , correct / len(eval_loader.dataset), correct_adv / len(eval_loader.dataset))
    state['test_loss'] = loss_avg / len(eval_loader)
    state['test_accuracy'] = correct / len(eval_loader.dataset)
    state['adv_accuracy'] = correct_adv / len(eval_loader.dataset)


if args.test:
    test(0)
    print(state)
    exit()

# Make save directory
if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

with open(os.path.join(args.save, args.dataset + '_' + 'wrn' +
                                  '_baseline_training_results.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

print('Beginning Training\n')

# Main loop
for epoch in range(0, args.epochs):
    state['epoch'] = epoch

    begin_epoch = time.time()

    train(epoch)
    test(epoch)

    # Save model
    torch.save(net.state_dict(),
               os.path.join(args.save, args.dataset + '_' + 'wrn' +
                            '_baseline_epoch_' + str(epoch) + '.pt'))
    # Let us not waste space and delete the previous model
    prev_path = os.path.join(args.save, args.dataset + '_' + 'wrn' +
                             '_baseline_epoch_' + str(epoch - 1) + '.pt')
    if os.path.exists(prev_path): os.remove(prev_path)

    # Show results

    with open(os.path.join(args.save, args.dataset + '_' + 'wrn' +
                                      '_baseline_training_results.csv'), 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
            (epoch + 1),
            time.time() - begin_epoch,
            state['train_loss'],
            state['test_loss'],
            100 - 100. * state['test_accuracy'],
        ))

    # # print state with rounded decimals
    # print({k: round(v, 4) if isinstance(v, float) else v for k, v in state.items()})

    print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}'.format(
        (epoch + 1),
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['test_loss'],
        100 - 100. * state['test_accuracy'])
    )
