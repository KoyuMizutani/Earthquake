import os
from datetime import datetime
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

from network import RegNetwork
from dataset import RegDataset

from sklearn.metrics import matthews_corrcoef


def main():
    parser = argparse.ArgumentParser(description='Earthquaker')

    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='Number of sweeps over the training data')
    parser.add_argument('--frequency', '-f', type=int, default=10,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--lr', type=float, default=0.1, 
                        help='learning rate')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='../result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--inputwidth', '-i', type=int, default=15,
                        help='input width')
    parser.add_argument('--inputdim', type=int, default=1,
                        help='input dim')
    parser.add_argument('--layers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--hiddendim', type=int, default=1000,
                        help='hidden layer dim')
    parser.add_argument('--dropprob', type=float, default=0.2,
                        help='dropout probability')
    parser.add_argument('--dataset', '-d', required=True,
                        help='Root directory of dataset')
    args = parser.parse_args()

    # 現在の日時を取得
    now = datetime.now()

    # ディレクトリ名にするための日時の文字列を生成
    # format: result_YYYYMMDD_HHMMSS
    dir_name = "result_" + now.strftime("%Y%m%d_%H%M%S")

    # 新しいディレクトリのパスを作成
    path = os.path.join(args.out, dir_name)

    # 新しいディレクトリを作成
    # 存在しない場合のみ作成する
    os.makedirs(path, exist_ok=True)

    out = path

    print('# Regression Model Training')
    print('# GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('# frequency: {}'.format(args.frequency))
    print('# learning rate: {}'.format(args.lr))
    print('# output directory: {}'.format(out))
    print('# input width: {}'.format(args.inputwidth))
    print('# input dim: {}'.format(args.inputdim))
    print('# layers: {}'.format(args.layers))
    print('# hidden dim: {}'.format(args.hiddendim))
    print('# dropout probability: {}'.format(args.dropprob))
    print('')

    with open(out + "/log.txt", "w") as f:
        f.write('# Regression Model Training\n')
        f.write('# GPU: {}\n'.format(args.gpu))
        f.write('# Minibatch-size: {}\n'.format(args.batchsize))
        f.write('# epoch: {}\n'.format(args.epoch))
        f.write('# frequency: {}\n'.format(args.frequency))
        f.write('# learning rate: {}\n'.format(args.lr))
        f.write('# output directory: {}\n'.format(out))
        f.write('# input width: {}\n'.format(args.inputwidth))
        f.write('# input dim: {}\n'.format(args.inputdim))
        f.write('# layers: {}\n'.format(args.layers))
        f.write('# hidden dim: {}\n'.format(args.hiddendim))
        f.write('# dropout probability: {}\n'.format(args.dropprob))

    net = RegNetwork(input_dim=args.inputdim, n_layers=args.layers, hidden_dim=args.hiddendim, drop_prob=args.dropprob)
    if args.resume:
        net.load_state_dict(torch.load(args.resume))
        print('Loaded {}'.format(args.resume))

    if args.gpu >= 0:
        print("GPU using")
        device = 'cuda:' + str(args.gpu)
        net = net.to(device)

    criterion = nn.MSELoss(reduction="sum")

    # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9) #lr 0.001
    optimizer = optim.Adam(net.parameters(), args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch)

    trainset = RegDataset(root=args.dataset, mode="train", input_width=args.inputwidth, input_dim=args.inputdim)
    valset = RegDataset(root=args.dataset, mode="val", input_width=args.inputwidth, input_dim=args.inputdim)
    trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=args.batchsize, shuffle=True, num_workers=2)

    print("# trainset size {}".format(len(trainset)))
    print("# valset size {}".format(len(valset)))

    # Setup result holder
    x = []
    train_loss_record = []
    val_loss_record = []
    max_corrcoef = -1.0
    # Train
    for ep in range(args.epoch):
        net.train()
        train_loss = 0.0
        train_mask_true_nums = 0
        for data in trainloader:
            inputs, labels = data

            if args.gpu >= 0:
                inputs = inputs.to(device)
                labels = labels.to(device)
            # Reset the parameter gradients
            optimizer.zero_grad()
            # Forward
            outputs = net(inputs)
            # Backward + Optimize
            mask = labels >= 0.5
            train_mask_true_nums += torch.sum(mask).item()
            masked_outputs = outputs[mask]
            masked_labels = labels[mask]
            loss = criterion(masked_outputs, masked_labels)
            # loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Add loss
            train_loss += loss.item()
        scheduler.step()

        train_loss /= train_mask_true_nums

        # Save the model
        if (ep + 1) % args.frequency == 0:
            path = out + "/model_" + str(ep + 1)
            torch.save(net.state_dict(), path)

        # Validation
        net.eval()
        val_loss = 0.0
        val_mask_true_nums = 0
        predict_array = []
        label_array = []
        with torch.no_grad():
            for data in valloader:
                images, labels = data
                if args.gpu >= 0:
                    images = images.to(device)
                    labels = labels.to(device)
                outputs = net(images)
                # Predict the label
                predicted = torch.where(outputs >= 0.5, outputs, torch.zeros_like(outputs))
                # Check whether estimation is right
                label_array += labels.reshape(-1).to("cpu").tolist()
                predict_array += predicted.reshape(-1).to("cpu").tolist()
                # Calculate loss
                mask = labels >= 0.5
                val_mask_true_nums += torch.sum(mask).item()
                masked_outputs = outputs[mask]
                masked_labels = labels[mask]
                loss = criterion(outputs, labels)
                # Add loss
                val_loss += loss.item()
        
        val_loss /= val_mask_true_nums
        # Record result
        x.append(ep + 1)
        train_loss_record.append(train_loss)
        val_loss_record.append(val_loss)
        corrcoef = np.corrcoef(np.array(label_array), np.array(predict_array))[0][1]
        if max_corrcoef < corrcoef:
            max_corrcoef = corrcoef
            path = out + "/model_best"
            torch.save(net.state_dict(), path)

        # Report loss of the epoch
        print('[epoch %d] train loss: %.3f, val loss: %.3f, corrcoef: %.3f' % (ep + 1, train_loss, val_loss, corrcoef))
        with open(out + "/log.txt", "a") as f:
            f.write('[epoch %d] train loss: %.3f, val loss: %.3f, coref: %.3f\n' % (ep + 1, train_loss, val_loss, corrcoef))

    print('Finished Training')
    print('Max corrcoef: {}'.format(max_corrcoef))
    with open(out + "/log.txt", "a") as f:
            f.write('Max coref: {}'.format(max_corrcoef))
    path = out + "/model_final"
    torch.save(net.state_dict(), path)

    # Draw graph
    fig = plt.figure(dpi=600)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(x, train_loss_record, label="train", color="red")
    ax2 = ax1.twinx()
    ax2.plot(x, val_loss_record, label="validation", color="blue")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper right')

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax2.set_ylabel("Validation Loss")

    plt.savefig(out + '/Loss.png')


if __name__ == '__main__':
    main()
