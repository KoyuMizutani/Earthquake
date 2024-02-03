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

from network import ClsNetwork
from dataset import ClsDataset

# from sklearn.metrics import matthews_corrcoef


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

    print('# Classification Model Training')
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
        f.write('# Classification Model Training\n')
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

    net = ClsNetwork(n_class=10, input_dim=args.inputdim, n_layers=args.layers, hidden_dim=args.hiddendim, drop_prob=args.dropprob)
    if args.resume:
        net.load_state_dict(torch.load(args.resume))
        print('Loaded {}'.format(args.resume))

    if args.gpu >= 0:
        print("GPU using")
        device = 'cuda:' + str(args.gpu)
        net = net.to(device)

    weights = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    if args.gpu >= 0:
        weights = weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights, reduction="sum")

    # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9) #lr 0.001
    optimizer = optim.Adam(net.parameters(), args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch)

    transform = transforms.Compose([transforms.ToTensor()])

    trainset = ClsDataset(root=args.dataset, mode="train", transform=transform, input_width=args.inputwidth, input_dim=args.inputdim)
    valset = ClsDataset(root=args.dataset, mode="val", transform=transform, input_width=args.inputwidth, input_dim=args.inputdim)
    trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=args.batchsize, shuffle=True, num_workers=2)

    print("# trainset size {}".format(len(trainset)))
    print("# valset size {}".format(len(valset)))

    # Setup result holder
    x = []
    train_loss_record = []
    val_loss_record = []
    max_coref = -1.0
    # Train
    for ep in range(args.epoch):
        net.train()
        train_loss = 0.0
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
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Add loss
            train_loss += loss.item()
        scheduler.step()

        train_loss /= len(trainset) * 64 * 64

        # Save the model
        if (ep + 1) % args.frequency == 0:
            path = out + "/model_" + str(ep + 1)
            torch.save(net.state_dict(), path)

        # Validation
        net.eval()
        val_loss = 0.0
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
                _, predicted = torch.max(outputs, 1)
                # Check whether estimation is right
                for i in range(len(predicted)):
                    for j in range(len(predicted[i])):
                        for k in range(len(predicted[i][j])):
                            label = labels[i][j][k].item()
                            predict = predicted[i][j][k].item()
                            label_array.append(label)
                            predict_array.append(predict)

                loss = criterion(outputs, labels)
                # Add loss
                val_loss += loss.item()
        
        val_loss /= len(valset) * 64 * 64
        # Record result
        x.append(ep + 1)
        train_loss_record.append(train_loss)
        val_loss_record.append(val_loss)
        coref = np.corrcoef(np.array(label_array), np.array(predict_array))[0][1]
        if max_coref < coref:
            max_coref = coref
            path = out + "/model_best"
            torch.save(net.state_dict(), path)

        # Report loss of the epoch
        print('[epoch %d] train loss: %.3f, val loss: %.3f, coref: %.3f' % (ep + 1, train_loss, val_loss, coref))
        with open(out + "/log.txt", "a") as f:
            f.write('[epoch %d] train loss: %.3f, val loss: %.3f, coref: %.3f\n' % (ep + 1, train_loss, val_loss, coref))

    print('Finished Training')
    print('Max coref: {}'.format(max_coref))
    with open(out + "/log.txt", "a") as f:
            f.write('Max coref: {}'.format(max_coref))
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
