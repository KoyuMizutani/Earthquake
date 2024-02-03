import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import matthews_corrcoef

from dataset import ClsDataset
from network import ClsNetwork

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.rcParams["font.size"] = 13.5

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

def SesimicIntensity2InstrumentalIntensity(II):
	if II == 0:
		return 0.25
	elif II == 1:
		return 1
	elif II == 2:
		return 2
	elif II == 3:
		return 3
	elif II == 4:
		return 4
	elif II == 5:
		return 4.75	#5-
	elif II == 6:
		return 5.25	#5+
	elif II == 7:
		return 5.75	#6-
	elif II == 8:
		return 6.25	#6+
	else:
		return 6.75	#7

def test():
    parser = argparse.ArgumentParser(description='Earthquaker')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
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
    parser.add_argument('--model', '-m', default='../result/model_best',
                        help='Path to the model for test')
    parser.add_argument('--dataset', '-d', required=True,
                        help='Root directory of dataset')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# input width: {}'.format(args.inputwidth))
    print('# input dim: {}'.format(args.inputdim))
    print('# layers: {}'.format(args.layers))
    print('# hidden dim: {}'.format(args.hiddendim))
    print('# dropout probability: {}'.format(args.dropprob))
    print('')

    # Set up a neural network to test
    net = ClsNetwork(n_class=10, input_dim=args.inputdim, n_layers=args.layers, hidden_dim=args.hiddendim, drop_prob=args.dropprob)
    # Load designated network weight
    net.load_state_dict(torch.load(args.model))
    print("Loaded {}".format(args.model))
    # Set model to GPU
    if args.gpu >= 0:
        # Make a specified GPU current
        print("GPU using")
        device = 'cuda:' + str(args.gpu)
        net = net.to(device)

    # Load the data
    transform = transforms.Compose([transforms.ToTensor()])
    testset = ClsDataset(root=args.dataset, mode="test", transform=transform, input_width=args.inputwidth, input_dim=args.inputdim)
    testloader = DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=2)
    print("Testset length: {}".format(len(testset)))

    # Test
    net.eval()
    predict_array = []
    label_array = []
    matrix = [[0 for _ in range(10)] for _ in range(10)]
    with torch.no_grad():
        for data in testloader:
            # Get the inputs; data is a list of [inputs, labels]
            images, labels = data
            if args.gpu >= 0:
                images = images.to(device)
                labels = labels.to(device)
            # Forward
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            # Check whether estimation is right
            for i in range(len(predicted)):
                for j in range(len(predicted[i])):
                    for k in range(len(predicted[i][j])):
                        label = labels[i][j][k].item()
                        predict = predicted[i][j][k].item()
                        label_array.append(label)
                        predict_array.append(predict)
                        matrix[label][predict] += 1
	
    label_array_to_binary = [1 if x >= 0.5 else 0 for x in label_array]
    predict_array_to_binary = [1 if x >= 0.5 else 0 for x in predict_array]
    print("Confusion matrix: ", confusion_matrix(np.array(label_array_to_binary), np.array(predict_array_to_binary)))
    print("f1 score: ", f1_score(np.array(label_array_to_binary), np.array(predict_array_to_binary)))

    # List of classes
    classes = ("0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7")
    print("matthews corrcoef", matthews_corrcoef(np.array(label_array), np.array(predict_array)))
    coref = np.corrcoef(np.array(label_array), np.array(predict_array))[0][1]
    print("JMA seismic scale corrcoef:", coref)
    label_array = [SesimicIntensity2InstrumentalIntensity(x) for x in label_array]
    predict_array = [SesimicIntensity2InstrumentalIntensity(x) for x in predict_array]
    coref = np.corrcoef(np.array(label_array), np.array(predict_array))[0][1]
    print("Instrumental seismic intensity corrcoef:", coref)
    label_array = np.array(label_array)
    predict_array = np.array(predict_array)
    x, y = label_array[(label_array != 0.25) & (predict_array != 0.25)], predict_array[(label_array != 0.25) & (predict_array != 0.25)]
    coref = np.corrcoef(x, y)[0][1]
    print("Instrumental seismic intensity corrcoef (0 removed):", coref)
    # print("↓↓↓ローカルで混同行列実行用")
    # print(matrix)
    plt.rcParams["font.size"] = 13.5
    fig = plt.figure(dpi=800)
    label = [9,8,7,6,5,4,3,2,1,0]    
    matrix = np.array(matrix)
    sm = matrix.sum(axis=1).reshape(10,1)
    matrix = matrix / sm
    matrix = np.flipud(matrix.T)
    arr_mask = (matrix < 0.01)
    blue = sns.light_palette("blue", 1000)
    sns.heatmap(matrix, vmax=1, vmin=0, cmap=blue, annot=True, linewidths=0.2, linecolor="black", yticklabels=label, fmt=".2f", annot_kws={"size":11.5}, mask=arr_mask, cbar_kws={'label': 'Ratio for each Intensity'})
    plt.xlabel("True Intensity")
    plt.ylabel("Predicted Intensity")

    plt.tight_layout()
    out = os.path.dirname(args.model)
    plt.savefig(out+ "/cm.png")

if __name__ == '__main__':
    test()
