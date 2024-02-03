import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import matthews_corrcoef

from network import RegNetwork
from dataset import RegDataset

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.stats import kde
plt.rcParams["font.size"] = 13.5
np.set_printoptions(threshold=np.inf)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

def InstrumentalIntensity2SesimicIntensity(II):
	if II < 0.5:
		return 0
	elif II < 1.5:
		return 1
	elif II < 2.5:
		return 2
	elif II < 3.5:
		return 3
	elif II < 4.5:
		return 4
	elif II < 5.0:
		return 5	#5-
	elif II < 5.5:
		return 6	#5+
	elif II < 6.0:
		return 7	#6-
	elif II < 6.5:
		return 8	#6+
	else:
		return 9	#7

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
    net = RegNetwork(input_dim=args.inputdim, n_layers=args.layers, hidden_dim=args.hiddendim, drop_prob=args.dropprob)
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
    testset = RegDataset(root=args.dataset, mode="test", input_width=args.inputwidth, input_dim=args.inputdim)
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
            predicted = torch.where(outputs >= 0.5, outputs, torch.zeros_like(outputs))
            # Check whether estimation is right
            label_array += labels.reshape(-1).to("cpu").tolist()
            predict_array += predicted.reshape(-1).to("cpu").tolist()
	
    label_array_to_binary = [1 if x >= 0.5 else 0 for x in label_array]
    predict_array_to_binary = [1 if x >= 0.5 else 0 for x in predict_array]
    print("Confusion matrix: ", confusion_matrix(np.array(label_array_to_binary), np.array(predict_array_to_binary)))
    print("f1 score: ", f1_score(np.array(label_array_to_binary), np.array(predict_array_to_binary)))

    # List of classes
    print("Instrumental correlation coefficient: ", np.corrcoef(np.array(label_array), np.array(predict_array))[0][1])
    label_array_to_seismic = [InstrumentalIntensity2SesimicIntensity(x) for x in label_array]
    predict_array_to_seismic = [InstrumentalIntensity2SesimicIntensity(x) for x in predict_array]
    print("JMA seismic scale correlation coefficient: ", np.corrcoef(np.array(label_array_to_seismic), np.array(predict_array_to_seismic))[0][1])
    print("MCC: ", matthews_corrcoef(np.array(label_array_to_seismic), np.array(predict_array_to_seismic)))
    label_array_to_instrumental = [SesimicIntensity2InstrumentalIntensity(x) for x in label_array_to_seismic]
    predict_array_to_instrumental = [SesimicIntensity2InstrumentalIntensity(x) for x in predict_array_to_seismic]
    print("Scaled instrumental correlation coefficient (instrumental -> JMA scale -> scaled instrumental): ", np.corrcoef(np.array(label_array_to_instrumental), np.array(predict_array_to_instrumental))[0][1])

    plt.rcParams["font.size"] = 13.5

    x_min, x_max = 0, 7
    y_min, y_max = 0, 7

    # 散布図の作成
    plt.scatter(label_array, predict_array, s=3)
    # Plot the identity line
    plt.plot([x_min, x_max], [y_min, y_max], 'r--')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.axis('equal')

    # タイトルとラベルを追加
    plt.title("Regression Predictions vs Ground Truth")
    plt.xlabel("ground truth")
    plt.ylabel("predictions")
    # グリッドを追加
    # plt.grid(True)

    plt.tight_layout()
    out = os.path.dirname(args.model)
    plt.savefig(out+ "/regression_scatter.png", dpi=800)

   

    # Create a figure
    fig, ax = plt.subplots()

    # Create a 2D kernel density estimation
    nbins = 300
    label_array = np.array(label_array)
    predict_array = np.array(predict_array)
    x, y = label_array[(label_array != 0.0) & (predict_array != 0.0)], predict_array[(label_array != 0.0) & (predict_array != 0.0)]
    # 散布図の作成
    plt.scatter(x, y, s=3)
    # Plot the identity line
    plt.plot([x_min, x_max], [y_min, y_max], 'r--')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.axis('equal')

    # タイトルとラベルを追加
    plt.title("Regression Predictions vs Ground Truth after 0 removal")
    plt.xlabel("ground truth")
    plt.ylabel("predictions")
    # グリッドを追加
    # plt.grid(True)

    plt.tight_layout()
    plt.savefig(out+ "/regression_scatter_zeroremove.png", dpi=800)
    print("Instrumental correlation coefficient after 0 removal: ", np.corrcoef(x, y)[0][1])
    label_array_to_seismic = [InstrumentalIntensity2SesimicIntensity(d) for d in x]
    predict_array_to_seismic = [InstrumentalIntensity2SesimicIntensity(d) for d in y]
    print("JMA seismic scale correlation coefficient after 0 removal: ", np.corrcoef(np.array(label_array_to_seismic), np.array(predict_array_to_seismic))[0][1])
    label_array_to_instrumental = [SesimicIntensity2InstrumentalIntensity(x) for x in label_array_to_seismic]
    predict_array_to_instrumental = [SesimicIntensity2InstrumentalIntensity(x) for x in predict_array_to_seismic]
    print("Scaled instrumental correlation coefficient after 0 removal (instrumental -> JMA scale -> scaled instrumental): ", np.corrcoef(np.array(label_array_to_instrumental), np.array(predict_array_to_instrumental))[0][1])
    print("MCC after 0 removal: ", matthews_corrcoef(np.array(label_array_to_seismic), np.array(predict_array_to_seismic)))



    # Filter the data based on the axis limits
    mask = (x_min <= x) & (x <= x_max) & (y_min <= y) & (y <= y_max)
    x_filtered, y_filtered = x[mask], y[mask]
    # print("correlation coefficient after filter: ", np.corrcoef(x_filtered, y_filtered)[0][1])

    # Create a 2D kernel density estimation with the filtered data
    k = kde.gaussian_kde([x_filtered, y_filtered])
    xi, yi = np.mgrid[x_min:x_max:nbins*1j, y_min:y_max:nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # Create a colormap
    colors = plt.cm.Blues(np.linspace(0, 1, 128))
    mymap = LinearSegmentedColormap.from_list('my_colormap', colors)

    # Plot the density estimation
    ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=mymap, shading='auto')

    # Add a colorbar
    plt.colorbar(ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=mymap, shading='auto'))

    # Plot the identity line
    ax.plot([x_min, x_max], [y_min, y_max], 'r--')

    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Predictions')
    ax.set_title('Heatmap of Regression Predictions vs Ground Truth')


    plt.tight_layout()
    plt.savefig(out+ "/regression_heatmap.png", dpi=800)

if __name__ == '__main__':
    test()
