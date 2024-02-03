import os
import glob
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.metrics import matthews_corrcoef

from network import RegNetwork
from network import ClsNetwork

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.stats import kde
plt.rcParams["font.size"] = 13.5
np.set_printoptions(threshold=np.inf)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from math import exp

class HybridDataset(Dataset):
    def __init__(self, root=None, mode="test", reg_input_width=15, reg_input_dim=1, cls_input_width=15, cls_input_dim=1):
        assert reg_input_width % 2 == 1, "reg_input_width must be odd number"
        assert cls_input_width % 2 == 1, "cls_input_width must be odd number"
        self.root = root
        self.reg_input_width = reg_input_width
        self.cls_input_width = cls_input_width
        self.reg_input_dim = reg_input_dim + 1 # depthを含める
        self.cls_input_dim = cls_input_dim + 1 # depthを含める

        # 全てのデータのパスを入れる
        data_dir = os.path.join(self.root, mode)
        self.all_data = glob.glob(data_dir + "/*")
        # all_dataは一次元配列

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        with open(self.all_data[idx], "r") as f:
            txt = f.readlines()[0]
        x, y, depth, mag = txt.split(",")
        x, y, depth, mag = int(x), int(y), float(depth), float(mag)
        lbl_data = np.loadtxt(self.all_data[idx], delimiter=',', dtype="float32", skiprows=1)
        len_data = len(lbl_data)
        reg_img = torch.zeros(self.reg_input_dim, len(lbl_data), len(lbl_data))
        half = self.reg_input_width//2
        for i in range(x - half, x + half + 1):
            for j in range(y - half, y + half + 1):
                if 0 <= i < len_data and 0 <= j < len_data:
                    reg_img[0][i][j] = depth/1000
                    for k in range(1, self.reg_input_dim):
                        reg_img[k][i][j] = (mag/10)**k
        cls_img = torch.zeros(self.cls_input_dim, len(lbl_data), len(lbl_data))
        half = self.cls_input_width//2
        for i in range(x - half, x + half + 1):
            for j in range(y - half, y + half + 1):
                if 0 <= i < len_data and 0 <= j < len_data:
                    cls_img[0][i][j] = depth
                    for k in range(1, self.cls_input_dim):
                        cls_img[k][i][j] = exp(mag)
        return reg_img, cls_img, lbl_data
    
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
    parser.add_argument('--batchsize', '-b', type=int, default=128)
    parser.add_argument('--gpu1', type=int, default=-1)
    parser.add_argument('--gpu2', type=int, default=-1)
    parser.add_argument('--reginputwidth', type=int, default=15)
    parser.add_argument('--clsinputwidth', type=int, default=15)
    parser.add_argument('--reginputdim', type=int, default=1)
    parser.add_argument('--clsinputdim', type=int, default=1)
    parser.add_argument('--reglayers', type=int, default=1)
    parser.add_argument('--clslayers', type=int, default=1)
    parser.add_argument('--reghiddendim', type=int, default=1000)
    parser.add_argument('--clshiddendim', type=int, default=1000)
    parser.add_argument('--regdropprob', type=float, default=0.2)
    parser.add_argument('--clsdropprob', type=float, default=0.2)
    parser.add_argument('--regmodel', default='../result/model_best')
    parser.add_argument('--clsmodel', default='../result/model_best')
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()

    print('GPU1: {}'.format(args.gpu1))
    print('GPU2: {}'.format(args.gpu2))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# reg input width: {}'.format(args.reginputwidth))
    print('# cls input width: {}'.format(args.clsinputwidth))
    print('# reg input dim: {}'.format(args.reginputdim))
    print('# cls input dim: {}'.format(args.clsinputdim))
    print('# reg layers: {}'.format(args.reglayers))
    print('# cls layers: {}'.format(args.clslayers))
    print('# reg hidden dim: {}'.format(args.reghiddendim))
    print('# cls hidden dim: {}'.format(args.clshiddendim))
    print('# reg dropout probability: {}'.format(args.regdropprob))
    print('# cls dropout probability: {}'.format(args.clsdropprob))
    print('')

    # Set up a neural network to test
    regnet = RegNetwork(input_dim=args.reginputdim, n_layers=args.reglayers, hidden_dim=args.reghiddendim, drop_prob=args.regdropprob)
    # Load designated network weight
    regnet.load_state_dict(torch.load(args.regmodel))
    print("Loaded {}".format(args.regmodel))
    # Set model to GPU
    if args.gpu1 >= 0:
        # Make a specified GPU current
        print("GPU{} using".format(str(args.gpu1)))
        device1 = 'cuda:' + str(args.gpu1)
        regnet = regnet.to(device1)
	
     # Set up a neural network to test
    clsnet = ClsNetwork(input_dim=args.clsinputdim, n_layers=args.clslayers, hidden_dim=args.clshiddendim, drop_prob=args.clsdropprob)
    # Load designated network weight
    clsnet.load_state_dict(torch.load(args.clsmodel))
    print("Loaded {}".format(args.clsmodel))
    # Set model to GPU
    if args.gpu2 >= 0:
        # Make a specified GPU current
        print("GPU{} using".format(str(args.gpu2)))
        device2 = 'cuda:' + str(args.gpu2)
        clsnet = clsnet.to(device2)

    # Load the data
    testset = HybridDataset(root=args.dataset, mode="test", reg_input_width=args.reginputwidth, reg_input_dim=args.reginputdim, cls_input_width=args.clsinputwidth, cls_input_dim=args.clsinputdim)
    testloader = DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=2)
    print("Testset length: {}".format(len(testset)))

    # Test
    regnet.eval()
    clsnet.eval()
    predict_array = []
    label_array = []
    with torch.no_grad():
        for data in testloader:
            # Get the inputs; data is a list of [inputs, labels]
            reg_images, cls_images, labels = data
            if args.gpu1 >= 0 and args.gpu2 >= 0:
                reg_images = reg_images.to(device1)
                cls_images = cls_images.to(device2)
            # Forward
            reg_outputs = regnet(reg_images)
            cls_outputs = clsnet(cls_images)
            reg_predicted = torch.where(reg_outputs >= 0.5, reg_outputs, torch.zeros_like(reg_outputs))
            _, cls_predicted = torch.max(cls_outputs, 1)
            # To CPU
            reg_predicted, cls_predicted = reg_predicted.to(device2), cls_predicted.to(device2)
            assert reg_predicted.size() == cls_predicted.size()
            predicted = torch.where(cls_predicted > 0, reg_predicted, torch.zeros_like(reg_predicted))
            # Record
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
    plt.title("Hybrid Predictions vs Ground Truth")
    plt.xlabel("ground truth")
    plt.ylabel("predictions")
    # グリッドを追加
    # plt.grid(True)

    plt.tight_layout()
    out = os.path.dirname(args.regmodel)
    plt.savefig(out+ "/hybrid_scatter.png", dpi=800)

   

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
    plt.title("Hybrid Predictions vs Ground Truth after 0 removal")
    plt.xlabel("ground truth")
    plt.ylabel("predictions")
    # グリッドを追加
    # plt.grid(True)

    plt.tight_layout()
    plt.savefig(out+ "/hybrid_scatter_zeroremove.png", dpi=800)
    print("Instrumental correlation coefficient after 0 removal: ", np.corrcoef(x, y)[0][1])
    label_array_to_seismic = [InstrumentalIntensity2SesimicIntensity(d) for d in x]
    predict_array_to_seismic = [InstrumentalIntensity2SesimicIntensity(d) for d in y]
    print("JMA seismic scale correlation coefficient after 0 removal: ", np.corrcoef(np.array(label_array_to_seismic), np.array(predict_array_to_seismic))[0][1])
    label_array_to_instrumental = [SesimicIntensity2InstrumentalIntensity(x) for x in label_array_to_seismic]
    predict_array_to_instrumental = [SesimicIntensity2InstrumentalIntensity(x) for x in predict_array_to_seismic]
    print("Scaled instrumental correlation coefficient after 0 removal (instrumental -> JMA scale -> scaled instrumental): ", np.corrcoef(np.array(label_array_to_instrumental), np.array(predict_array_to_instrumental))[0][1])
    print("MCC after 0 removal: ", matthews_corrcoef(np.array(label_array_to_seismic), np.array(predict_array_to_seismic)))

    fig, ax = plt.subplots()

    # Filter the data based on the axis limits
    mask = (x_min <= x) & (x <= x_max) & (y_min <= y) & (y <= y_max)
    x_filtered, y_filtered = x[mask], y[mask]
    # print("correlation coefficient after filter: ", np.corrcoef(x_filtered, y_filtered)[0][1])

    # Create a 2D kernel density estimation with the filtered data
    k = kde.gaussian_kde([x_filtered, y_filtered])
    xi, yi = np.mgrid[x_min:x_max:nbins*1j, y_min:y_max:nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # Create a colormap
    # colors = plt.cm.Blues(np.linspace(0, 1, 128))
    # mymap = LinearSegmentedColormap.from_list('my_colormap', colors)

    # Plot the density estimation
    ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.jet, shading='auto')

    # Add a colorbar
    plt.colorbar(ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.jet, shading='auto'))
    ax.axis("tight")

    # Plot the identity line
    ax.plot([x_min, x_max], [y_min, y_max], 'r--')

    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Hybrid Predictions')
    ax.set_title('Density Heatmap of Hybrid Predictions vs. GT', pad=15)


    plt.tight_layout()
    plt.savefig(out+ "/hybrid_heatmap.png", dpi=800, bbox_inches="tight")

if __name__ == '__main__':
    test()