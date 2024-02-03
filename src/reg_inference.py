import argparse
import torch
import numpy as np
from network import RegNetwork
import csv
len_data = 64
# 予測プログラム

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

def reg_inference():
    parser = argparse.ArgumentParser(description='Earthquaker')
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
    parser.add_argument('--x', '-x', default='0',
                        help='x zahyou')
    parser.add_argument('--y', '-y', default='0',
                        help='y zahyou')
    parser.add_argument('--depth', '-depth', default='10',
                        help='depth of shingen')
    parser.add_argument('--magnitude', '-mag', default='7',
                        help='magnitude of earthquake')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# x: {}'.format(args.x))
    print('# y: {}'.format(args.y))
    print('# depth: {}'.format(args.depth))
    print('# magnitude: {}'.format(args.magnitude))
    print('# input width: {}'.format(args.inputwidth))
    print('# input dim: {}'.format(args.inputdim))
    print('# layers: {}'.format(args.layers))
    print('# hidden dim: {}'.format(args.hiddendim))
    print('# dropout probability: {}'.format(args.dropprob))
    print('')
    # Set up a neural network to test
    net = RegNetwork(input_dim=args.inputdim, n_layers=args.layers, hidden_dim=args.hiddendim, drop_prob=args.dropprob)
    # Load designated network weight
    net.load_state_dict(torch.load(
        args.model, map_location=torch.device('cpu')))
    # Set model to GPU
    if args.gpu >= 0:
        # Make a specified GPU current
        print("GPU using")
        device = 'cuda:' + str(args.gpu)
        net = net.to(device)
    # Load the input
    inputs = torch.zeros(1, args.inputdim + 1, len_data, len_data)
    assert args.inputwidth % 2 == 1 # 奇数であることを確かめる
    half = args.inputwidth//2
    for i in range(int(args.x)-half, int(args.x)+half + 1):
        for j in range(int(args.y)-half, int(args.y)+half + 1):
            if 0 <= i < len_data and 0 <= j < len_data:
                inputs[0][0][i][j] = float(args.depth) / 1000
                for k in range(1, args.inputdim + 1):
                    inputs[0][k][i][j] = (float(args.magnitude)/10)**k

    if args.gpu >= 0:
        inputs = inputs.to(device)

    outputs = net(inputs)
    predicted = torch.where(outputs >= 0.5, outputs, torch.zeros_like(outputs))
    pre_list = predicted.to("cpu").detach().numpy().squeeze().tolist()
    pre_list = [[InstrumentalIntensity2SesimicIntensity(pre) for pre in pres] for pres in pre_list]
    with open('predicted_reg_data.csv', 'w') as file:
        writer = csv.writer(file, lineterminator=',')
        writer.writerows(pre_list)
    print("predicted_data.csv is created!")


if __name__ == '__main__':
    reg_inference()
