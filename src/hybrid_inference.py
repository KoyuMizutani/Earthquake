import argparse
import torch
import numpy as np
from network import RegNetwork
from network import ClsNetwork
import csv
len_data = 64
# 予測プログラム
from math import exp

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
    parser.add_argument('--x', '-x', default='0',
                        help='x zahyou')
    parser.add_argument('--y', '-y', default='0',
                        help='y zahyou')
    parser.add_argument('--depth', '-depth', default='10',
                        help='depth of shingen')
    parser.add_argument('--magnitude', '-mag', default='7',
                        help='magnitude of earthquake')
    args = parser.parse_args()

    print('GPU1: {}'.format(args.gpu1))
    print('GPU2: {}'.format(args.gpu2))
    print('# x: {}'.format(args.x))
    print('# y: {}'.format(args.y))
    print('# depth: {}'.format(args.depth))
    print('# magnitude: {}'.format(args.magnitude))
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
	
    # Load the input
    reg_inputs = torch.zeros(1, args.reginputdim + 1, len_data, len_data)
    assert args.reginputwidth % 2 == 1 # 奇数であることを確かめる
    half = args.reginputwidth//2
    for i in range(int(args.x) - half, int(args.x) + half + 1):
        for j in range(int(args.y) - half, int(args.y) + half + 1):
            if 0 <= i < len_data and 0 <= j < len_data:
                reg_inputs[0][0][i][j] = float(args.depth) / 1000
                for k in range(1, args.reginputdim + 1):
                    reg_inputs[0][k][i][j] = (float(args.magnitude)/10)**k
    cls_inputs = torch.zeros(1, args.clsinputdim + 1, len_data, len_data)
    assert args.clsinputwidth % 2 == 1 # 奇数であることを確かめる
    half = args.clsinputwidth//2
    for i in range(int(args.x) - half, int(args.x) + half + 1):
        for j in range(int(args.y) - half, int(args.y) + half + 1):
            if 0 <= i < len_data and 0 <= j < len_data:
                cls_inputs[0][0][i][j] = float(args.depth)
                for k in range(1, args.clsinputdim + 1):
                    cls_inputs[0][k][i][j] = exp(float(args.magnitude))

    if args.gpu1 >= 0:
        reg_inputs = reg_inputs.to(device1)
    if args.gpu2 >= 0:
        cls_inputs = cls_inputs.to(device2)
    regnet.eval()
    clsnet.eval()
    
    # Inference
    reg_outputs = regnet(reg_inputs)
    cls_outputs = clsnet(cls_inputs)
    reg_predicted = torch.where(reg_outputs >= 0.5, reg_outputs, torch.zeros_like(reg_outputs))
    _, cls_predicted = torch.max(cls_outputs, 1)
    # To the same GPU
    reg_predicted, cls_predicted = reg_predicted.to(device2), cls_predicted.to(device2)
    assert reg_predicted.size() == cls_predicted.size()
    predicted = torch.where(cls_predicted > 0, reg_predicted, torch.zeros_like(reg_predicted))

    pre_list = np.array(cls_predicted.to("cpu")).squeeze().tolist()
    with open('hogehogefugafuga.csv', 'w') as file:
        writer = csv.writer(file, lineterminator=',')
        writer.writerows(pre_list)
    print("predicted_data.csv is created!")

    pre_list = predicted.to("cpu").detach().numpy().squeeze().tolist()
    pre_list = [[InstrumentalIntensity2SesimicIntensity(pre) for pre in pres] for pres in pre_list]
    with open('predicted_hybrid_data.csv', 'w') as file:
        writer = csv.writer(file, lineterminator=',')
        writer.writerows(pre_list)
    print("predicted_data.csv is created!")


if __name__ == '__main__':
    reg_inference()