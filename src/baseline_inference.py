import os
import glob
import numpy as np
import csv

import argparse

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.stats import kde
plt.rcParams["font.size"] = 13.5
np.set_printoptions(threshold=np.inf)
    
INPUT_WIDTH = 64
START_LONGITUDE = 128.0
START_LATITUDE = 46.0
END_LONGITUDE = 146.0
END_LATITUDE = 30.0
MASK = np.loadtxt("/home/mizutani/exp/EQPrediction/AI/src/mask.csv", delimiter=',', dtype=int)
MASK = np.where(MASK > 0, np.ones_like(MASK), np.zeros_like(MASK))
AMP = np.loadtxt("/home/mizutani/exp/EQPrediction/AI/src/amp.csv", delimiter=',', dtype=float)

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

def net(x, y, depth, mag):
    input_width = INPUT_WIDTH
    start_longitude = START_LONGITUDE * (np.pi/180)  # Convert degree to radian
    end_longitude = END_LONGITUDE * (np.pi/180)  # Convert degree to radian
    start_latitude = START_LATITUDE * (np.pi/180)  # Convert degree to radian
    end_latitude = END_LATITUDE * (np.pi/180)  # Convert degree to radian
    mask = MASK  # This assumes MASK is a numpy array
    
    amp = AMP ## 工学的基盤からの増幅率

    batch_size = len(x)

    # Calculate grid of coordinates for cells
    cell_width = (end_longitude - start_longitude) / input_width
    cell_longs = np.linspace(start_longitude + cell_width / 2, end_longitude - cell_width / 2, input_width)[None, :]  # [1, input_width]
    cell_longs = np.tile(cell_longs, (input_width, 1))  # [input_width, input_width]
    
    cell_height = (end_latitude - start_latitude) / input_width
    cell_lats = np.linspace(start_latitude + cell_height / 2, end_latitude - cell_height / 2, input_width)[:, None]  # [input_width, 1]
    cell_lats = np.tile(cell_lats, (1, input_width))  # [input_width, input_width]

    # Calculate epicenter coordinates for each batch
    epicenter_longs = start_longitude + (end_longitude - start_longitude) * (x[:, None, None] + 0.5) / input_width  # [batch_size, 1, 1]
    epicenter_lats = start_latitude + (end_latitude - start_latitude) * (y[:, None, None] + 0.5) / input_width  # [batch_size, 1, 1]

    # Haversine formula to calculate the distance between two points on the earth
    dlon = cell_longs - epicenter_longs  # [batch_size, input_width, input_width]
    dlat = cell_lats - epicenter_lats  # [batch_size, input_width, input_width]
    a = np.sin(dlat / 2)**2 + np.cos(epicenter_lats) * np.cos(cell_lats) * np.sin(dlon / 2)**2  # [batch_size, input_width, input_width]
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))  # [batch_size, input_width, input_width]
    X = np.sqrt((6371.0 * c) ** 2 + depth[:, None, None] ** 2)  # [batch_size, input_width, input_width]

    mag = 0.78 * mag + 1.08  # [batch_size] convert to moment magnitude

    log_PGV_b = 0.58 * mag[:, None, None] + 0.0038 * depth[:, None, None] - 1.29 - np.log10(X + 0.0028 * 10 ** (0.5 * mag[:, None, None])) - 0.002 * X  # [batch_size, input_width, input_width]
    PGV_b = 10**(log_PGV_b)  # [batch_size, input_width, input_width]
    PGV = PGV_b * amp  # [batch_size, input_width, input_width]
    I_high = 2.002 + 2.603 * np.log10(PGV) - 0.213 * (np.log10(PGV)) ** 2  # [batch_size, input_width, input_width]
    I_low = 2.165 + 2.262 * np.log10(PGV)  # [batch_size, input_width, input_width]
    h = np.where(I_high > 4.0, I_high, I_low)  # [batch_size, input_width, input_width]

    # Apply mask
    h = h * mask  # [batch_size, input_width, input_width]

    return h

# -x 38 -y 35 -depth 13.08 -mag 6.8
def inference():
    parser = argparse.ArgumentParser(description='Earthquaker')
    parser.add_argument('--x', '-x', default='0',
                        help='x zahyou')
    parser.add_argument('--y', '-y', default='0',
                        help='y zahyou')
    parser.add_argument('--depth', '-depth', default='10',
                        help='depth of shingen')
    parser.add_argument('--magnitude', '-mag', default='7',
                        help='magnitude of earthquake')
    args = parser.parse_args()
    # Load the data
    x, y, depth, mag = np.array([[int(args.x)]]), np.array([[int(args.y)]]), np.array([[float(args.depth)]]), np.array([[float(args.magnitude)]])
    # print("Testset length: {}".format(len(x)))

    # Test
    outputs = net(x, y, depth, mag)
    predicted = outputs
    predicted = np.where(outputs >= 0.5, outputs, np.zeros_like(outputs))
    pre_list = np.array(predicted).squeeze().tolist()
    pre_list = [[InstrumentalIntensity2SesimicIntensity(pre) for pre in pres] for pres in pre_list]
    with open('predicted_baseline_data.csv', 'w') as file:
        writer = csv.writer(file, lineterminator=',')
        writer.writerows(pre_list)
    print("predicted_data.csv is created!")


if __name__ == '__main__':
    inference()