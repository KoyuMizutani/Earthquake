import os
import glob
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.stats import kde
plt.rcParams["font.size"] = 13.5
np.set_printoptions(threshold=np.inf)

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

def create_dataset():
    data_dir = "../data_reg/test"
    all_data_paths = glob.glob(data_dir + "/*")
    # all_dataは一次元配列

    x_ = []
    y_ = []
    depth_ = []
    mag_ = []
    all_outputs = []
    for path in all_data_paths:
        with open(path, "r") as f:
            txt = f.readlines()[0]
        x, y, depth, mag = txt.split(",")
        x, y, depth, mag = int(x), int(y), float(depth), float(mag)
        output = np.loadtxt(path, delimiter=',', dtype=float, skiprows=1)
        x_.append(x)
        y_.append(y)
        depth_.append(depth)
        mag_.append(mag)
        all_outputs.append(output)
    return np.array(x_), np.array(y_), np.array(depth_), np.array(mag_), np.array(all_outputs)
    
INPUT_WIDTH = 64
START_LONGITUDE = 128.0
START_LATITUDE = 46.0
END_LONGITUDE = 146.0
END_LATITUDE = 30.0
MASK = np.loadtxt("/home/mizutani/exp/EQPrediction/AI/src/mask.csv", delimiter=',', dtype=int)
MASK = np.where(MASK > 0, np.ones_like(MASK), np.zeros_like(MASK))
AMP = np.loadtxt("/home/mizutani/exp/EQPrediction/AI/src/amp.csv", delimiter=',', dtype=float)

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
    # Load the data
    x, y, depth, mag, all_outputs = create_dataset()
    print("Testset length: {}".format(len(x)))

    # Test
    predict_array = []
    label_array = []
    outputs = net(x, y, depth, mag)
    predicted = outputs
    predicted = np.where(outputs >= 0.5, outputs, np.zeros_like(outputs))
    # Check whether estimation is right
    label_array += all_outputs.reshape(-1).tolist()
    predict_array += predicted.reshape(-1).tolist()

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
    plt.title("Baseline Predictions vs Ground Truth")
    plt.xlabel("ground truth")
    plt.ylabel("predictions")
    # グリッドを追加
    # plt.grid(True)

    plt.tight_layout()
    out = "../fig"
    plt.savefig(out+ "/baseline_scatter.png", dpi=800)

   

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
    plt.title("Baseline Predictions vs Ground Truth after 0 removal")
    plt.xlabel("ground truth")
    plt.ylabel("predictions")
    # グリッドを追加
    # plt.grid(True)

    plt.tight_layout()
    plt.savefig(out+ "/baseline_scatter_zeroremove.png", dpi=800)
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
    # colors = plt.cm.Blues(np.linspace(0, 1, 128))
    # mymap = LinearSegmentedColormap.from_list('my_colormap', colors)

    # Plot the density estimation
    ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.jet, shading='auto')

    # Add a colorbar
    plt.colorbar(ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.jet, shading='auto'))

    # Plot the identity line
    ax.plot([x_min, x_max], [y_min, y_max], 'r--')
    ax.axis("tight")
	
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Baseline Predictions')
    ax.set_title('Density Heatmap of Baseline Predictions vs. GT', pad=15)


    plt.tight_layout()
    plt.savefig(out+ "/baseline_heatmap.png", dpi=800, bbox_inches="tight")

if __name__ == '__main__':
    test()