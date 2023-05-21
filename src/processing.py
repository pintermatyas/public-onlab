import numpy as np
import pathlib
import pandas as pd
import scipy.io
import scipy
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal
import scipy.optimize as optimize
from scipy.signal import find_peaks


def upsample_signal(signal, upsample_ratio=8):
    upsampled_signal = scipy.signal.resample(signal, num=upsample_ratio*len(signal))
    return upsampled_signal

def peak_detection_correlated(measurement_data, idx, **kwargs):
    peaks = []
    for dot_num in range(7):
            measurement = measurement_data[idx].query("dpePortSel == @dot_num")
            
            signal1_noisy = np.concatenate((np.zeros(measurement.iloc[0]['Kmaxm1']),  np.abs(measurement.iloc[0]['RxyT'])))
            signal2_noisy = np.concatenate((np.zeros(measurement.iloc[1]['Kmaxm1']), np.abs(measurement.iloc[1]['RxyT'])))
            signal3_noisy = np.concatenate((np.zeros(measurement.iloc[2]['Kmaxm1']), np.abs(measurement.iloc[2]['RxyT'])))
            signal4_noisy = np.concatenate((np.zeros(measurement.iloc[3]['Kmaxm1']), np.abs(measurement.iloc[3]['RxyT'])))
                        
            # Calculate cross-correlation between all pairs of signals
            corr12 = np.correlate(signal1_noisy, signal2_noisy, mode='full')
            corr13 = np.correlate(signal1_noisy, signal3_noisy, mode='full')
            corr14 = np.correlate(signal1_noisy, signal4_noisy, mode='full')
            corr23 = np.correlate(signal2_noisy, signal3_noisy, mode='full')
            corr24 = np.correlate(signal2_noisy, signal4_noisy, mode='full')
            corr34 = np.correlate(signal3_noisy, signal4_noisy, mode='full')

            # Padding cross-correlation signals to ensure they have the same length
            max_length = max(len(corr12), len(corr13), len(corr14), len(corr23), len(corr24), len(corr34))
            corr12 = np.pad(corr12, (0, max_length - len(corr12)), mode='constant')
            corr13 = np.pad(corr13, (0, max_length - len(corr13)), mode='constant')
            corr14 = np.pad(corr14, (0, max_length - len(corr14)), mode='constant')
            corr23 = np.pad(corr23, (0, max_length - len(corr23)), mode='constant')
            corr24 = np.pad(corr24, (0, max_length - len(corr24)), mode='constant')
            corr34 = np.pad(corr34, (0, max_length - len(corr34)), mode='constant')

            # Combine cross-correlations
            sum_corr = corr12 + corr13 + corr14 + corr23 + corr24 + corr34

            # Find the peak of the combined cross-correlation
            peak_index = np.argmax(sum_corr)
            peak_time = peak_index - len(signal1_noisy)//2
            peaks.append(peak_time)

    peaks = [index2toa(p, M=1) for p in peaks]
    return peaks

def find_first_peak_above_threshold(signal, threshold_ratio):
    signal = np.array(signal)
    peaks, _ = find_peaks(signal)
    peak_values = signal[peaks]

    highest_peak = np.max(peak_values)
    threshold = threshold_ratio * highest_peak

    above_threshold_peaks = peaks[peak_values > threshold]

    if len(above_threshold_peaks) > 0:
        first_peak = above_threshold_peaks[0]
        return first_peak
    else:
        return None

def inverse_weighted_average(data, weights):
    data = np.array(data)
    weights = np.array(weights)
    normalized_weights = 1 / weights
    weighted_sum = np.sum(data * normalized_weights)
    sum_of_weights = np.sum(normalized_weights)
    # print(normalized_weights)
    weighted_average = weighted_sum / sum_of_weights
    return weighted_average
    

def tdoa_new(time_of_arrivals, antenna_coords):

    # Difference of between the antenna's time of arrivals converted into difference of distance
    diffs = []

    c = 3E8

    for i in range(len(time_of_arrivals)):
        for j in range(len(time_of_arrivals)):
            difference_of_arrival_in_time = abs(time_of_arrivals[i] - time_of_arrivals[j])
            difference_of_distance = difference_of_arrival_in_time*c
            diffs.append(difference_of_distance)

    # Height difference between UE and DOTs is approx. 3m
    height_diff = 3

    def fun(coordinates):
        x = coordinates[0]
        y = coordinates[1]
        
        # Difference of distance between the estimated position and the DOTs
        dist_difference = []
        
        for i in range(len(antenna_coords)):
            for j in range(len(antenna_coords)):
                difference = np.sqrt((x - antenna_coords[i][0])**2 + (y - antenna_coords[i][1])**2 + height_diff**2) - np.sqrt((x - antenna_coords[j][0])**2 + (y - antenna_coords[j][1])**2 + height_diff**2)
                dist_difference.append(abs(difference))
            
        return [diffs[i] - dist_difference[i] for i in range(len(diffs))]
    
    ant_x = [antenna_coord[0] for antenna_coord in antenna_coords]
    ant_y = [antenna_coord[1] for antenna_coord in antenna_coords]
    
    mean_x = inverse_weighted_average(ant_x, time_of_arrivals)
    mean_y = inverse_weighted_average(ant_y, time_of_arrivals)

    initial_guess = [mean_x, mean_y]
    
    res = optimize.least_squares(fun, initial_guess)
    return res.x

def estimate_position(distances, positions):
    # Define the objective function to minimize
    def objective_function(point):
        return np.sum((np.linalg.norm(point - positions, axis=1) - distances) ** 2)

    # Initial guess for the point's position
    initial_guess = np.mean(positions, axis=0)

    # Minimize the objective function to estimate the point's position
    result = scipy.optimize.minimize(objective_function, initial_guess)

    return result.x

def load_dot_coordinates(coordinate_data_filename, measurement_path):
    recievers = pd.read_csv(pathlib.Path(coordinate_data_filename), index_col='id_short')
    active_dot_ids = [int(x) for x in measurement_path.stem.split('_')[-7:]]
    active_dots = recievers.loc[active_dot_ids]
    return active_dots[['label', 'x', 'y', 'z']]


def load_rte_matrix(filename):
    rte_data = scipy.io.loadmat(filename)
    return pd.DataFrame(rte_data['rTE_matrix'])

def load_measurement_data(path):
    measurement_data = scipy.io.loadmat(path)
    data = []
    for idx in range(100):
        data.append(pd.DataFrame(measurement_data['UDP_data'][idx]).applymap(
                                        lambda x: x[0][0] if len(x[0]) == 1 else x[0])[[
                                            'dpePortSel', 'dpeBrSel', 'RxyT', 'Kmaxm1'
                                        ]].drop_duplicates(subset=['dpePortSel', 'dpeBrSel'], keep='first'))
    return data

def load_RxyT(path):
    measurement_data = scipy.io.loadmat(path)
    data = []
    for idx in range(100):
        data.append(pd.DataFrame(measurement_data['UDP_data'][idx]).applymap(
                                        lambda x: x[0][0])[['RxyT']])
    return data

def peak_detection_fft(measurement_data, idx, upsample_ratio, **kwargs):
    peaks = []
    for dot_num in range(7):
        peak_of_dots = []
        value_of_peaks = []
        for _, measurement in measurement_data[idx].query("dpePortSel == @dot_num").iterrows():
            record = np.abs(upsample_signal(np.concatenate((np.zeros(measurement['Kmaxm1']), (measurement['RxyT']))), upsample_ratio=upsample_ratio))
            first_peak = find_first_peak_above_threshold(record, 0.4)
            # first_peak = np.argmax(record)
            peak_of_dots.append(first_peak)
            value_of_peaks.append(record[first_peak])

        peak = np.average(peak_of_dots, weights=value_of_peaks)
        peaks.append(peak)
    peaks = [index2toa(p, M=upsample_ratio) for p in peaks]
    return peaks

def plot_results(antenna_coords, signal_transmitter_coord):
    # Plot antennas in blue
    x_vals = [coord[0] for coord in antenna_coords]
    y_vals = [coord[1] for coord in antenna_coords]
    
    plt.scatter(x_vals, y_vals, color='blue')
    plt.xlim = [-20, 100]
    plt.ylim = [-20, 10]
    
    # Plot signal transmitter in red
    plt.scatter(signal_transmitter_coord[0], signal_transmitter_coord[1], color='red', s=40, alpha=0.1)

def index2toa(index, sample_frequency=122.8e6, M=8):
    sample_period = 1e9 / sample_frequency  # ns
    return (index/M) * sample_period

def remove_outliers(points, threshold=5):
    x = points[:, 0]
    y = points[:, 1]
    
    median_x = np.median(x)
    median_y = np.median(y)
    
    mad_x = np.median(np.abs(x - median_x))
    mad_y = np.median(np.abs(y - median_y))
    
    z_score_x = 0.6745 * (x - median_x) / mad_x
    z_score_y = 0.6745 * (y - median_y) / mad_y
    
    return points[(np.abs(z_score_x) < threshold) & (np.abs(z_score_y) < threshold)]

class Data:
    def __init__(self, id, configfile_path, **kwargs) -> None:
        self.measurement_id = id

        self.configfile_path = pathlib.Path(configfile_path)
        self.configuration = pd.read_csv(self.configfile_path, index_col='id').loc[self.measurement_id]

        folder_path = self.configuration['data_folder'] + "/"

        self.coordinate_data_path = pathlib.Path(self.configfile_path.parent, self.configuration['radio_dot_file'])
        self.measurement_path=pathlib.Path(folder_path + self.configuration['measurement_file']) 
        self.rte_path=pathlib.Path(folder_path + self.configuration['rTE_file'])

        self.rte_matrix=load_rte_matrix(pathlib.Path(self.configuration['data_folder'], self.configuration['rTE_file']))
        self.target_location=self.configuration[['x', 'y', 'z']]
        self.measurement = load_measurement_data(self.measurement_path)
        self.active_dots = [int(x) for x in self.measurement_path.stem.split('_')[-7:]]
        self.active_dot_coordinates = load_dot_coordinates(self.coordinate_data_path, self.measurement_path)
        self.RxyT = load_RxyT(self.measurement_path)
    
    def tdoa_localization(self, upsample_ratio, **kwargs):
      
        coords = []
        for k, coord in self.active_dot_coordinates.iterrows():
            coords.append((coord['x'], coord['y'], coord['z']))

        toa=[]

        for i in range(100):
            toa.append([x*1E-9 for x in peak_detection_fft(self.measurement, idx = i, upsample_ratio = upsample_ratio)])

        results = []
        for i in range(100):
            tdoa_results=tdoa_new(toa[i], coords)
            results.append(tdoa_results) 

        result_len_before_drop = len(results)
        results = remove_outliers(np.array(results))
        print(f"Dropped {result_len_before_drop - len(results)} points in measurement {self.measurement_id}")

        for res in results:
            plot_results(coords, res)
        
        return results
    
