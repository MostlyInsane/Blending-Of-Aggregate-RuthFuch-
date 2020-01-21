import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.style.use('ggplot')
import numpy as np

def percent_passing(sample):  # Calculates Cumulative Percent Passing

    percent_retained = []
    percent_passing = []
    for i in range(sample.shape[0]):
        percent_retained.append(sample.iloc[i, 1] * 100 / sample.sum()[1])
        percent_passing.append(100 - sum(percent_retained))
    sample['Cumulative percent passing'] = percent_passing


def best_fit(sample):  # Finds The BestFit 'LINEAR STRAIGHT LINE' Parameters For The Given Curve

    modif_theta = []
    data_fr = reshape_matrix(sample)
    x_value = np.array(data_fr.iloc[0:, 0:2])
    y_value = np.transpose(np.array([data_fr.iloc[0:, 3]]))
    theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(x_value), x_value)), np.transpose(x_value)),
                      y_value)
    for sublist in theta:
        for item in sublist:
            modif_theta.append(item)
    return modif_theta


def reshape_matrix(sample):     # ReFormats DataFrame To Add A Bias Term (Just for Computational Purposes)
    bias = []
    for i in range(sample.shape[0]):
        bias.append(1)
    df = pd.DataFrame({'bias': bias})
    df = pd.concat([df, sample], axis=1)
    return df

def extract_sample(sample):     # ReFormats DataFrame Such That The Least Possible 100% Passing Sieve Is Obtained
    i = 0
    sample = np.array(sample.iloc[0:, 0:3])
    while i < sample.shape[0]:
        if sample[i + 1][1] != 0:
            break
        for j in range(2):
            if sample[i][1] == 0:
                sample = np.delete(sample, i, axis=0)
                break
    sample = pd.DataFrame(sample, columns=['Sieve Size (mm)', 'Weight Retained (gm)', 'Cumulative percent passing'])
    return sample

def translate_coordinates(sample, grading):

    grade_1  = (100, 97.5, 78, 65, 46, 35, 14, 5, 0)                    # Contains AVG Cumulative Weight for Grading_1
    transform_by = (55, 60, 51.5, 51.8, 41.25, 32.64, 13.7, 4.925,0)    # Origin Shifted Resulting in Different Pseudo_Scales In X-Axis

    grad_2     = (100, 95, 83, 68, 46, 35, 14, 5)                       # Contains AVG Cumulative Weight for Grading_2

    trans_x_coor = []

    for i in range(sample.shape[0]):
        for j in range(len(grade_1)):
            if grade_1[j] == sample.iloc[i, 2]:
                trans_x_coor.append(sample.iloc[i, 0] + transform_by[j])
                break
            if grade_1[j] > sample.iloc[i, 2] > grade_1[j + 1]:
                trans_x_coor.append(sample.iloc[i, 0] + transform_by[j+1])
                break
    df = pd.DataFrame(trans_x_coor, columns=['Trans_X'])
    df = pd.concat([df, sample.iloc[0:,1:3]], axis=1)
    return df

class AggregateMixProportion:

    def __init__(self):
        data = pd.read_excel(r'/Users/nikhil/Desktop/Project/Blending Of Aggregate/Utilities/Input.xls')
        self.sample_A = data.iloc[0:, [0, 1]]   # Sample A
        self.sample_B = data.iloc[0:, [0, 2]]   # Sample B
        self.sample_C = data.iloc[0:, [0, 3]]   # Sample C

        percent_passing(self.sample_A)  # Cumulative Percent A
        percent_passing(self.sample_B)  # Cumulative Percent B
        percent_passing(self.sample_C)  # Cumulative Percent C

        self.extr_sample_A = extract_sample(self.sample_A)  # ReFormat DataFrame Such That Only 100% Passing Sieve Is Obtained (Sample A)
        self.extr_sample_B = extract_sample(self.sample_B)  # ReFormat DataFrame Such That Only 100% Passing Sieve Is Obtained (Sample B)
        self.extr_sample_C = extract_sample(self.sample_C)  # ReFormat DataFrame Such That Only 100% Passing Sieve Is Obtained (Sample C)

        self.trans_sample_A = translate_coordinates(self.extr_sample_A, 1)  # Translated Such That Different Regions Have Different Scales
        self.trans_sample_B = translate_coordinates(self.extr_sample_B, 1)  # Translated Such That Different Regions Have Different Scales
        self.trans_sample_C =translate_coordinates(self.extr_sample_C, 1)  # Translated Such That Different Regions Have Different Scales

        self.theta_A = best_fit(self.trans_sample_A)    # Obtained BestFit (Sample A)
        self.theta_B = best_fit(self.trans_sample_B)    # Obtained BestFit (Sample B)
        self.theta_C = best_fit(self.trans_sample_C)    # Obtained BestFit (Sample C)

        self.resh_sample_A = reshape_matrix(self.trans_sample_A)    # Included Bias Term Computational Purpose (Sample A)
        self.resh_sample_B = reshape_matrix(self.trans_sample_B)    # Included Bias Term Computational Purpose (Sample B)
        self.resh_sample_C = reshape_matrix(self.trans_sample_C)    # Included Bias Term Computational Purpose (Sample C)

        self.straight_line_A = np.matmul(np.array(self.resh_sample_A.iloc[0:, 0:2]), np.array(self.theta_A))    # Predicted Value (Sample A)
        self.straight_line_B = np.matmul(np.array(self.resh_sample_B.iloc[0:, 0:2]), np.array(self.theta_B))    # Predicted Value (Sample B)
        self.straight_line_C = np.matmul(np.array(self.resh_sample_C.iloc[0:, 0:2]), np.array(self.theta_C))    # Predicted Value (Sample C)

    def plot_curve(self):
        plt.plot(self.trans_sample_A.iloc[0:, 0], self.trans_sample_A.iloc[0:, 2], linewidth=1)
        plt.plot(self.trans_sample_B.iloc[0:, 0], self.trans_sample_B.iloc[0:, 2], linewidth=1)
        plt.plot(self.trans_sample_C.iloc[0:, 0], self.trans_sample_C.iloc[0:, 2], linewidth=1)
        plt.plot(self.trans_sample_A.iloc[0:, 0], self.straight_line_A, '--')
        plt.plot(self.trans_sample_B.iloc[0:, 0], self.straight_line_B, '--')
        plt.plot(self.trans_sample_C.iloc[0:, 0], self.straight_line_C, '--')
        plt.show()

obj = AggregateMixProportion()
obj.plot_curve()
