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
        if sample[i + 1][2] != 100:
            break
        for j in range(2):
            if sample[i][2] == 100:
                sample = np.delete(sample, i, axis=0)
                break
    sample = pd.DataFrame(sample, columns=['Sieve Size (mm)', 'Weight Retained (gm)', 'Cumulative percent passing'])
    return sample

def transform_by1 (sample, grade):

    grade_used = grade[0: sample.shape[0]-1]
    grade_used.append(0)
    origin_X = (grade_used - sample.iloc[0:, 0]).tolist()
    return origin_X

def translate_coordinates(sample, grading):
    print '************ SAMPLE CHANGE*************'
    trans_x_coor   = []
    grade = [100, 77.5, 47.5, 36, 25, 17, 8, 0]                 # Exception can occur when compared with transform by as lengths are not equal
    print sample
    transform_by = transform_by1(sample, grade)
    for i in range(sample.shape[0]):        # Translation Of Origin For Different Regions
        for j in range(len(grade)):
            if grade[j] == sample.iloc[i, 2]:
                if j > len(transform_by):
                    trans_x_coor.append(sample.iloc[i, 0])
                    print
                else:
                    trans_x_coor.append(sample.iloc[i, 0])
                    break
            if grade[j] > sample.iloc[i, 2] > grade[j + 1]:
                if j+1 > len(transform_by):
                    nested_origin = np.interp(sample.iloc[i,2], [0, grade[i]], [0, transform_by[len(transform_by)-2]])      # May or MayNot Be Generalised
                    trans_x_coor.append(sample.iloc[i, 2] - nested_origin)
                    trans_x_coor[i] += sample.iloc[i,0] - trans_x_coor[i]
                else:
                    nested_origin = np.interp(sample.iloc[i,2], [grade[j+1], grade[j]], [transform_by[j+1], transform_by[j]])
                    trans_x_coor.append(sample.iloc[i, 2] - nested_origin)
                    trans_x_coor[i] += sample.iloc[i,0] - trans_x_coor[i]           # YoYo Nigga
                    break
    #trans_x_coor = (np.array(trans_x_coor) * 45) / 100      # Scaling X_Coordinate back
    df = pd.DataFrame(trans_x_coor, columns=['Trans_X (Sieve)'])
    df = pd.concat([df, sample.iloc[0:,1:3]], axis=1)
    print df
    return df

class AggregateMixProportion:

    def __init__(self):
        data = pd.read_excel(r'/Users/nikhil/Desktop/Project/Blending Of Aggregate/Utilities/Research_Data.xls')
        self.sample_A = data.iloc[0:, [0, 1]]   # Sample A
        self.sample_B = data.iloc[0:, [0, 2]]   # Sample B
        self.sample_C = data.iloc[0:, [0, 3]]   # Sample C

        percent_passing(self.sample_A)  # Cumulative Percent A
        percent_passing(self.sample_B)  # Cumulative Percent B
        percent_passing(self.sample_C)  # Cumulative Percent C

        self.sample_A.iloc[0:, 2] = [100, 63, 19, 8, 5, 3, 0, 0]
        self.sample_B.iloc[0:, 2] = [100, 100, 100, 93, 55, 36, 3, 0]
        self.sample_C.iloc[0:, 2] = [100, 100, 100, 100, 100, 97, 88, 0]

        self.extr_sample_A = extract_sample(self.sample_A)  # ReFormat DataFrame Such That Only 100% Passing Sieve Is Obtained (Sample A)
        self.extr_sample_B = extract_sample(self.sample_B)  # ReFormat DataFrame Such That Only 100% Passing Sieve Is Obtained (Sample B)
        self.extr_sample_C = extract_sample(self.sample_C)  # ReFormat DataFrame Such That Only 100% Passing Sieve Is Obtained (Sample C)

        self.trans_sample_A = translate_coordinates(self.extr_sample_A, 1)  # Translated Such That Different Regions Have Different Scales
        self.trans_sample_B = translate_coordinates(self.extr_sample_B, 1)  # Translated Such That Different Regions Have Different Scales
        self.trans_sample_C = translate_coordinates(self.extr_sample_C, 1)  # Translated Such That Different Regions Have Different Scales

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
        plt.scatter(self.trans_sample_A.iloc[0:, 0], self.trans_sample_A.iloc[0:, 2])
        plt.scatter(self.trans_sample_B.iloc[0:, 0], self.trans_sample_B.iloc[0:, 2])
        plt.scatter(self.trans_sample_C.iloc[0:, 0], self.trans_sample_C.iloc[0:, 2])
        plt.plot(self.trans_sample_A.iloc[0:, 0], self.trans_sample_A.iloc[0:, 2], linewidth=1)
        plt.plot(self.trans_sample_B.iloc[0:, 0], self.trans_sample_B.iloc[0:, 2], linewidth=1)
        plt.plot(self.trans_sample_C.iloc[0:, 0], self.trans_sample_C.iloc[0:, 2], linewidth=1)
        #plt.plot(self.trans_sample_A.iloc[0:, 0], self.straight_line_A, '--')
        #plt.plot(self.trans_sample_B.iloc[0:, 0], self.straight_line_B, '--')
        #plt.plot(self.trans_sample_C.iloc[0:, 0], self.straight_line_C, '--')
        plt.xlabel('Sieve Size (mm)', color='black')
        plt.ylabel('Cumulative Percent Passing (%)', color='black')
        plt.legend(('Sample A', 'Sample B', 'Sample C'))
        plt.show()

obj = AggregateMixProportion()
obj.plot_curve()