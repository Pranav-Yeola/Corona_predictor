import numpy as np
import csv
import sys

from preprocessing import preprocess
from validate import validate

"""
Predicts the target values for data in the file at 'test_X_file_path', using the weights learned during training.
Writes the predicted values to the file named "predicted_test_Y_pr.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data_and_weights(test_X_file_path, weights_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    weights_b = list(np.genfromtxt(weights_file_path, delimiter=',', dtype=np.float64))
    weights = np.array(weights_b[:len(weights_b)-1])
    b = weights_b[len(weights_b)-1]
    return test_X, weights,b

def sigmoid(Z):
    sigma = 1/(1+(np.e)**(-1 * Z))
    return sigma

def predict_target_values(test_X, weights,b):
    Hx = sigmoid((np.dot(test_X,weights.T) + b))
    pred_Y = []
    for i in range(len(Hx)):
        if Hx[i] >= 0.5:
            pred_Y.append(1)
        else:
            pred_Y.append(0)
    pred_Y = np.array(pred_Y)
    return pred_Y.astype(int)


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()

def predict(test_X_file_path):
    test_X, weights,b = import_data_and_weights(test_X_file_path, "WEIGHTS_FILE.csv")
    test_X = preprocess(test_X)
    pred_Y = predict_target_values(test_X, weights,b)
    write_to_csv_file(pred_Y, "predicted_test_Y_pr.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    # validate(test_X_file_path, actual_test_Y_file_path="train_Y_pr.csv") 
