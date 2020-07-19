import csv
import numpy as np
from preprocessing import preprocess

def import_data():
    X = np.genfromtxt("train_X_pr.csv",delimiter=',',dtype = np.float128,skip_header = 1)
    Y = np.genfromtxt("train_Y_pr.csv",delimiter=',',dtype = np.float128)
    return X, Y

def sigmoid(Z):
    sigma = 1/(1+(np.e)**(-1 * Z))
    return sigma

def compute_cost(X, Y, W, b, Lambda):
    Hx = sigmoid(np.dot(X,W) + b)
    loss = Y * np.log(Hx) + (1 - Y) * np.log(1-Hx)
    cost = ((-1/len(X)) * np.sum(loss)) + (Lambda * np.sum(np.square(W)))/(2*len(X))
    return cost

def compute_gradient_of_weights_using_regularization(X, Y, W, b,Lambda):
    Z = np.dot(X,W) + b
    A = sigmoid(Z)
    dZ = A - Y
    dW = (1/len(X))*(np.dot(X.T,dZ) + Lambda * W)
    dB = (1/len(X))*np.sum(dZ)
    return dW,dB

def get_optimized_weights_using_gradient_descent(X,Y,alpha,max_iter,Lambda):
    W = np.zeros((len(X[1]),1))
    B = 0
    prev_ite_cost = 0
    iter_cnt = 0
    while True:
        iter_cnt += 1
        dW,dB = compute_gradient_of_weights_using_regularization(X,Y,W,B,Lambda)
        W -= alpha * dW
        B -= alpha * dB
        cost = compute_cost(X,Y,W,B,Lambda)

        if iter_cnt % 2000 == 0:
            print("{:<10} {:<10} {} ".format(iter_cnt,round(cost,10),abs(prev_ite_cost - cost)))

        if abs(prev_ite_cost - cost) < 0.0000001 or iter_cnt == max_iter:
            print(iter_cnt,cost)
            break
        
        prev_ite_cost = cost

    #print('W',W.shape)
    W = (W.T)[0]
    #print(W,B)
    #print('W',W.shape)
    WB = np.append(W,B)
    #print(WB)
    #print('WB',WB.shape)
    return [WB]


def train_model(X,Y):
    alpha = 0.01
    max_iter = 100000
    Lambda = 0

    weights = get_optimized_weights_using_gradient_descent(X,Y,alpha,max_iter,Lambda)
    return weights


def save_model(weights, weights_file_name):
    with open(weights_file_name, 'w') as weights_file:
        wr = csv.writer(weights_file)
        wr.writerows(weights)
        weights_file.close()

if __name__ == "__main__":
    X,Y_1d = import_data()
    X = preprocess(X)
    Y = np.zeros((1,len(Y_1d)))
    Y[0] = Y_1d
    Y = Y.T
    weights = train_model(X,Y)
    save_model(weights,"WEIGHTS_FILE.csv")
