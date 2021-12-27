import numpy as np 
import matplotlib.pyplot as plt

def normalize(arr):
    """Normalize numpy array data by using standard normalization procedure
       NOT IN PLACE modify
    Args:
        arr ([numpy array]): [must be valid]

    Returns:
        [numpy array]: [normlization version of array]
        [numpy array]: [minValue]
        [int]:         [difference]
    """
    minValue = arr.min(0)
    maxValue = arr.max(0)
    difference = maxValue - minValue
    result = (arr-minValue)/difference
    result[np.isnan(result) | np.isinf(result)] = 0
    return result,minValue,difference

def getData(filename,size=None):
    fin = open(filename)
    ret = []
    info = fin.readline()
    info = info.strip()
    info = int(info)
    if(size == None):
        size = info
    for index in range(size):
        line = fin.readline().strip()
        line = line.replace(","," ")
        line = line.split()
        insert = []
        for element in line:
            insert.append(float(element))
        ret.append(insert)
    fin.close()
    X = np.array(ret)
    m,n = X.shape
    Y = X[:,n-1].reshape(m,1)
    X = X[:,:n-1]
    return X,Y 

def sigmoid(z):
    ret = 1 / (1+ np.exp(-z))
    return ret

def sigmoidGradient(z):
    g = (1-sigmoid(z)*sigmoid(z))
    return g