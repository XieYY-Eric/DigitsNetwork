import os
import numpy as np 

class DataCollector:
    def __init__(self):
        """
        This will create a Data object  
        Keyword arguments:
        argument -- description
        Return: Data()
        """
        self.trainDir = r"datas\train-images.idx3-ubyte"
        self.trainlbDIr = r"datas\train-labels.idx1-ubyte"
        self.testDir = r"datas\test-images.idx3-ubyte"
        self.testlbDir = r"datas\test-labels.idx1-ubyte"
        
    def __readIdx1(self,filename:str):
        """read a file, extract the basic binary data and store in a nx1 array
        designed for label data
        Args:
            filename (str): [filename]
        Returns:
            [Numpy(m,1)]: [A numpy array that contains label datas]
        """
        try:
            #read the data in binary format
            file = open(filename,"rb")
            binData = file.read()
            #first 4 bytes are magic number
            #second 4 bytes are size
            #the rest bytes are single label data
            magicNumber = int(binData[:4].hex(),16)
            size = int(binData[4:8].hex(),16)
            data = np.zeros((size,1))
            #store the data into nx1 numpy array
            for i in range(size):
                data[i][0] = binData[i+8]
        except Exception as e:
            print("[001]: __readIdx1 Exception occurs")
        finally:
            file.close()
            return data
 
    def __readIdx3(self,filename:str):
        """read a file, extract the basic binary data and store in a nx1 array
        designed for label data
        Args:
            filename (str): [filename]
        Returns:
            [Numpy(m,row,column)]: [A numpy array that contains sample datas]
        """
        try:
            #read the data in binary format
            file = open(filename,"rb")
            binData = file.read()
            #first 4 bytes are magic number
            #second 4 bytes are size
            #third 4 --> rows
            #fourth 4 -->colums
            #the rest bytes are single label data
            magicNumber = int(binData[:4].hex(),16)
            size = int(binData[4:8].hex(),16)
            row = int(binData[8:12].hex(),16)
            column = int(binData[12:16].hex(),16)
            data = np.empty((size*row*column,1))
            #store the data into 1xn numpy array
            # for i in range(size):
            #     data[i][0] = binData[i+16]
            for i in range(row*column*size):
                data[i,0] = binData[i+16]
            data = data.reshape((size,row,column))       
        except Exception as e:
            print("[002]: __readIdx3 Exception occurs")
        finally:
            file.close()
            return data 
        
    def getData(self):
        """
        trainData/testData are three dimensional data, 
        Returns:
            [tuple]: (trainData,trainlb),(testData,testlb)
        """
        trainData = self.__readIdx3(self.trainDir)
        print("Data read completed:",trainData.shape)
        trainlb = self.__readIdx1(self.trainlbDIr)
        print("Data read completed:",trainlb.shape)

        testData = self.__readIdx3(self.testDir)
        print("Data read completed:",testData.shape)
        testlb = self.__readIdx1(self.testlbDir)
        print("Data read completed:",testlb.shape)
        
        return (trainData,trainlb),(testData,testlb)
