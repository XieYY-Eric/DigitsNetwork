import numpy as np
import os
import MachineLib
import DataCollector
import History

class network:
    def __init__(self):
        self.unit_InputLayer = 28*28
        self.unit_SecondLayer = 25 #25 turns out to be underfitting, changed to 30
        self.unit_OutputLayer = 10
        self.lambda_t = 0.001  #the lower the lambda, the lower the cost
        self.__minV = None
        self.__Diff = None
        self.__hasParameter = False
        self.__parameter = None
        self.__theta1 = None
        self.__theta2 = None
        self.__learnRate = 1

    def train(self, datas, labels,iteration:int,save = True,reuse = False):
        """[summary]
        datas will be convert to n x (row*column)
        Args:
            datas ([type]): (size,row,column)
            labels ([type]): (size, 1)
        """
        #process data ,reshape into mxn array
        m,row,column = datas.shape
        datas = datas.reshape((m,row*column))
        #start by creating a new set of paramter combined into one
        theta1Size = self.unit_SecondLayer*(self.unit_InputLayer+1)
        theta2Size = self.unit_OutputLayer*(self.unit_SecondLayer+1)
        if not reuse:
            self.__parameter  = np.zeros((theta1Size+theta2Size,1))
            #apply randomlization
            self.__parameter = self.randomInitialization(self.__parameter,0.12)
        else:
            print("reusing parameter")
            self.__parameter = self.getParameter("result.csv")
            self.__parameter = self.__parameter.reshape((self.__parameter.shape[0],1))
        self.__hasParameter = True
        #apply normalization
        datas,self.__minV,self.__Diff = MachineLib.normalize(datas)
        cost = []
        it = []
        for i in range(iteration):
            J,G = self.nnCostFunction(self.__parameter,self.unit_InputLayer,self.unit_SecondLayer,self.unit_OutputLayer,datas,labels,self.lambda_t)
            self.__parameter -= self.__learnRate*G
            cost.append(J)
            it.append(i+1)
            print("Iteration {} completed".format(i+1))
        cost = np.array(cost)
        it = np.array(it)
        history = History.History(it,cost)
        if save:
            np.savetxt("result.csv",self.__parameter,delimiter=",")
            print("parameter saved to file")
        return history

    def assignNormalizeData(self,datas):
        m,r,c = datas.shape
        datas = datas.reshape((m,r*c))
        datas,self.__minV,self.__Diff = MachineLib.normalize(datas)

    def predict(self,testDatas)->int:
        """[summary]
        Predict one set of data
        Args:
            testDatas ([numpy(1,n)]): [description]

        Returns:
            int: [0-9]
        """
        if not self.__hasParameter:
            self.__parameter = self.getParameter("result.csv")
            self.__hasParameter = True
        self.__theta1 = self.__parameter[:self.unit_SecondLayer*(self.unit_InputLayer+1)]
        self.__theta2 = self.__parameter[self.unit_SecondLayer*(self.unit_InputLayer+1):]
        self.__theta1 = self.__theta1.reshape((self.unit_SecondLayer,self.unit_InputLayer+1))
        self.__theta2 = self.__theta2.reshape((self.unit_OutputLayer,self.unit_SecondLayer+1))

        m = 1
        n = testDatas.shape[0]*testDatas.shape[1]
        #transform the data to normalized verison
        testDatas = testDatas.reshape((m,n))
        testDatas = (testDatas - self.__minV) / self.__Diff
        testDatas[np.isnan(testDatas) | np.isinf(testDatas)] = 0

        #forward propragation to see the maximum value
        z1 = np.c_[np.ones((m,1)),testDatas]
        z2 = z1.dot(self.__theta1.T)
        a2 = MachineLib.sigmoid(z2)
        a2 = np.c_[np.ones((m,1)),a2]
        z3 = a2.dot(self.__theta2.T)
        h = MachineLib.sigmoid(z3)
        return int(h.argmax(1))

    def getParameter(self,filename):
        """[read data from a filename and form a numpy array (nx1) which is our trained parameters
            primary use for predicting testData]
            Exception:
                if file doesn't exist, return None
        Args:
            filename ([Str]): [parameter files]

        Returns:
            [numpy(nx1)]: [parameters]
        """
        if os.path.exists(filename):
            return np.genfromtxt(filename,dtype = float,delimiter=",")
        else:
            return None
    
    def randomInitialization(self,theta,epsilon=0.12) -> None:
        """[not in place random Initialization]

        Args:
            theta ([type]): [description]
            epsilon (float, optional): [description]. Defaults to 0.12.
        return:
            numpy array
        """
        theta = np.random.uniform(low=-epsilon, high=epsilon, size=theta.shape)
        return theta

    def nnCostFunction(self,parameters,input_layer_size,hidden_layer_size,num_labels,X,y,lambdaV):
        """[netwerd forward and backward propagation, used to compute cost and gradient]
        Args:
            parameters ([numpy(n,1)]): [description]
            input_layer_size ([int]): [description]
            hidden_layer_size ([int]): [description]
            num_labels ([int]): [description]
            X ([numpy(m,n)]): [description]
            y ([numpy(m,1)]): [description]
            lambdaV ([int]): [description]
        """
        #get parameter for both layers
        theta1,theta2 = parameters[:hidden_layer_size*(input_layer_size+1)],parameters[hidden_layer_size*(input_layer_size+1):]
        theta1 = theta1.reshape((hidden_layer_size,input_layer_size+1))
        theta2 = theta2.reshape((num_labels,hidden_layer_size+1))
        m,n = X.shape
        #return the following result
        J = 0
        theta1grad = np.zeros(theta1.shape)
        theta2grad = np.zeros(theta2.shape)
        #compute cost 
        a1 = np.c_[np.ones((m,1)),X]  # m x n

        z2 = a1.dot(theta1.T)          # 
        a2 = np.c_[np.ones((m,1)),MachineLib.sigmoid(z2)]
        
        z3 = a2.dot(theta2.T)
        h = MachineLib.sigmoid(z3)  #mx10
        yVector = np.zeros((m,num_labels))
        for i in range(m):
            yVector[i,int(y[i,0])] = 1
        #cost function for this neural network
        J = sum(sum((-yVector * np.log(h)) + (-(1-yVector)*(np.log(1-h))),2))/m 

        #compute gradient through back propragation 

        X = np.c_[np.ones((m,1)), X]
        DELTA_1 = np.zeros(theta1.shape)
        DELTA_2 = np.zeros(theta2.shape)  
        #use for loop to process each sample data
        for i in range(m):
            x_t = X[i,:].reshape(1,785)  #each sample input data-->1x785
            y_t = yVector[i, :]
            z2_t = x_t.dot(theta1.T)
            a2_t = MachineLib.sigmoid(z2_t) #1x25
            a2_t = np.c_[np.ones((a2_t.shape[0],1)),a2_t]
            z3_t = a2_t.dot(theta2.T)
            a3_t = MachineLib.sigmoid(z3_t) #1x10

            delta_3 = (a3_t - y_t)
            tmp = theta2.T.dot(delta_3.T) 
            delta_2 = tmp[1:, :].T * MachineLib.sigmoidGradient(z2_t)
            DELTA_1 = DELTA_1 + delta_2.T.dot(x_t)
            DELTA_2 = DELTA_2 + delta_3.T.dot(a2_t)  

        t1 = np.c_[np.zeros((theta1.shape[0],1)),theta1[:,1:] ]
        t2 = np.c_[np.zeros((theta2.shape[0],1)),theta2[:,1:] ]
        theta1grad = (1/m)*DELTA_1 + (lambdaV/m)*t1
        theta2grad = (1/m)*DELTA_2 + (lambdaV/m)*t2
        #Unroll gradients
        theta1grad = theta1grad.reshape((theta1grad.shape[0]*theta1grad.shape[1],1))
        theta2grad = theta2grad.reshape((theta2grad.shape[0]*theta2grad.shape[1],1))
        thetaGrad = np.zeros((theta1grad.shape[0]+theta2grad.shape[0],1))
        thetaGrad[:theta1grad.shape[0],:] = theta1grad
        thetaGrad[theta1grad.shape[0]:,:] = theta2grad
        #ToDo: regulazaition

        # theta1Sum = sum(sum(theta1[:,1:(input_layer_size + 1)]*theta1[:,1:(input_layer_size + 1)]))
        # theta2Sum = sum(sum(theta2[:,1:(hidden_layer_size + 1)]*theta2[:,1:(hidden_layer_size + 1)]))
        # total = lambdaV*(theta1Sum+theta2Sum) / (2*m)
        # J += total

        return J,thetaGrad

    def evaluate(self,X,y):
        m = X.shape[0]
        correct = 0
        result = {i:0 for i in range(10)}
        for i in range(m):
            if i%1000 == 0:
                print("1000 data completed")
            p = self.predict(X[i])
            if p == y[i]:
                correct += 1
            result[p] += 1
        incorrect = m-correct
        print("Number of sample set:{}".format(m))
        print("Accurate rate:{:.2%}".format(correct/m))
        print(result)
