import matplotlib.pyplot as plt

class History:
    def __init__(self,iteration,cost):
        self.__cost = cost
        self.__iteration = iteration

    def showGraph(self):
        plt.plot(self.__iteration,self.__cost)
        plt.xlabel("iteration")
        plt.ylabel("cost")
        plt.title('Gradient Descent')
        plt.show()
    
    def summary(self):
        for i in range(len(self.__iteration)):
            print("iteration {0} || cost {1}".format(self.__iteration[i],self.__cost[i]))