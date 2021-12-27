import DataCollector
import matplotlib.pyplot as plt
import network
import MachineLib
import History
import numpy as np

collector = DataCollector.DataCollector()
print("Collecting data")
(trainData,trainLb),(sampleData,sampleLabel) = collector.getData()
print("Data fetch completed")

m,r,c = trainData.shape
validSize = int(0.2 * m)
validationData,validationLabel = trainData[:validSize],trainLb[:validSize]
trainData,trainLb = trainData[validSize:],trainLb[validSize:]




net = network.network()
m = trainData.shape[0]
batchNum = 128

print("training netword")
history = net.train(trainData,trainLb,10)
print("Completed")
net.assignNormalizeData(trainData)
history.summary()
history.showGraph()
net.evaluate(sampleData,sampleLabel)

for i in range(1,100,7):
    print(sampleLabel[i])
    print(net.predict(sampleData[i]))
    plt.imshow(sampleData[i], cmap='gray')
    plt.show()


