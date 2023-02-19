import numpy as np
import math

# set this to true if you want to see all the data miss-labled by the network
PRINT_MISSED_LABELS = False

# seed for reproducibility
seed = 69
np.random.seed(seed)

class Network:
    def __init__(self):

        # Set size of layers 
        self.input_layer = 2
        self.hidden_layer = 4
        self.output_layer = 1

        # Initialize the weights and biases
        self.w1 = np.random.rand(self.input_layer, self.hidden_layer)
        self.b1 = np.random.rand(1, self.hidden_layer) #np.zeros((1, self.hidden_layer))
        self.w2 = np.random.rand(self.hidden_layer, self.output_layer)
        self.b2 = np.random.rand(1, self.output_layer) #np.zeros((1, self.output_layer))

        #stores the last gradients subtracted from the weights and biases
        #dL i.r.t w1, b1, w2, b2
        self.last_gradients = []
        self.printLables = ["w1", "b1", "w2", "b2"]
    
    #--sigmoid-activation function and derivative
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    


    #--mean squared error loss function
    def mse_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true)**2)


    def train(self, amountEpochs, learningRate, X, y):
        w1, b1, w2, b2 = self.w1, self.b1, self.w2, self.b2

        lossHistory = []
        for epoch in range(amountEpochs):
            # Forward pass
            a1 = np.dot(X, w1) + b1
            h = self.sigmoid(a1)

            a2 = np.dot(h, w2) + b2
            y_pred = self.sigmoid(a2)

            # calculate loss and add it to history
            epoch_loss = self.mse_loss(y_pred, y)
            lossHistory.append(epoch_loss)

            # -----------------------------calculation of the gradinet
            dL_dy_pred = 2* (y_pred - y)
            dy_pred_da2 = self.sigmoid_derivative(a2)
            da2_dh = w2.T
            dh_da1 = self.sigmoid_derivative(a1)
            da1_dw1 = X.T
            da1_db1 = da_2_db2 = np.ones((1, len(X)))
            da2_dw2 = h.T

            #more advanced gradients
            dL_da2 = dL_dy_pred * dy_pred_da2
            dL_dh = np.dot(dL_da2, da2_dh)

            #actual gradients of weights and biases 
            dL_da1 = dL_dh*dh_da1
            
            dL_dw1 = np.dot(da1_dw1, dL_da1)
            dL_db1 = np.dot(da1_db1, dL_da1)

            dL_dw2 = np.dot(da2_dw2, dL_da2)
            dL_db2 = np.dot(da_2_db2, dL_da2)

            #Update weights and biases
            w1 -= learningRate * dL_dw1
            b1 -= learningRate * dL_db1
            w2 -= learningRate * dL_dw2
            b2 -= learningRate * dL_db2
        
        #did this for cleaner code inside the for-loop
        self.w1, self.b1, self.w2, self.b2 = w1, b1, w2, b2
        self.last_gradients = [dL_dw1, dL_db1, dL_dw2, dL_db2]

    
    def printWeightsAndBiases(self):
        print("Printing weights and biases")
        self.printWithLabel([self.w1, self.b1, self.w2, self.b2])

    def printLastGradients(self):
        print("Printing last calculated gradients - only updated after one complete training")
        self.printWithLabel(self.last_gradients)

    def printWithLabel(self, printInfoArray):
        for label, printInfo in zip(self.printLables, printInfoArray):
            print(label)
            print(printInfo)

    def doForwardPass(self, singleDataPoint):
        a1 = np.dot(singleDataPoint, self.w1) + self.b1
        h = self.sigmoid(a1)

        a2 = np.dot(h, self.w2) + self.b2
        y_pred = self.sigmoid(a2)
        return y_pred
        

def generateTrainingData(datapoints):
    points = []
    labels = []
    for x in range(datapoints):
        point = [np.random.random(), np.random.random()]
        points.append(point)
        #generate the label
        labels.append([
            math.sqrt(point[0]**2 + point[1]**2) <= 1
        ])
    return np.array(points), np.array(labels)




def getRelativeAmountCorrectlyLabled(network, dataPoints, labels):
    #check what percentage of training data the network labels correctly
    correct_labled_data = 0
    for i in range(len(dataPoints)):
        network_est = network.doForwardPass(dataPoints[i])
        if (round(network_est[0][0]) == labels[i]):
            correct_labled_data += 1
        elif (PRINT_MISSED_LABELS):
            print(str(dataPoints[i]) + ", network_est :" + str(network_est[0][0])  + ", true label :" + str(labels[i]))

    return correct_labled_data / len(dataPoints) * 100

def testNetwork(amountTrainingSamples, amountTestingSamples):

    #train network and get rate
    trainingData, labels = generateTrainingData(amountTrainingSamples)
    network.train(amountEpochs=10000, learningRate=0.1, X=trainingData, y=labels)
    rel_amount_correc_training = getRelativeAmountCorrectlyLabled(network, trainingData, labels)

    #generate new data and check how many unseen datasamples are labeled correctly
    testingData, labels_testing = generateTrainingData(amountTestingSamples)
    rel_amount_correc_testing = getRelativeAmountCorrectlyLabled(network, testingData, labels_testing)

    infoString = """
    Finished training and testing the network. Seed = {4}
        - {0} % correct estimations on training data ({1} data-samples)
        - {2} % correct estimations on testing data ({3} test-samples)
    """
    print(infoString.format(rel_amount_correc_training,amountTrainingSamples,
                             rel_amount_correc_testing,amountTestingSamples, seed))
    
network = Network()
testNetwork(amountTrainingSamples = 200,
            amountTestingSamples = 10000)


