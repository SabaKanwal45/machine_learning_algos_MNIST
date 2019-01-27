import numpy as np
import csv

def loadDataset(filename):
    #dataset = pd.read_csv(filename)
    l=1
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    with open('C:\\Users\\saba\\Documents\\ML\\'+filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            if(l!=1):
                train_labels.append(int(dataset[x][0]))
                results = list(map(int, dataset[x][1:len(dataset)-1]))
                train_images.append(results)
            else:
                print(dataset[x][0])
                print('inside else')
                l=l+1
    #print(train_labels)
    return train_images,train_labels

class Neural_Network(object):
    def __init__(self):
        #parameters
        self.inputSize = 785
        self.outputSize = 10
        self.hiddenSize = 40

        #weights
        self.W1 = np.random.randn(
            self.inputSize,
            self.hiddenSize)  # (3x2) weight matrix from input to hidden layer
        self.W1 = self.W1/100
        self.W2 = np.random.randn(
            self.hiddenSize,
            self.outputSize)  # (3x1) weight matrix from hidden to output layer
        self.W2 = self.W2/100

    def forward(self, X):
        #forward propagation through our network
        self.z = np.dot(
            X,
            self.W1)  # dot product of X (input) and first set of 3x2 weights
        self.z2 = self.sigmoid(self.z)  # activation function
        self.z3 = np.dot(
            self.z2, self.W2
        )  # dot product of hidden layer (z2) and second set of 3x1 weights
        o = self.sigmoid(self.z3)  # final activation function
        return o

    def sigmoid(self, s):
        # activation function
        a = np.exp(-s)
        
        return 0.01*(1 / (1 + a))

    def sigmoidPrime(self, s):
        #derivative of sigmoid
        return s * (1 - s)

    def backward(self, X, y, o):
        # backward propgate through the network
        self.o_error = y - o  # error in output
        self.o_delta = self.o_error * self.sigmoidPrime(
            o)  # applying derivative of sigmoid to error

        self.z2_error = self.o_delta.dot(
            self.W2.T
        )  # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error * self.sigmoidPrime(
            self.z2)  # applying derivative of sigmoid to z2 error

        self.W1 += X.T.dot(
            self.z2_delta)  # adjusting first set (input --> hidden) weights
        self.W2 += self.z2.T.dot(
            self.o_delta)  # adjusting second set (hidden --> output) weights

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

    def saveWeights(self):
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("w2.txt", self.W2, fmt="%s")

    def predict(self,test_images,test_labels):
        output = self.forward(test_images)
        print(len(output))
        print(len(output[0]))
        predictions = []
        for i in range(len(output)):
            idx = np.argmax(output[i])
            predictions.append(idx)
        getAccuracy(test_labels,predictions)


def getAccuracy(test_labels, predictions):
    correct = 0
    for x in range(len(predictions)):
        if test_labels[x] == predictions[x]:
            correct += 1
    print("total examples")
    print(len(predictions))
    print(correct)
    return (correct / float(len(predictions))) * 100.0

def main():
    NN = Neural_Network()
    train_images,train_labels = loadDataset("mnist_train1.csv")
    test_images,test_labels = loadDataset("mnist_test.csv")
    training_set = train_images[0:60000]
    training_labels = train_labels[0:60000]
    validation_set = train_images[200:205]
    validation_labels = train_labels[200:205]
    # Scale Images
    X_data = np.array(training_set)/255
    # Scale labels
    new_labels = []
    x = np.array(X_data)
    y = []
    for i in range(len(training_set)):
        y.append([1])
    new_training_set = np.append(y, x, axis=1)
    for i in range(len(training_labels)):
        new_labels.append([0,0,0,0,0,0,0,0,0,0])
        new_labels[i][training_labels[i]] = 1
    print(len(new_labels))
    print(len(new_labels[0]))
    for i in range(50):
        temp_result = NN.forward(new_training_set)
        error = 0
        for j in range(len(training_labels)):
            if (np.argmax(temp_result[j])!= training_labels[j]):
                #print('inside error')
                #print(np.argmax(temp_result[j]))
                #print(training_labels[j])
                error +=1
        print ("Missclassificats: \n" +str(error) )
        NN.train(new_training_set,new_labels[i])
    NN.saveWeights()
    # test images
    X_test = np.array(validation_set)/255
    new_test_data = np.append(y, X_test, axis=1)
    NN.predict(new_test_data,validation_labels)

main()