import numpy as np
import csv
import random
import math
import operator



def loadDataset(filename):
    #dataset = pd.read_csv(filename)
    l=1
    train_images = []
    train_labels = []
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

def dotProduct(vector1,vector2,length):
	value = 0
	for x in range(length):
		value= value+int(vector1[x])*int(vector2[x])
	return value

def dotProductMatrix(training_set):
    matrix = []
    for index in range(len(training_set)):
        row = []
        for inner_index in range(len(training_set)):
            if index>inner_index:
                row.append(matrix[inner_index][index])
            else:
                row.append(dotProduct(training_set[index],training_set[inner_index],len(training_set[index])))
        matrix.append(row)
    return matrix

def dottest(training_set,test_data):
    matrix = []
    for index in range(len(training_set)):
        row = []
        for inner_index in range(len(test_data)):
            row.append(dotProduct(training_set[index],test_data[inner_index],len(training_set[index])))
        matrix.append(row)
    return matrix

def one_training_pass(training_set,labels,alphas,dotproducts):
    misclassifications = 0;
    for index in range(len(training_set)):
        sum = 0
        for inner_index in range(len(training_set)):
            value = 1+dotproducts[index][inner_index]
            sum = sum + labels[inner_index]*alphas[inner_index]*math.pow(value,2)
        sum = labels[index]*sum
        if sum<=0:
            alphas[index]+=1
            misclassifications+=1
        #print(alphas)
    return alphas,misclassifications

def testExample(training_set,labels,test_set,test_label,alphas,dotproducts,index):
    sum = 0
    for inner_index in range(len(training_set)):
        value = 1+dotproducts[inner_index][index]
        sum = sum + labels[inner_index]*alphas[inner_index]*math.pow(value,2)
        sum = test_label*sum
        if sum<=0:
            return -1
        else:
            return 1


def train_perceptron_for_n(training_set,training_labels,dotproducts,n):

    new_labels = training_labels
    #initial w vector
    alpha = []
    for i in range(len(training_set)):
        alpha.append(0)
    #append 1 for w0
    for i in range(len(new_labels)):
        if new_labels[i]==n:
            new_labels[i] = 1
        else:
            new_labels[i] = -1
    error = len(training_set)
    while (float(error)/len(training_set))>0.001:
        #print("error "+str(float(error)/len(training_set)*100))
        alpha,error = one_training_pass(training_set,new_labels,alpha,dotproducts)
    print("error rate: "+str(float(error)/len(training_set)*100))
    print("misclassifications: "+str(error))
    print("alpha ........................................................................"+str(n))
    print(alpha)
    return alpha,new_labels

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
    train_images, train_labels = loadDataset("mnist_train1.csv")
    test_images, test_labels = loadDataset("mnist_test.csv")
    training_set = train_images[0:50000]
    training_labels = train_labels[0:50000]
    test_data = test_images[0:10000]
    test_label = test_labels[0:10000]
    print(len(training_set))
    dotproducts = dotProductMatrix(training_set)
    print('dot products..............................................................');
    print(dotproducts)
    #print(math.pow(5,2))
    alphas_for_0,labels_0 = train_perceptron_for_n(training_set, training_labels, dotproducts,0)
    alphas_for_1,labels_1  = train_perceptron_for_n(training_set, training_labels, dotproducts, 1)
    alphas_for_2,labels_2  = train_perceptron_for_n(training_set, training_labels, dotproducts, 2)
    alphas_for_3,labels_3 = train_perceptron_for_n(training_set, training_labels, dotproducts, 3)
    alphas_for_4,labels_4 = train_perceptron_for_n(training_set, training_labels, dotproducts, 4)
    alphas_for_5,labels_5  = train_perceptron_for_n(training_set, training_labels, dotproducts, 5)
    alphas_for_6,labels_6 = train_perceptron_for_n(training_set, training_labels, dotproducts, 6)
    alphas_for_7,labels_7  = train_perceptron_for_n(training_set, training_labels, dotproducts, 7)
    alphas_for_8,labels_8  = train_perceptron_for_n(training_set, training_labels, dotproducts, 8)
    alphas_for_9,labels_9 = train_perceptron_for_n(training_set, training_labels, dotproducts, 9)

    obtained_labels = []

    dotproducts_test = dottest(training_set,test_data)
    print(dotproducts_test)
    for index in range(len(test_data)):
        identified = 0
        label = -1
        if test_label[index]==0:
            label = 1
        result = testExample(training_set,labels_0, test_data, label, alphas_for_0,dotproducts_test,index)
        if result==1:
            identified = 1
            obtained_labels.append(0)
        label = -1
        if test_label[index]==1:
            label = 1
        result = testExample(training_set,labels_1, test_data, label, alphas_for_1,dotproducts_test,index)
        if result==1 and identified==0:
            identified = 1
            obtained_labels.append(1)
        label = -1
        if test_label == 2:
            label = 1
        result = testExample(training_set, labels_2, test_data, label, alphas_for_2,dotproducts_test,index)
        if result == 1 and identified == 0:
            identified = 1
            obtained_labels.append(2)
        label = -1
        if test_label[index] == 3:
            label = 1
        result = testExample(training_set, labels_3, test_data, label, alphas_for_3,dotproducts_test,index)
        if result == 1 and identified == 0:
            identified = 1
            obtained_labels.append(3)
        label = -1
        if test_label[index] == 4:
            label = 1
        result = testExample(training_set, labels_4, test_data, label, alphas_for_4,dotproducts_test,index)
        if result == 1 and identified == 0:
            identified = 1
            obtained_labels.append(4)
        label = -1
        if test_label[index] == 5:
            label = 1
        result = testExample(training_set, labels_5, test_data, label, alphas_for_5,dotproducts_test,index)
        if result == 1 and identified == 0:
            identified = 1
            obtained_labels.append(5)
        label = -1
        if test_label[index] == 6:
            label = 1
        result = testExample(training_set, labels_6, test_data, label, alphas_for_6,dotproducts_test,index)
        if result == 1 and identified == 0:
            identified = 1
            obtained_labels.append(6)
        if test_label[index] == 7:
            label = 1
        result = testExample(training_set, labels_7, test_data, label, alphas_for_7,dotproducts_test,index)
        if result == 1 and identified == 0:
            identified = 1
            obtained_labels.append(7)
        if test_label[index] == 8:
            label = 1
        result = testExample(training_set, labels_8, test_data, label, alphas_for_8,dotproducts_test,index)
        if result == 1 and identified == 0:
            identified = 1
            obtained_labels.append(8)
        if test_label[index] == 9:
            label = 1
        result = testExample(training_set, labels_9, test_data, label, alphas_for_9,dotproducts_test,index)
        if result == 1 and identified == 0:
            identified = 1
            obtained_labels.append(9)
        if identified==0:
            obtained_labels.append(-1)
    print("accuracy on testing" +str(getAccuracy(test_labels,obtained_labels)))




main()