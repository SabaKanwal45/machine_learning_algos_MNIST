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
                #print(dataset[x][0])
                #print('inside else')
                l=l+1
    #print(train_labels)
    return train_images,train_labels

def compute_probability_with_label(d_dist,prob_dist,label,testing_example):
    prob = d_dist[label]
    for index in range(len(testing_example)):
        if(testing_example[index]>127):
            prob *= prob_dist[label][index]
        else:
            prob *= 1-prob_dist[label][index]
    return prob



def compute_probability_distribution(training_set,training_label,min,max,k):
    digit_distribution = [0 for row in range(10)]
    probability_distribution = [[0 for col in range(len(training_set[0]))] for row in range(10*max)]
    for index in range(len(training_set)):
        digit_distribution[training_label[index]] += 1
        for inner_index in range(len(training_set[index])):
            #value = training_set[index][inner_index]
            #probability_distribution[(training_label+1) * value][inner_index] += 1
            if training_set[index][inner_index] > 127:
                probability_distribution[training_label[index]][inner_index] += 1
    for index in range(len(probability_distribution)):
        for inner_index in range(len(probability_distribution[index])):
            probability_distribution[index][inner_index] = (k+probability_distribution[index][inner_index])/(digit_distribution[index]+2*k)
    for index in range(len(digit_distribution)):
        digit_distribution[index] = (digit_distribution[index])/(len(training_set))
    return digit_distribution,probability_distribution



def main():

    train_images, train_labels = loadDataset("mnist_train1.csv")
    test_images, test_labels = loadDataset("mnist_test.csv")
    training_set = train_images[0:60000]
    training_labels = train_labels[0:60000]
    test_data = test_images[0:10000]
    test_label = test_labels[0:10000]
    d_dist, prob_dist=compute_probability_distribution(training_set, training_labels, 0, 1,1)
    print(len(training_set))
    print(d_dist)
    sum = 0
    for index in range(len(d_dist)):
        sum+=d_dist[index]
    print(sum)
    correct = 0
    for index in range(len(test_data)):
        dist = [0 for row in range(10)]
        dist[0] = compute_probability_with_label(d_dist, prob_dist, 0, test_data[index])
        dist[1] = compute_probability_with_label(d_dist, prob_dist, 1, test_data[index])
        dist[2] = compute_probability_with_label(d_dist, prob_dist, 2, test_data[index])
        dist[3] = compute_probability_with_label(d_dist, prob_dist, 3, test_data[index])
        dist[4] = compute_probability_with_label(d_dist, prob_dist, 4, test_data[index])
        dist[5] = compute_probability_with_label(d_dist, prob_dist, 5, test_data[index])
        dist[6] = compute_probability_with_label(d_dist, prob_dist, 6, test_data[index])
        dist[7] = compute_probability_with_label(d_dist, prob_dist, 7, test_data[index])
        dist[8] = compute_probability_with_label(d_dist, prob_dist, 8, test_data[index])
        dist[9] = compute_probability_with_label(d_dist, prob_dist, 9, test_data[index])
        predicted_label = np.argmax(dist)
        print("example label "+str(test_label[index])+" Predicted label "+str(predicted_label))
        if(test_label[index]==predicted_label):
            correct +=1
    print("Correctly Predicted examples "+str(correct))








main()