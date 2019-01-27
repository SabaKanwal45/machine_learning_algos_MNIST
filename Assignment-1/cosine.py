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


def return_index_of_KNN(dist,k):
    temp_array = np.array(dist)
    if k>len(dist):
        k = len(dist)
    idx = np.argpartition(temp_array, -k)[-k:]
    return idx[0:k]

def get_K_Neighbors(training_set, test_instance, k,distances=[]):
    length = len(test_instance)
    for x in range(len(training_set)):
        distances.append(cos_sim(training_set[x], test_instance,length))
    idx =np.argmax(distances)
    print(idx)
    print(distances[idx])
    return return_index_of_KNN(distances,k)

def getVotes(test_labels,indexes):
    labelVotes = {}
    for x in range(len(indexes)):
        response = test_labels[indexes[x]]
        if response in labelVotes:
            labelVotes[response] += 1
        else:
            labelVotes[response] = 1
    print(labelVotes)
    sortedVotes = sorted(labelVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(test_labels, predictions):
    correct = 0
    for x in range(len(predictions)):
        if test_labels[x] == predictions[x]:
            correct += 1
    print("total examples")
    print(len(predictions))
    print(correct)
    return (correct / float(len(predictions))) * 100.0


def magnitude(vector,length):
	value = 0
	for x in range(length):
		value= value+int(vector[x])*int(vector[x])
	return math.sqrt(value)

def dotProduct(vector1,vector2,length):
	value = 0
	for x in range(length):
		value= value+int(vector1[x])*int(vector2[x])
	return value




def cos_sim(traingingInstance, testInstance,length):
	"""Takes 2 vectors a, b and returns the cosine similarity according 
	to the definition of the dot product
	"""
	#dot_product = np.dot(a, b)
	#norm_a = np.linalg.norm(a)
	#norm_b = np.linalg.norm(b)
	dot_product = dotProduct(traingingInstance,testInstance,length)
	#print(dot_product)
	mag_traingingInstance = magnitude(traingingInstance,length)
	#print(mag_traingingInstance)
	mag_testInstance = magnitude(testInstance,length)
	#print(mag_testInstance)
	return dot_product / (mag_traingingInstance * mag_testInstance)

def main():
	train_images,train_labels = loadDataset("mnist_train1.csv")
	test_images,test_labels = loadDataset("mnist_test.csv")
	training_set = train_images[0:50000]
	training_labels = train_labels[0:50000]
	validation_set = train_images[50100:50200]
	validation_labels = train_labels[50100:50200]
	test_data = test_images[0:99]
	test_label= test_labels[0:99]
	final_k=1
	predictions =[]
	for k in range(20):
		for x in range(len(validation_set)):
			indexes = get_K_Neighbors(training_set, validation_set[x], final_k,[])
			#print(indexes)
			result = getVotes(training_labels,indexes)
			predictions.append(result)
		print('> predicted=' + repr(result) + ', actual=' + repr(validation_labels[x]))
		accuracy = getAccuracy(validation_labels, predictions)
		print("value of k: ")
		print(k)
		print(" value of accuracy ")
		print(accuracy)
	for x in range(test_data):
		indexes = get_K_Neighbors(train_images, test_data[x], final_k,[])
		#print(indexes)
		result = getVotes(train_labels,indexes)
		predictions.append(result)
	print('> predicted=' + repr(result) + ', actual=' + repr(test_label[x]))
	accuracy = getAccuracy(test_label, predictions)


main()