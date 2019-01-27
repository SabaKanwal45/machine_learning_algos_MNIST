# load mnist handwirrten digit data set
#from loader import MNIST
#for array operations
import numpy as np
#for plotting images
import matplotlib.pyplot as plt
import operator
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


"""def load_mnist_dataset():
    load_data = MNIST('./dataset')
    train_images, train_labels = load_data.load_training()
    test_images, test_labels = load_data.load_testing()
    return train_images,train_labels,test_images,test_labels"""


def show_mnist_image(image):
    _image = np.array(image, dtype='float')
    pixels = _image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()

def show_training_des(training_set):
    print("Training Examples: "+len(training_set))
    print("Pixels in one image: "+len(training_set[0]))
    print("Maximum Pixel value: "+len(max(training_set[0])))

def return_index_of_KNN(dist,k):
    temp_array = np.array(dist)
    if k>len(dist):
        k = len(dist)
    idx = np.argpartition(temp_array, k)
    return idx[0:k]

def euclidean_Distance(train_instance, test_instance, length):
    distance = 0
    for x in range(length):
        distance += pow((train_instance[x] - test_instance[x]), 2)
    return distance


def getVotes(test_labels,indexes):
    labelVotes = {}
    for x in range(len(indexes)):
        response = test_labels[indexes[x]]
        if response in labelVotes:
            labelVotes[response] += 1
        else:
            labelVotes[response] = 1
    sortedVotes = sorted(labelVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(test_labels, predictions):
    correct = 0
    for x in range(len(predictions)):
        if test_labels[x] == predictions[x]:
            correct += 1
    print("total examples")
    print(correct)
    return (correct / float(len(predictions))) * 100.0

def get_K_Neighbors(training_set, test_instance, k,distances=[]):
    length = len(test_instance)
    for x in range(len(training_set)):
        distances.append(euclidean_Distance(test_instance, training_set[x], length))
    return return_index_of_KNN(distances,k)

def main():
    train_images,train_labels = loadDataset("mnist_train.csv")
    test_images,test_labels = loadDataset("mnist_test.csv")
    #print(train_labels)
    #train_images,train_labels,test_images,test_labels = load_mnist_dataset()
    #split train data into validation and training data to find value of hyper parameter
    # 50,000 training images
    training_set = train_images[0:49999]
    training_labels = train_labels[0:49999]
    #10,000 validation images
    validation_set = train_images[58000:59999]
    validation_labels = train_labels[58000:59999]
    final_k = 1
    k=3
    predictions =[]
    #len(validation_set)
    #print(validation_labels[0])
    print("58000 to 59999")
    for x in range(len(validation_set)):
        indexes = get_K_Neighbors(training_set, validation_set[x], k,[])
        result = getVotes(training_labels,indexes)
        predictions.append(result)
        #print('> predicted=' + repr(result) + ', actual=' + repr(validation_labels[x]))
    accuracy = getAccuracy(validation_labels, predictions)
    print("value of k: ")
    print(k)
    print(" value of accuracy ")
    print(accuracy)
    for x in range(test_data):
        indexes = get_K_Neighbors(train_images, test_images[x], final_k,[])
        #print(indexes)
        result = getVotes(test_labels,indexes)
        predictions.append(result)
    print('> predicted=' + repr(result) + ', actual=' + repr(test_labels[x]))
    accuracy = getAccuracy(test_labels, predictions)

    """print(train_labels[1])
    print(training_labels[1])
    print(validation_labels[9999])
    print(train_labels[59999])"""
    """print(len(train_images))
    print(len(train_images[0]))
    print(max(train_images[0]))
    print(train_labels[0])
    print(len(test_images))
    print(len(test_images[0]))
    print(max(test_images[0]))
    print(test_labels[0])
    show_mnist_image(test_images[0])
    show_mnist_image(train_images[0])"""

main()