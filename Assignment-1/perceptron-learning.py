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


def predict_value(w,instance):
    sum = 0.0
    for i in range(len(w)):
        sum+=float(w[i])*int(instance[i])
    if(sum>0.0):
        return 1
    else:
        return -1

def update_weight(w,instance,actual,predicted):
    for i in range(len(w)):
        w[i]= w[i] + 0.1* (int(actual)-int(predicted))* int(instance[i])
    #print(w)
    return w


def one_training_pass(training_set,labels,w):
    print('iteration')
    weight_change = 0;
    for i in range(len(training_set)):
        predicted = predict_value(w,training_set[i])
        if(predicted != labels[i]):
            print('not equal')
            w = update_weight(w,training_set[i],labels[i],predicted)
            weight_change += 1
    return w,weight_change

def train_perceptron_for_n(training_set,training_labels,n):

    new_labels = training_labels
    #initial w vector
    w = [0]
    for i in range(len(training_set[0])):
        w.append(0)
    #append 1 for w0
    x = np.array(training_set)
    y = []
    for i in range(len(training_set)):
        y.append([1])
    new_training_set = np.append(y, x, axis=1)
    for i in range(len(new_labels)):
        if(new_labels[i]==n):
            new_labels[i] = 1
        else:
            new_labels[i] = -1
    no_of_steps = 1
    #print(new_labels)
    #print(new_labels)
    w,change = one_training_pass(new_training_set,new_labels,w)

    while (int(change/len(new_training_set))!=0 or no_of_steps==0):
        print('...........................................................')
        print(change)
        print(w)
        w,change = one_training_pass(new_training_set,new_labels,w)
        no_of_steps += 1
    #print(w)

    #print(new_training_set)
    print(w)
    print(len(w))
    return w



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
    train_images,train_labels = loadDataset("mnist_train1.csv")
    test_images,test_labels = loadDataset("mnist_test.csv")
    #p0 = train_perceptron_for_n(train_images,train_labels,0)638 Misclassifications
    #p0=[-5754.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -26.0, -494.0, 1328.0, 3328.0, 1403.0, -226.0, 5919.0, 15673.0, 7589.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -104.0, 0.0, -17.0, -1258.0, -3247.0, -4311.0, -8047.0, -3754.0, -7247.0, -14106.0, -10400.0, -17650.0, -13767.0, -9831.0, -17252.0, -17034.0, -12097.0, -19006.0, -11490.0, -4991.0, -2995.0, -988.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -812.0, -762.0, -1139.0, -6267.0, -5988.0, 1949.0, -11972.0, -26428.0, -2054.0, -17263.0, 5738.0, -9627.0, -2995.0, -5766.0, -1872.0, -9440.0, -3588.0, -4989.0, -14011.0, -8352.0, 1062.0, -2027.0, -830.0, -188.0, 0.0, 0.0, 0.0, 0.0, -40.0, -98.0, -10951.0, -366.0, -5574.0, -5713.0, 5338.0, -8884.0, 1768.0, -1602.0, 4373.0, -7567.0, 7508.0, -10823.0, 5573.0, -3486.0, 3023.0, -1270.0, 136.0, -393.0, -1871.0, -9356.0, -4839.0, -450.0, 0.0, 0.0, 0.0, 0.0, -190.0, -871.0, -6408.0, 10797.0, -4624.0, -9731.0, 4425.0, 5825.0, -1535.0, -5069.0, 5678.0, -2353.0, 2656.0, -2676.0, -862.0, 1490.0, -5686.0, 334.0, -4252.0, -1362.0, -1575.0, -4328.0, -9092.0, -979.0, -820.0, 0.0, 0.0, -91.0, -502.0, -5231.0, -304.0, -4529.0, -989.0, 4272.0, -15300.0, 1611.0, -2112.0, -1927.0, 2345.0, 790.0, 767.0, -3156.0, 2203.0, -712.0, 3470.0, -4078.0, 6305.0, 4146.0, -3343.0, -11136.0, -25113.0, -9310.0, -2378.0, 0.0, -688.0, -166.0, -718.0, -4479.0, 1545.0, 706.0, 2226.0, -7680.0, 4300.0, 607.0, 193.0, -978.0, 3381.0, -1358.0, -1270.0, -5407.0, 4157.0, -1479.0, 4532.0, -6102.0, 688.0, -2580.0, 3625.0, -3017.0, -15695.0, -16573.0, -183.0, 3498.0, 0.0, 689.0, 2495.0, -4386.0, -10321.0, -4195.0, 4952.0, 1442.0, -424.0, 3164.0, -4289.0, 5361.0, -4823.0, 6042.0, -1818.0, 9417.0, -860.0, 3447.0, -3601.0, 6156.0, -915.0, 3463.0, -440.0, -12263.0, -18206.0, -8687.0, -324.0, 0.0, 0.0, 1553.0, 1736.0, -10384.0, 2937.0, 2951.0, -9633.0, 581.0, -38.0, 5633.0, -3343.0, -4046.0, 2917.0, -1715.0, -1752.0, 2165.0, 2179.0, -667.0, 8728.0, -1513.0, 756.0, -8639.0, 4080.0, -3622.0, -17755.0, -14771.0, -78.0, 0.0, 0.0, 5937.0, -3686.0, 4714.0, 4789.0, 52.0, 6293.0, -1856.0, 338.0, -2581.0, 1655.0, 1501.0, -3441.0, 2002.0, 2395.0, -98.0, 3643.0, 5201.0, -4448.0, 3671.0, -1139.0, 11313.0, 2811.0, 86.0, -20706.0, -19028.0, 0.0, 0.0, 1650.0, -3566.0, 943.0, -11766.0, -6569.0, 7369.0, 836.0, -198.0, -1321.0, -3308.0, 4083.0, -7217.0, 2176.0, -8400.0, -4691.0, -2236.0, 2454.0, -2285.0, 5847.0, -1405.0, 4848.0, -7004.0, 4633.0, 1699.0, -13482.0, -11592.0, 0.0, 0.0, 16.0, -1188.0, 1790.0, -13076.0, 3682.0, -7870.0, -684.0, -2982.0, 1926.0, -3058.0, -1690.0, 506.0, 2164.0, -3067.0, -10763.0, -3499.0, -796.0, -697.0, 1053.0, 5173.0, -463.0, 332.0, -2398.0, 5476.0, -5056.0, -6047.0, 0.0, 0.0, 63.0, 49.0, 5491.0, 4176.0, -2491.0, 13268.0, 810.0, 167.0, -1646.0, 2753.0, 1374.0, -376.0, -10600.0, 2412.0, -11471.0, -2263.0, -2690.0, 150.0, 1798.0, -4137.0, -2260.0, 5738.0, 3149.0, -7028.0, 10018.0, -3449.0, 0.0, 0.0, 0.0, -172.0, -7394.0, -48.0, 1849.0, -3204.0, 5669.0, 885.0, 5601.0, -5145.0, 736.0, 3314.0, -608.0, -17964.0, 948.0, -5306.0, 2592.0, 173.0, 1745.0, 4252.0, 1026.0, -2820.0, 7206.0, -4221.0, 1470.0, -12228.0, 0.0, 0.0, 0.0, -1211.0, -13496.0, 4930.0, 3145.0, -208.0, 6946.0, -862.0, 1793.0, 5254.0, -2687.0, 1654.0, -4068.0, -10492.0, -4007.0, -5943.0, -955.0, -2992.0, -4491.0, 1454.0, -1113.0, 7817.0, -5614.0, 14298.0, -5642.0, -8738.0, 0.0, 0.0, 0.0, -666.0, -12109.0, -5991.0, 6232.0, -3667.0, 3330.0, 2364.0, 3464.0, -640.0, 2736.0, -6259.0, -12082.0, -11198.0, 3299.0, -6214.0, 2040.0, -4552.0, 1498.0, 1229.0, 1780.0, -10153.0, 2193.0, -10727.0, -15540.0, -15077.0, 816.0, 0.0, 0.0, 0.0, -7558.0, 1589.0, 2480.0, 2026.0, 3082.0, -2544.0, -962.0, 651.0, -4036.0, 4643.0, -8390.0, -4825.0, -3816.0, -5556.0, 100.0, -2576.0, 1035.0, 2586.0, -5946.0, 11453.0, 318.0, 1967.0, -5499.0, -3133.0, 5162.0, 0.0, 0.0, -392.0, -7973.0, 972.0, -2485.0, -976.0, -3493.0, 4084.0, 1014.0, 6738.0, 1659.0, -4876.0, -9490.0, -4263.0, 801.0, -728.0, 1420.0, 525.0, -2254.0, -8889.0, 396.0, -4223.0, -4688.0, 4482.0, -7633.0, -6828.0, -4484.0, 0.0, 416.0, -481.0, -13440.0, -9039.0, -3010.0, 4699.0, -2133.0, 2722.0, -550.0, 2980.0, -5959.0, 11664.0, -10657.0, 1568.0, -5154.0, 168.0, -2531.0, 324.0, -3259.0, 5967.0, -1819.0, 3021.0, -2078.0, 1107.0, -12200.0, -1598.0, -256.0, 0.0, 1048.0, -906.0, -13122.0, 2197.0, 271.0, -4426.0, 6571.0, -1525.0, 15.0, 2801.0, 7139.0, -1257.0, 3147.0, -3938.0, 1836.0, -5725.0, -1356.0, 2352.0, -1967.0, 2715.0, 1645.0, -4078.0, -6430.0, -9246.0, -3354.0, 137.0, 0.0, 0.0, 0.0, -4670.0, 125.0, 4791.0, -823.0, 2980.0, -3005.0, -1372.0, 4331.0, 1747.0, 486.0, 4892.0, 259.0, -4735.0, -913.0, 3268.0, -10104.0, -364.0, -7183.0, -4438.0, -5188.0, 11201.0, -9946.0, -2947.0, -338.0, 5761.0, 0.0, 0.0, 0.0, -2923.0, 2452.0, -12241.0, -1761.0, 2000.0, 1269.0, 692.0, -1232.0, 527.0, 2155.0, -5796.0, 11490.0, -3203.0, -29.0, -239.0, 516.0, -7763.0, 3243.0, 6632.0, -3173.0, -159.0, -3590.0, -13346.0, 223.0, 712.0, 0.0, 0.0, 0.0, -376.0, -945.0, -14589.0, 1065.0, 3416.0, -3581.0, -2238.0, 1353.0, 3848.0, -2225.0, -34.0, -457.0, 2817.0, -2999.0, -4966.0, 8488.0, -8588.0, 799.0, -17110.0, -3007.0, 2766.0, -1748.0, -13619.0, 270.0, 0.0, 0.0, 0.0, 0.0, -1085.0, -1670.0, 1961.0, -1219.0, -2591.0, -1708.0, -1423.0, 4226.0, -4575.0, -3033.0, 4307.0, -4110.0, 2240.0, 1406.0, -9490.0, 3780.0, 781.0, -10329.0, -6357.0, -6538.0, -9305.0, -7373.0, -340.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2512.0, -6587.0, -12540.0, -15691.0, -20639.0, -5856.0, -21110.0, -29.0, -13924.0, -17823.0, -9091.0, -18244.0, -6944.0, -20540.0, -24922.0, -11418.0, -15667.0, -13303.0, -7947.0, -5311.0, -495.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1068.0, -1492.0, -2616.0, -5313.0, -9429.0, -8518.0, -15063.0, -9232.0, -14586.0, -14235.0, -16395.0, -13118.0, -9134.0, -7466.0, -983.0, -8300.0, -10617.0, -5729.0, -1515.0, -28.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -114.0, -490.0, -316.0, 0.0, -8.0, -956.0, -3507.0, -74.0, -1411.0, -2794.0, -1878.0, -997.0, -585.0, -732.0, -206.0, -253.0, -1960.0, -6096.0, -1392.0, 0.0, 0.0, 0.0, 0.0]
    p1 = train_perceptron_for_n(train_images,train_labels,1)
    p2 = train_perceptron_for_n(train_images,train_labels,2)
    p3 = train_perceptron_for_n(train_images,train_labels,3)
    p4 = train_perceptron_for_n(train_images,train_labels,4)
    p5 = train_perceptron_for_n(train_images,train_labels,5)
    p6 = train_perceptron_for_n(train_images,train_labels,6)
    p7 = train_perceptron_for_n(train_images,train_labels,7)
    p8 = train_perceptron_for_n(train_images,train_labels,8)
    p9 = train_perceptron_for_n(train_images,train_labels,9)
    x = np.array(test_images)
    y = []
    for i in range(len(test_images)):
        y.append([1])
    new_testing_data = np.append(y, x, axis=1)
    predictions = []
    for i in range(len(new_testing_data)):
        if(predict_value(p0,new_testing_data[i])==1):
            predictions.append(0)
        else:
            if(predict_value(p1,new_testing_data[i])==1):
                predictions.append(1)
            else:
                if(predict_value(p2,new_testing_data[i])==1):
                    predictions.append(2)
                else:
                    if(predict_value(p3,new_testing_data[i])==1):
                        predictions.append(3)
                    else:
                        if(predict_value(p4,new_testing_data[i])==1):
                            predictions.append(4)
                        else:
                            if(predict_value(p5,new_testing_data[i])==1):
                                predictions.append(5)
                            else:
                                if(predict_value(p6,new_testing_data[i])==1):
                                    predictions.append(6)
                                else:
                                    if(predict_value(p7,new_testing_data[i])==1):
                                        predictions.append(7)
                                    else:
                                        if(predict_value(p8,new_testing_data[i])==1):
                                            predictions.append(8)
                                        else:
                                            if(predict_value(p9,new_testing_data[i])==1):
                                                predictions.append(9)
                                            else:
                                                predictions.append(0)
    print("predicted value "+prediction[i]+"actual value "+test_labels[i])
    accuracy = getAccuracy(test_labels, predictions)
    print("accuracy "+accuracy)





main()