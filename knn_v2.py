# 1. Read Dataset
# load data in numpy arrays and check the data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_dir = "data/"

def load_data(data_dir):
    data_train = pd.read_csv(data_dir + "train.csv")
    print("shape of train.csv: ", data_train.shape)
    
    label_train = data_train.values[:,0]
    image_train = data_train.values[:,1:]
    
    data_test = pd.read_csv(data_dir + "test.csv")
    print("shape of test.csv: ", data_test.shape)
    
    return label_train, image_train, data_test.values

label_train, image_train, image_pred = load_data(data_dir)

# plot images of 0~9
row_plot = 5
for num in range(0,10):
    index = np.nonzero([i == num for i in label_train])
    index = np.random.choice(index[0], row_plot)
    for i in range(0,row_plot):
        plt.subplot(row_plot, 10, i * 10 + num + 1)
        plt.imshow(image_train[index[i]].reshape(int(image_train.shape[1] ** 0.5), int(image_train.shape[1] ** 0.5)))
        plt.axis("off")
        if i == 0:
            plt.title(str(num))
            
plt.show()

############################################################
# 2. Train using model of KNN
# 80% of image_train for traning dataset and 20% of image_train for test dataset
from sklearn.model_selection import train_test_split

image_train, image_test, label_train, label_test = train_test_split(image_train, label_train,
                                                                   test_size = 0.2,
                                                                   random_state = 0)

print("shape of train: ", image_train.shape)
print("shape of test: ", image_test.shape)

# build class of KNN
class KNN(object):
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, X_test):
        y_pred = []
        for i in range(len(X_test)):
            diffMatrix = np.tile(X_test[i], (len(self.X_train), 1)) - self.X_train
            distances = np.sum(diffMatrix ** 2, axis = 1) ** 0.5
            sortedDistancesIndex = distances.argsort()
            classifier = {}
            for n in range(self.n_neighbors):
                vote = self.y_train[sortedDistancesIndex[n]]
                classifier[vote] = classifier.get(vote, 0) + 1
                
            max = 0
            prediction = 0
            for k, v in classifier.items():
                if v > max:
                    prediction = k
                    max = v
            
            y_pred.append(prediction)
            
        return(y_pred)

# find optimum k
import time
from sklearn.metrics import accuracy_score

def opt_k(image_train, label_train, image_test, label_test, range_k):
    max = 0
    k_ans = 0
    accuracy_scores = []
    for k in range_k:
        print("when k = " + str(k), "training begins")
        start_time = time.time()
        knn = KNN(k)
        knn.fit(image_train, label_train)
        label_pred = knn.predict(image_test)
        accuracy = accuracy_score(label_test, label_pred)
        end_time = time.time()
        print("   computing time: " + str(end_time - start_time) + "and accuracy = " + str(accuracy))
        accuracy_scores.append(accuracy)
        if accuracy > max:
            k_ans = k
            max = accuracy
            
    plt.plot(range_k, accuracy_scores)
    plt.xlabel("k")
    plt.ylabel("accuracy")
    
    print("the optimum k is ", k_ans)
    
    return k_ans

k = opt_k(image_train, label_train, image_test, label_test, range(1,6))

knn = KNN(k)
knn.fit(image_train, label_train)
label_pred = knn.predict(image_pred)
print(label_pred)

# let's check it
i = 311
plt.imshow(image_pred[i].reshape(28,28))
plt.show()

print("KNN Predicts digit: ", label_pred[i])

# output 
output = pd.DataFrame({"ImageId": range(1, len(label_pred) + 1), "Label": label_pred})
print (output)
output.to_csv("output.csv", index = False, header = True)
