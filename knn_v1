# 1. Read Dataset
# load data in numpy arrays and check the data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_dir = "data/"
def data_load(data_dir):
    data_train = pd.read_csv(data_dir + "train.csv")
    print("shape of train.csv: ", data_train.shape)
    label_train = data_train.values[:,0]
    image_train = data_train.values[:,1:]
    
    data_test = pd.read_csv(data_dir + "test.csv")
    print("shape of test.csv: ", data_test.shape)
    
    return label_train, image_train, data_test.values
    
label_train, image_train, image_pred = data_load(data_dir)

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
# 2. Train using model of SK Learn
# 80% of image_train for traning dataset and 20% of image_train for test dataset
from sklearn.model_selection import train_test_split

image_train, image_test, label_train, label_test = train_test_split(image_train, label_train,
                                                                   test_size = 0.2,
                                                                   random_state = 0)

print("shape of train: ", image_train.shape)
print("shape of test: ", image_test.shape)

# Use sklearn package and find the optimum k
import time
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

k_range = range(1, 6)
k_ans = 0

def opt_k(image_train, label_train, image_test, label_test, range_k):
    max = 0
    k_ans = 0
    accuracy_scores = []
    for k in range_k:
        print("when k = " + str(k), "training begins")
        start_time = time.time()
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(image_train, label_train)
        label_pred = knn.predict(image_test)
        accuracy = accuracy_score(label_test, label_pred)
        print(classification_report(label_test, label_pred))
        print(confusion_matrix(label_test, label_pred))
        end_time = time.time()
        print("   computing time: " + str(end_time - start_time), "and accuracy = " + str(accuracy))
        accuracy_scores.append(accuracy)
        if accuracy > max:
            k_ans = k
            max = accuracy        
    plt.plot(range_k, accuracy_scores)
    plt.xlabel("k")
    plt.ylabel("accuracy")
    plt.show()
    print("the optimum k is ", k_ans)
    return k_ans

k = opt_k(image_train, label_train, image_test, label_test, range(1,6))

knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(image_train, label_train)
label_pred = knn.predict(image_pred)
print(label_pred)

# let's check it
i = 987
plt.imshow(image_pred[i].reshape(28,28))
plt.show()

print("KNN Predicts digit: ", label_pred[i])

# output 
output = pd.DataFrame({"ImageId": range(1, len(label_pred) + 1), "Label": label_pred})
print (output)
output.to_csv("output.csv", index = False, header = True)
