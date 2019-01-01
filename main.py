
import numpy as np
import pickle
from keras.datasets import fashion_mnist
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from keras.preprocessing import image
import seaborn as sns

#preparing data ->
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
images_train = []
for image_train in x_train:
    images_train.append(image_train.flatten())
images_test = []
for image_test in x_test:
    images_test.append(image_test.flatten())
images_train = np.array(images_train)
images_test = np.array(images_test)

F_classifier = OneVsRestClassifier(LogisticRegression(verbose=1, max_iter=10))
F_classifier.fit(images_train, y_train)
conf_matrix = confusion_matrix(y_test, F_classifier.predict(images_test))
print("Confusion_matrix:")
print(conf_matrix)
print(F_classifier.score(images_test, y_test))

multi_class_F_classifier = LogisticRegression(verbose=1, max_iter=7, multi_class="multinomial", solver="sag")
multi_class_F_classifier.fit(images_train, y_train)
conf_matrix = confusion_matrix(y_test, multi_class_F_classifier.predict(images_test))
print("Confusion_matrix:")
print(conf_matrix)
sns.heatmap(conf_matrix)
print(multi_class_F_classifier.score(images_test, y_test))

pickle.dump(multi_class_F_classifier, open('multi_class_F_classifier.model', 'wb'))

multi_class_F_classifier_from_file = pickle.load(open('multi_class_F_classifier.model', 'rb'))
conf_matrix = confusion_matrix(y_test, multi_class_F_classifier_from_file.predict(images_test))
print("Confusion_matrix:")
print(conf_matrix)
sns.heatmap(conf_matrix)

image_file = 'pencil.jpg'
img = image.load_img(image_file, target_size=(28,28), color_mode="grayscale")
x = image.img_to_array(img)
print(x.shape)
print(x.flatten())
print(multi_class_F_classifier.predict(x.flatten().reshape(1, -1)))