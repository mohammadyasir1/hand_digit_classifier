import tensorflow
import keras
from keras import Sequential
from keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn.metrics import accuracy_score
import numpy as np

(X_train, y_train),(X_test, y_test) = keras.datasets.mnist.load_data()

# scaling the values in 0-1 range
X_train = X_train/255
X_test = X_test/255

# creating ANN architechtures
model = Sequential()

# converting 28x28 into (1,784) using keras flatten layer that convert higher dimension layer/array into 1D
model.add(Flatten(input_shape=(28,28))) # input layer
model.add(Dense(128, activation='relu')) # hidden layer
model.add(Dense(64, activation='relu')) # hidden layer
model.add(Dense(10, activation='softmax')) # use softmax for multiclass classification

print(model.summary())

# I use sparse_categorical_crossentropy bcz i dont want to encode one hot
# whereas if we use categorical_crossentropy the we have to one hot endcoding for labels
model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

y_prob = model.predict(X_test)
y_pred = y_prob.argmax(axis=1)
acc = accuracy_score(y_test, y_pred)

print('model accuracy: ',acc*100)

##### opencv code #####
drawing = False # true if mouse is pressed
pt1_x , pt1_y = None , None

# mouse callback function
def draw(event,x,y,flags,param):
    global pt1_x,pt1_y,drawing

    if event==cv.EVENT_LBUTTONDOWN:
        drawing=True
        pt1_x,pt1_y=x,y

    elif event==cv.EVENT_MOUSEMOVE:
        if drawing==True:
            cv.line(pad,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=1)
            pt1_x,pt1_y=x,y
    elif event==cv.EVENT_LBUTTONUP:
        drawing=False
        cv.line(pad,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=1)

pad = np.zeros((28, 28))
cv.namedWindow('Pad')
ans = np.zeros((50,250))
cv.setMouseCallback('Pad', draw)

font = cv.FONT_HERSHEY_SIMPLEX

while True:
    cv.imshow('Pad', pad)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv.waitKey(1) & 0xFF == ord('c'):
        ans = np.zeros((50,250))
        # gray = cv.cvtColor(pad, cv.COLOR_RGB2GRAY)
        resized_img = cv.resize(pad, (28,28))
        scaled_img = resized_img/255
        img_reshaped = np.reshape(scaled_img, [1,28,28])
        pred = model.predict(img_reshaped)
        lab = np.argmax(pred)
        print('I guess it\'s a :',lab)
        cv.putText(ans, 'I guess it\'s a : '+str(lab), (25,25), font, 0.7, (255,255,255), thickness=2,lineType=cv.LINE_AA)
        pad = np.zeros((28,28))

    cv.imshow('Pad', pad)
    cv.imshow('Answer', ans)
