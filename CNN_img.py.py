import keras
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.constraints import maxnorm
#from keras.optimizers import SGD 

def Normalize(x):
	min_val = np.min(x)
	max_val = np.max(x)
	x = (x - min_val)/(max_val - min_val)
	return x

batch_size      = 32
num_classes     = 10
epochs          = 100
smpl_id         = 3000
learning_rate   = 0.01
num_predictions = 20 
decay = learning_rate/epochs

(x_train,y_train),(x_test,y_test) = cifar10.load_data()

label_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
label_counts = dict(zip(*np.unique(y_train, return_counts = True)))

print("The number of Images for each class")
for key,value in label_counts.items():
	print('label counts of [{}] ({}) : {}'.format(key,label_names[key].upper(),value))
    
#x_train = x_train['data'].reshape((len(x_train['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
#x_test  = x_test['data'].reshape((len(x_test['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
	
x_train = Normalize(x_train)
x_test  = Normalize(x_test)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test,num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32,32,3), activation='relu', padding='same')) 
model.add(Dropout(0.2)) 
model.add(Conv2D(32, (3, 3), activation='relu', padding='same')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Conv2D(64, (3, 3), activation='relu', padding='same')) 
model.add(Dropout(0.2)) 
model.add(Conv2D(64, (3, 3), activation='relu', padding='same')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Conv2D(128, (3, 3), activation='relu', padding='same')) 
model.add(Dropout(0.2)) 
model.add(Conv2D(128, (3, 3), activation='relu', padding='same')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Flatten()) 
model.add(Dropout(0.2)) 
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3))) 
model.add(Dropout(0.2)) 
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3))) 
model.add(Dropout(0.2)) 
model.add(Dense(num_classes, activation='softmax'))

#sgd = SGD(lr=learning_rate, momentum=0.9, decay=decay, nesterov=False)
rmsprop = keras.optimizers.RMSprop(learning_rate, decay=decay)
model.compile(loss='categorical_crossentropy',optimizer=rmsprop,metrics=['accuracy'])
print(model.summary())

print('Start training the CNN Model')
datagen = ImageDataGenerator(featurewise_center=False,samplewise_center=False,zca_epsilon=1e-06,  
                             width_shift_range=0.1,height_shift_range=0.1,  
                             fill_mode='nearest',horizontal_flip=True)
datagen.fit(x_train)
cnn_model = model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),epochs=epochs,
                        steps_per_epoch=len(x_train)/batch_size,
                        validation_data=(x_test, y_test),workers=4)

results = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', results[0])
print('Test accuracy:', results[1])

plt.figure(figsize=(12,10))
plt.plot(cnn_model.history['acc'])
plt.plot(cnn_model.history['val_acc'])
plt.legend(['train','test'])
plt.ylabel('Accuracy %', fontsize=12)
plt.xlabel('Epochs', fontsize=12)
plt.title('Accuracy')
plt.savefig("Accuracy.png",dpi=300,format="png")

plt.figure(figsize=(12,10))
plt.plot(cnn_model.history['loss'])
plt.plot(cnn_model.history['val_loss'])
plt.legend(['train','test'])
plt.xlabel('Epochs', fontsize=12)
plt.title('Loss')
plt.savefig("Loss.png",dpi=300,format="png")
plt.figure()

smpl_img = x_train[smpl_id] 
smpl_lbl = y_train[smpl_id]

print('Image - Min Val: {}   Max val: {}'.format(smpl_img.min(),smpl_img.max()))
print('Image shape: {}'.format(smpl_img.shape))
print('Label - Label Id: {} Name: Cat'.format(smpl_lbl))
plt.imshow(smpl_img)  

classes = range(0,10)
class_labels = dict(zip(classes, label_names))

batch = x_test[0:9]
labels = np.argmax(y_test[0:9],axis = -1)
predictions = model.predict(batch, verbose = 1)
class_result = np.argmax(predictions,axis=-1)
fig, axs = plt.subplots(3, 3, figsize = (15, 6))
fig.subplots_adjust(hspace = 1)
axs = axs.flatten()

for i, img in enumerate(batch):
    for key, value in class_labels.items():
        if class_result[i] == key:
            title = 'Prediction: {}\nActual: {}'.format(class_labels[key], class_labels[labels[i]])
            axs[i].set_title(title)
            axs[i].axes.get_xaxis().set_visible(False)
            axs[i].axes.get_yaxis().set_visible(False)            
    axs[i].imshow(img.transpose([0,1,2]))    
plt.show()