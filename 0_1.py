import numpy as np  
from keras.models import Sequential
from keras.models import model_from_json

with open("model.config", "r") as text_file:
    json_string = text_file.read()

    
model = Sequential()
model = model_from_json(json_string)
model.load_weights("model.weight", by_name=False)

# read my data
for i in range(0, 10):
    X2 = np.genfromtxt(str(i)+'.csv', delimiter=',').astype('float32')  
    X1 = X2.reshape(1,28*28) / 255
    predictions = model.predict_classes(X1)
    # get prediction result
    print(predictions)

#from matplotlib import pyplot as plt
#plt.imshow(X2.reshape(28,28)*255)
#plt.show() 

