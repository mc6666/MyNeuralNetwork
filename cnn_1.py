import numpy as np  
from keras.models import Sequential
from keras.models import model_from_json
from keras import backend as K

with open("cnn.config", "r") as text_file:
    json_string = text_file.read()

    
model = Sequential()
model = model_from_json(json_string)
model.load_weights("cnn.weight", by_name=False)

# read my data
img_rows, img_cols = 28, 28
for i in range(0, 10):
    X2 = np.genfromtxt(str(i)+'.csv', delimiter=',').astype('float32')  
    if K.image_data_format() == 'channels_first':
        X1 = X2.reshape(1, 1, img_rows, img_cols)
    else: # channels_last: 色彩通道(R/G/B)資料放在第4維度，第2、3維度放置寬與高
        X1 = X2.reshape(1, img_rows, img_cols, 1)
        
    X1 = X1 / 255
    predictions = model.predict_classes(X1)
    # get prediction result
    print(predictions)

# display 9 image    
#from matplotlib import pyplot as plt
#plt.imshow(X2.reshape(28,28)*255)
#plt.show() 

