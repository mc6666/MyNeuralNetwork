from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import os
import sys

# 計算相似矩陣
def cosine_similarity(ratings):
    sim = ratings.dot(ratings.T)
    if not isinstance(sim, np.ndarray):
        sim = sim.toarray()
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)
    
def main():
    # 自 images 目錄找出所有 JPEG 檔案    
    y_test=[]
    x_test=[]
    for img_path in os.listdir("images"):
        if img_path.endswith(".jpg"):
            img = image.load_img("images/"+img_path, target_size=(224, 224))
            y_test.append(img_path[0:4])
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            if len(x_test) > 0:
                x_test = np.concatenate((x_test,x))
            else:
                x_test=x
    
    # 轉成 VGG 的 input 格式
    x_test = preprocess_input(x_test)

    # include_top=False，表示會載入 VGG16 的模型，不包括加在最後3層的卷積層，通常是取得 Features (1,7,7,512)
    model = VGG16(weights='imagenet', include_top=False) 


    # 萃取特徵
    features = model.predict(x_test)
    # 計算相似矩陣
    features_compress = features.reshape(len(y_test),7*7*512)
    sim = cosine_similarity(features_compress)

    # 隨機取1個樣本測試
    #inputNo = np.random.randint(0,len(y_test),1)[0]
    # 依命令行參數，取1個樣本測試測試
    inputNo = int(sys.argv[1]) # tiger, np.random.randint(0,len(y_test),1)[0]
    top = np.argsort(-sim[inputNo], axis=0)[1:3]

    # 取得最相似的前2名序號
    recommend = [y_test[i] for i in top]
    print(recommend)

if __name__ == "__main__":
    main()
