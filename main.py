from sklearn.neural_network import MLPClassifier
import numpy as np
from PIL import Image
from sklearn.model_selection import cross_val_score, train_test_split


f = open("flower_images copy/flower_labels.csv")
names = f.readlines()
data = []
labels = []
img_rows = 32
img_cols = 32

for i in range(1, len(names), 1):
    names[i] = names[i].strip('\n')
    l = names[i].split(',')
    if (l[1] == "0"):
        labels.append(0)
        img = Image.open("flower_images copy/" + l[0])
        img = img.resize((img_rows, img_cols))
        img_array = np.array(img, dtype=np.float32)
        data.append(img_array)
    elif (l[1] == "5"):
        labels.append(1)
        img = Image.open("flower_images copy/" + l[0])
        img = img.resize((img_rows, img_cols))
        img_array = np.array(img, dtype=np.float32)
        data.append(img_array)

data = np.array(data)
labels = np.array(labels)
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, stratify=labels)
X_train_flattened = np.array([img.flatten() for img in X_train])
X_val_flattened = np.array([img.flatten() for img in X_val])
print(X_val_flattened.shape)
print(y_val.shape)
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
mlp.fit(X_train_flattened, y_train)
y_pred = mlp.predict(X_val_flattened)
crossvalidationscore = cross_val_score(mlp, X_val_flattened, y_val, cv=5, scoring='accuracy')
fivefoldaccuracy = np.mean(crossvalidationscore)
print(fivefoldaccuracy)
