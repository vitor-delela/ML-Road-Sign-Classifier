from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from GermanDatasetMgmt import GermanDataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

model = load_model('model_10epochs_rmsprop_dropout.h5')
class_mapping = pd.read_csv('./German_Dataset/signname.csv') 

my_loader = GermanDataLoader() 
(train_images, train_labels), \
(valid_images, valid_labels), \
(test_images, test_labels) = my_loader.load_data()

print("len(test_labels): ", len(test_labels))
print("test_labels: ", test_labels)

y_true = test_labels # rótulos verdadeiros
y_pred = np.argmax(model.predict(test_images), axis=1) # rótulos previstos
unique_labels = sorted(set(y_true).union(y_pred))

cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
plt.figure(figsize=(35, 35))
plt.tight_layout()
plt.xticks(rotation=45, ha='right') 
plt.yticks(rotation=45) 
heatmap = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_mapping.loc[unique_labels, 'SignName'].values, yticklabels=class_mapping.loc[unique_labels, 'SignName'].values)

plt.title("Confusion Matrix")
plt.xlabel("Predict")
plt.ylabel("Correct")
plt.savefig('./confusion_matrix.png')
# plt.show()
