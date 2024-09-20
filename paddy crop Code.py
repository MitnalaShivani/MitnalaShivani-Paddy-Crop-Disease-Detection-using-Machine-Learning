from keras.utils.np_utils import to_categorical

from keras.models import Sequential, load model

import pickle

from sklearn.model selection import train_test split

from sklearn.metrics import accuracy score

from keras.callbacks import ModelCheckpoint

from keras.layers import MaxPooling2D

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D ww

from keras.layers import Dense, Dropout, Activation, Flatten

global filename, canvas, text, images, root
global X, Y, filename, X train, X test, y train, y test, cnn model labels = ['Leaf Blast', 'BrownSpot', 'Hispa', 'Healthy'] global accuracy, precision, recall, fscore

def getID(name):

index = 0

for i in range(len(labels)):

if labels[i] == name:

index = i

break

return index

def uploadDataset():

global filename

filename = filedialog.askdirectory(initialdir=".")

text.delete('1.0', END)

text.insert(END, filename+" dataset loaded\n\n")

Y = np.load('model/Y.txt.npy')

unique, count = np.unique (Y, return counts = True)

height = count
bars = labels

y pos = np.arange(len(bars))

plt.bar(y pos, height)

plt.xticks(ypos, bars)

plt.xlabel("Disease Names")

plt.ylabel("Images Count")

plt.title("Different Diseases found in Dataset")

plt.show()

def preprocessDataset():

global X, Y, filename, X train, X test, y train, y test
text.delete('1.0', END)

if os.path.exists("model/X.txt.npy"):

X = np.load('model/X.txt.n

Y = np.load('model/Y.txt.npy')

else:

X = []

Y = []

for root, dirs, directory in os.walk(filename):

for j in range(len(directory)):

name = os.path.basename(root)

if 'Thumbs.db' not in directory[j]:

img = cv2.imread(root+"/"+directory[j])

img = cv2.resize(img, (32, 32))

im2arr = np.array(img)

im2arr = im2arr.reshape(32, 32, 3)

X.append(im2arr)

label = getID(name)

Y.append(label)

X = np.asarray(X)

Y = np.asarray(Y)

np.save('model/X.txt',X)

np.save('model/Y.txt',Y)

X = X.astype('float32')

X = X/255

indices = np.arange(X.shape[0])

np.random.shuffle(indices)

X = X[indices]

Y = Y[indices]

Y = to categorical(Y)

X train, X test, y train, y test = train_test split(X, Y, test size=0.2) #split dataset into train and test
text.insert(END, "Total Images found in dataset: "+str(X.shape[0])+"\n\n")

text.insert(END,"Dataset Train & Test Split Details\n\n")

text.insert(END, "80% dataset used for training: "+str(X train.shape[0])+"\n")

text.insert(END, "20% dataset used for training: "+str(X test.shape[0])+"\n")

ext.update_idletasks()

test = X[3]

cv2.imshow("Sample Processed Image",cv2.resize(test, (150,250)))

cv2.waitKey(0)

#function to calculate all metrics

def calculateMetrics(algorithm, testy, predict):

p = precision score (testy, predict, average='macro') * 100

r = recall_score(testy, predict, average='macro') * 100 www

f = f1_score(testY, predict, average='macro') * 100

a = accuracy score (testy, predict)

accuracy.append(a)

precision.append(p)

recall.append(r)

fscore.append(f)

text.insert(END, algorithm+" Accuracy: "+str(a)+"\n")

text.insert(END, algorithm+" Precision: "+str(p)+"\n")

text.insert(END, algorithm+" Precision: "+str(p)+"\n")

text.insert(END, algorithm+" Recall: "+str(r)+"\n")

text.insert(END, algorithm+" FSCORE: "+str(f)+"\n\n")

labels = ['Leaf Blast', 'BrownSpot', 'Hispa', 'Healthy']

conf_matrix = confusion_matrix(testy, predict)

ax= sns.heatmap(conf matrix, xticklabels = labels, yticklabels = labels, annot = True, map="viridis", fmt="g");

ax.set ylim([0,len(labels)])

plt.title(algorithm+" Confusion matrix")

plt.ylabel('True class')

plt.xlabel('Predicted class')

plt.show()

def runSVM():

global X train, X test, y train, y test

global accuracy, precision, recall, fscore

text.delete('1.0', END)

accuracy =

precision = []

recall = []

fscore = [] www

X_train1 = np.reshape(X train, (X_train.shape[0], (X train.shape[1] * X train.shape[2] * X train.shape[3])))

X_test1 = np.reshape(X test, (X test.shape[0], (X_test.shape[1] * X_test.shape[2] * X test.shape[3])))

y_train1 = np.argmax(y train, axis=1)

y_test1 = np.argmax(y test, axis=1)

X_train1 = X_train1 [0:500,0:100]

X_test1 = X_test1[600:700,0:100]

y_train1 = y_train1[0:500]

y_test1 = y_test1[600:700]

svm_cls = svm.SVC()
cnn_model.add(Convolution2D(32, (2, 2), activation = 'relu'))

cnn model.add(MaxPooling2D (pool size = (3, 3)))

cnn_model.add(Flatten())

cnn_model.add(Dense(units = 256, activation = 'relu'))

cnn model.add(Dense(units = y train.shape[1], activation = 'softmax'))

cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy']) ~

if os.path.exists("model/cnn_weights.hdf5") == False:

model_check point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save best only = True)

hist = cnn_model.fit(X train, y train, batch_size = 16, epochs = 80, validation_data=(X_test, y test), callbacks=[model_check point], verbose=1)

else:

cnn_model = load model("model/cnn_weights.hdf5")

predict = cnn model.predict(X test)

predict = np.argmax(predict, axis=1)

y_test1 = np.argmax(y test, axis=1)

calculate Metrics("CNN Algorithm", y_test1, predict)

def graph():

df = pd.DataFrame ([['SVM', 'Precision', precision [0]], ['SVM', 'Recall', recall[0]],['SVM', 'F1 Score', fscore[0]], ['SVM', 'Accuracy', accuracy[0]],

www

['CNN', 'Precision', precision[1]], ['CNN', 'Recall', recall[1]], ['CNN','F1Score', fscore[1]], ['CNN', 'Accuracy', accuracy[1]], ], columns=['Parameters', 'Algorithms', 'Value']) www www

df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')

plt.show()

def getDisease(name):

output =

if name == "Hispa":

output = "Chemical: Quinalphos 25 EC Dosage: 2000 ml/ha\n"

output += "Chemical: Monocrotophos 36 WSC Dosage: 850 ml/ha\n"
output += "Chemical: Chlorpyriphos 20 EC Dosage: 1500 ml/ha\n"

output += "Chemical: Triazophos 40 EC Dosage: 2 ml/liter of water"

if name == "Leaf Blast":

output = "Chemical: Tricylazole 75% WP Dosage: 0.6 gm/litre\n" ww

output += "Chemical: Propiconazole 25% EC Dosage: 1 ml/litre\n"

output += "Chemical: Carbendazims 50% WP Dosage: 1gm/litre\n"

if name == 'BrownSpot':

output = "At tillering and late booting stages, spray Carbendazim 12% + Mancozeb 63% WP @ 1gm/litre\n"

output += "Zineb @ 2gm/litre of water\n"

output += "After 15 days, repeat the spray"

return output

def predictDisease(filename):

global cnn model

text.delete('1.0', END)

image = cv2.imread(filename)

img = cv2.resize(image, (32, 32))

im2arr = np.array(img)

im2arr = im2arr.reshape(1,32,32,3)
img = np.asarray(im2arr)

img = img.astype('float32')

img = img/255

preds = cnn_model.predict(img) www

predict = np.argmax(preds)

max_value = np.amax(preds)

remedies =

if max value > 0.95:

img = cv2.imread(filename)

img = cv2.resize(img, (700,300))

cv2.putText(img, 'Paddy Disease Predicted as: '+labels [predict], (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.9, (0, 0, 255), 2)

text.insert(END, 'Paddy Disease Predicted as: '+labels [predict]+"\n") www

.insert(END, "REMEDIES DETAILS\n")

remedies = getDisease (labels [predict])

text.insert(END, remedies)

else:

img = cv2.imread(filename)

img = cv2.resize(img, (700,400))

cv2.putText(img, 'Not a paddy leaf image', (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.9, (0, 0, 255), 2)

text.insert(END, 'Paddy Disease Predicted as: '+labels [predict])

cv2.imwrite("images/output.png", img)

def predict():

global canvas, images, root

filename = filedialog.askopenfilename(initialdir="testlmages") www www

predictDisease(filename)

img = Image.open("images/output.png")

img = img.resize((700, 300))

picture = ImageTk.Photolmage(img)

canvas.configure(image = picture)

canvas.image = picture

root.update_idletasks()

def Main():

global text, canvas, images, root

root = tkinter.Tk()

root.geometry("1300x1200")

root.title("Paddy Crop Disease Detection using Machine Learning")

root.resizable(True, True)

font = ('times', 14, 'bold')

title = Label(root, text='Paddy Crop Disease Detection using Machine Learning')

title.config(bg='yellow3', fg='white')

title.config(font=font)

title.config(height=3, width=120)

title.place(x=0,y=5)

font1 = ('times', 12, 'bold')

img = Image.open("images/download.png")

img.resize((600, 300))

picture = ImageTk. Photolmage(img)

canvas = Label(root, image=picture)

canvas.place(x=300,y=200)

uploadButton = Button(root, text="Upload Paddy Disease Dataset", command=uploadDataset)

uploadButton.place(x=60,v=80)

uploadButton.config(font=font1)
uploadButton = Button(root, text="Upload Paddy Disease Dataset", command=uploadDataset)

uploadButton.place( x = 60 y = 80 )

uploadButton.config(font=font1)

preprocessButton = Button(root, text="Preprocess Dataset", command preprocessDataset)

preprocessButton.place(x=400,y=80)

preprocessButton.config(font=font1)

svmButton = Button(root, text="Run SVM Algorithm", command=runSVM)

svmButton.place( x = 600 y=80)

svmButton.config(font=font1)

cnnButton = Button(root, text="Run CNN Algorithm", command=runCNN)

cnnButton.place (x = 60, y = 130)

cnnButton.config(font=font1)

graphButton = Button(root, text="Comparison Graph", command=graph)

graphButton.place(x=400,y=130)

graphButton.config(font=font1)

predictButton = Button(root, text="Predict Disease from Test Image", command=predict)

predictButton.place(x=600,y=130)

predictButton.config(font=font1)

text=Text(root,height=10, width=140)

scroll Scrollbar(text)

text.configure(yscrollcommand=scroll.set)

text.place(x=10,y=510)

root.mainloop()

if _name_ == '_main_' :

Main ()



