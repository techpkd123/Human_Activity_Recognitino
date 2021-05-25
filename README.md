# TensorFlow for Human Activity Recognition with LSTMs

Human activity recognition using smartphone sensors like accelerometer is one of the hectic topics of research. HAR is one of the time series classification problem. In this project various machine learning and deep learning models have been worked out to get the best final result. In the same sequence, we can use LSTM (long short term memory) model of the Recurrent Neural Network (RNN) to recognize various activities of humans like standing, climbing upstairs and downstairs etc.

LSTM model is a type of recurrent neural network capable of learning order dependence in sequence prediction problems. This model is used as this helps in remembering values over arbitrary intervals.

Human Activity Recognition dataset can be downloaded from the repo itself.

## Activities:

Walking

Upstairs

Downstairs

Sitting

Standing



## Choosing a dataset
Uploading the dataset in the drive to work on google colaboratory
Dataset cleaning and data Preprocessing
Choosing a model and building deep learned network model


## Data PreProcessing:
It begins with the data pre-processing. It is the phase where ~90% of time is consumed in actual data science projects. Here, raw data is taken and converted in some useful and efficient formats.

 ## Model building

     start = time.time()
      model = LogisticRegression(C=0.1)
      print(model)
      model.fit(trainData,trainLabelE)
      LogisticReg_Score=model.score(testData,testLabelE)
      print(LogisticReg_Score)  # 0.95
      end = time.time()
      total_time =end-start
      print('Time of Logistic Regression %.4f' % total_time)
      
   ## Performing several iterations of model training to get the highest accuracy and reduced loss
    mlpSGD  =  nn.MLPClassifier(hidden_layer_sizes=(90,) \
                        , max_iter=1000, alpha=1e-4  \
                        , solver='sgd' ,verbose=10   \
                        , tol=1e-19    , random_state =1 \
                        , learning_rate_init=.001)

     mlpADAM  =  nn.MLPClassifier(hidden_layer_sizes=(90,) \
                        , max_iter=1000, alpha=1e-4  \
                        , solver='adam' ,verbose=10   \
                        , tol=1e-19    , random_state =1 \
                        , learning_rate_init=.001)

     nnModelADAM = mlpADAM.fit(Scaled_trainData, trainLabelE)

     predicted = nnModelADAM.predict(Scaled_testData)
     matrix = confusion_matrix(testLabelE, predicted)
     print(matrix)
     print(nnModelADAM.score(Scaled_testData,testLabelE))
     
   # Keras Neural Network
    def create_model():
    model = Sequential()
    model.add(Dense(n_hidden_units,input_dim=n_input,activation="relu"))
    model.add(Dense(n_hidden_units,input_dim=n_input,activation="relu"))
    model.add(Dense(n_output,activation="softmax"))
    # Compile Model
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
    return model

    estimator = KerasClassifier(build_fn=create_model, epochs=20, batch_size=10, verbose=False)
    estimator.fit(Scaled_trainData,Y_train)
    print("Keras Classifier Score:{}".format(estimator.score(Scaled_testData,Y_test))) # 0.95
    
   ###### Result 
     Keras Classifier Score:0.966406524181366
 
      
  ## DecisionTree Confusuion Matrix
   ![image](https://user-images.githubusercontent.com/49801313/119567428-f0258900-bdc9-11eb-8ed9-2d9cd5b1a256.png)

## Installation Instructions

#### Make sure you have all the dependencies installed like:-

   Tensorflow(Latest version)if not using
    
    -pip install tensorflow
    
   Then Clone the Repo using
      
     -git clone https://github.com/techpkd123/Human_Activity_Recognitino.git
    
   Then you are good to go and can explore the project(i.e. notebook)
