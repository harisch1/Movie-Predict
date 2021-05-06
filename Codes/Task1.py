import numpy as np
import preprocess
import NeuralNetworks
import NeuralNetwork2
X,Y = preprocess.preProcess('Teleplay.csv')
Y = np.reshape(Y, (Y.shape[0],1))
X_pred= preprocess.preProcess('New_Teleplay.csv')
X_pred.resize(len(X_pred) ,  89)

#nn = NeuralNetworks.NeuralNetwork(X,Y)
#nn.train(epochs = 1000)
prediction = np.array(NeuralNetwork2.neural(X, Y, X_pred))
prediction.round(decimals = 2)
#nn.pred(X_pred)
np.savetxt("18086809D_Task1.csv", prediction, delimiter=",")