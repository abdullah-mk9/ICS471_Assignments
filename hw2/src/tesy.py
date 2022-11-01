
import matplotlib.pyplot as plt
import numpy as np

# Package imports
import numpy as np
import sklearn
import sklearn.linear_model
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
# splits the data into train, val, test

def split_stratified_into_train_val_test(df_input, stratify_colname='y',
                                         frac_train=0.6, frac_val=0.15, frac_test=0.25,
                                         random_state=None):

    X = df_input.drop(columns=[stratify_colname])  # Contains all columns.
    # Dataframe of just the column on which to stratify.
    y = df_input[[stratify_colname]]

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(
                                                              1.0 - frac_train),
                                                          random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                      y_temp,
                                                      stratify=y_temp,
                                                      test_size=relative_frac_test,
                                                      random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test, y_train, y_val, y_test

np.random.seed(777)




import pandas as pd
np.random.seed(777)
df = pd.read_csv('hw2\src\data.csv')

scaler = preprocessing.StandardScaler().fit(df[df.columns[:-1]])
results = scaler.transform(df[df.columns[:-1]])
df[df.columns[:-1]] = results

print(df['bus'].unique())
df['bus'] = pd.factorize(df['bus'])[0]
df['bus']

X_train, X_val, X_test, y_train, y_val, y_test = split_stratified_into_train_val_test(df, stratify_colname='bus',
                                                                                      frac_train=0.7, frac_val=0.15, frac_test=0.15,
                                                                                      random_state=None)
X_train, X_val, X_test, y_train, y_val, y_test = X_train.T.to_numpy(), X_val.T.to_numpy(
), X_test.T.to_numpy(), y_train.T.to_numpy(), y_val.T.to_numpy(), y_test.T.to_numpy()
X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape

X, valx, testx, y,valy,testy = X_train.T, X_val.T, X_test.T, y_train.T, y_val.T, y_test.T

n_x = X.shape[1]
M = len(X_test)
n_h = 4

# M = 150 # number of points per class
# n_x = 18 # dimensionality
# n_h = 4 # number of classes
# X = np.zeros((M*n_h,n_x)) # data matrix (each row = single example)
# y = np.zeros(M*n_h, dtype='uint8') # class n_h

# print(X.shape)


#Train a Linear Classifier

# initialize parameters randomly
W = 0.01 * np.random.randn(n_x,n_h)
print(W.shape)
b = np.zeros((1,n_h))
print(b.shape)


# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength

# gradient descent loop
num_examples = X.shape[0]
print(num_examples)
for i in range(10000):

  # evaluate class scores, [M x n_h]
  scores = np.dot(X, W) + b

  # compute the class probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [M x n_h]

  # compute the loss: average cross-entropy loss and regularization
  correct_logprobs = -np.log(probs[range(num_examples),y])
  data_loss = np.sum(correct_logprobs)/num_examples
  reg_loss = 0.5*reg*np.sum(W*W)
  loss = data_loss + reg_loss
  if i % 10 == 0:
    print ("iteration %d: loss %f" % (i, loss))

  # compute the gradient on scores
  dscores = probs
  dscores[range(num_examples),y] -= 1
  dscores /= num_examples

  # backpropate the gradient to the parameters (W,b)
  dW = np.dot(X.T, dscores)
  db = np.sum(dscores, axis=0, keepdims=True)

  dW += reg*W # regularization gradient

  # perform a parameter update
  W += -step_size * dW
  b += -step_size * db
  
  
scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
print ('training accuracy: %.2f' % (np.mean(predicted_class == y)))