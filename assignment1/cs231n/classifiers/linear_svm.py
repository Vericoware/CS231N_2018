import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        ################ add ############################
        dW[:,y[i]] += -X[i,:].T #每个j,求偏导是loss/margin--margin/correct--(-X[i])
        dW[:,j] += X[i,:].T #每个j,求偏导是loss/margin--margin/correct--(X[i])
        #################################################

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
 ################# add ################################
  dW /= num_train #every has be calculate in j & margin > 0
 #################################################
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
 ################ add #############################
  dW += reg*W
 #################################################

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  num_train = X.shape[0]
  scores_cor = scores[list(range(num_train)),y]
  # also the list(range(num_train)) can be repalced by np.arange(num_train)
  scores_cor = np.reshape(scores_cor,(num_train,-1))
  margin = scores-scores_cor+1
  margin = np.maximum(0,margin)
  margin[list(range(num_train)),y] = 0
  loss = (np.sum(margin))/num_train
  loss += reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  margin[margin>0] = 1 # 大于0返回真值
  counter_r = np.sum(margin, axis=1)
  margin[list(range(num_train)),y] = -counter_r
  dW += np.dot(X.T,margin)/num_train +reg*W

  ## 在margin为1的情况，X矩阵的每行的图像x都要对应赋给W的每一列
  ## 而在正确的标签行要是负的margin每行大于1的个数。
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
