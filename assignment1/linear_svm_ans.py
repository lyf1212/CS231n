from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    # dW是与W规格相同的记录每个位置上梯度大小的梯度矩阵.
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0 
    for i in range(num_train):
        scores = X[i].dot(W)
        # 注意x[i]是行向量而每一列W代表一个类，所以要用x[i]乘上W.
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                # 梯度只在margin>0时计算
                # 对于不正确的分类都是正号，而对正确的y[i]是负号.
                dW[:,j] += X[i].T
                dW[:,y[i]] -= X[i].T
                
                
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW /= num_train
    dW += 2*reg*W
    
    
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    num_train = X.shape[0]
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 先计算出所有评分，每一行是一个训练数据的C种类别的评分。之后利用切片和reshape方法
    # 求得一列真实分类的评分，然后可以利用广播机制复现损失函数.
    # 值得注意的是，reshape中-1指的是按照大小填充，不作约束.
    pred_score = X.dot(W)
    truth_score = pred_score[range(num_train), y].reshape(-1, 1)
    margin = np.maximum(pred_score - truth_score + 1, 0)
    # 原损失函数中只考虑错误分类的评分离谱程度，对于正确者不予考虑。这里要在计算后设置为0.
    margin[range(num_train), y] = 0
    loss = np.sum(margin) / num_train + reg*np.sum(W**2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 如果margin[i][j]不是0，那么W的第j列要被加上X[i].T的梯度，W的第y[i]列要被减去X[i].T的梯度.
    # 把margin[i][y[i]]都设置为这一行大于0的数目，最后要减去这么多X[i].T.
    # dW 的原型在X.T.dot(margin)上，因为原来X[i]是行向量.
    margin[margin>0] = 1
    margin[range(num_train), y] = -np.sum(margin, axis=1)
    dW = (X.T).dot(margin)
    dW = dW / num_train + 2*reg*W
    
    
    
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
