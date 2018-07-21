import numpy as np
import random
import math


class LogisticRegression(object):

    def __init__(self):
        self.w = None
        self.one_vs_all_w = None

    # sigmoid函数，预测 h(z)
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def loss(self, X_batch, y_batch):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
        data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """

        #########################################################################
        # TODO:                                                                 #
        # calculate the loss and the derivative                                 #
        #########################################################################
        m = X_batch.shape[0]
        z = X_batch.dot(self.w)
        # log log10  log2  log1p
        # ln  底数10  底数2  log（x+1）
        J = (-y_batch.T.dot(np.log(self.sigmoid(z))) - (1 - y_batch.T).dot(np.log(1 - self.sigmoid(z)))) / m
        grad = np.dot(X_batch.T, self.sigmoid(z) - y_batch) / m
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################
        return J, grad

    def train(self, X, y, learning_rate=1e-3, num_iters=100,
              batch_size=200, verbose=True):

        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
         training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels;
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape
        # 初始化参数向量 w
        if self.w is None:
            self.w = 0.001 * np.random.randn(dim)
        # 记录loss的列表
        loss_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None
            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            index = np.random.choice(num_train, size=batch_size, replace=verbose)
            X_batch = X[index]
            y_batch = y[index]
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch)
            loss_history.append(loss)

            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            self.w = self.w - learning_rate * grad
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: N x D array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
        array of length N, and each element is an integer giving the predicted
        class.
        """
        # 这里原来的写错了，原为X.shape[1]，应该如下所示
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        h = self.sigmoid(X.dot(self.w))
        for i in range(X.shape[0]):
            if h[i] > 0.5:
                y_pred[i] = 1
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def loss_one_vs_all(self, X_batch, y_batch, i):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
        data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        m = X_batch.shape[0]
        z = X_batch.dot(self.one_vs_all_w[i])
        J = (-y_batch.T.dot(np.log(self.sigmoid(z))) - (1 - y_batch.T).dot(np.log(1 - self.sigmoid(z)))) / m
        grad = np.dot(X_batch.T, self.sigmoid(z) - y_batch) / m
        return J, grad

    def predict_one_vs_all(self, X):
        h_one_vs_all = self.sigmoid(X.dot(self.one_vs_all_w.T))
        y_pred = np.argmax(h_one_vs_all, axis=1)
        return y_pred

    def one_vs_all(self, X, y, learning_rate=1e-3, num_iters=100,
                   batch_size=200, verbose=True):
        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
         training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels;
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        """
        num_train, dim = X.shape
        # 初始化one_vs_all
        if self.one_vs_all_w is None:
            self.one_vs_all_w = 0.001 * np.random.randn(10, dim)

        for class_i in range(10):
            for it in range(num_iters):
                X_batch = None
                y_batch = None
                index = np.random.choice(num_train, size=batch_size, replace=verbose)
                X_batch = X[index]
                y_batch = y[index]
                lables = np.ones(batch_size)
                for i in range(batch_size):
                    if y_batch[i] != class_i:
                        lables[i] = 0
                # evaluate loss and gradient
                loss, grad = self.loss_one_vs_all(X_batch, lables, class_i)
                # perform parameter update
                self.one_vs_all_w[class_i] = self.one_vs_all_w[class_i] - learning_rate * grad
                if verbose and it % 300 == 0:
                    print('class : %d | iteration %d / %d: loss %f' % (class_i, it, num_iters, loss))
