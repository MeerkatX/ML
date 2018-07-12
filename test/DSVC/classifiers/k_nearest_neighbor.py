import numpy as np


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just 
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the 
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                # 三种不同的方法求 欧氏距离
                # (平方，所以self.X_train[j]-X[i]与X[i]-self.X_train[j]）相同
                dists[i, j] = np.sqrt(np.sum(np.square(self.X_train[j] - X[i])))
                # 不利用np.square
                # dists[i,j]=np.sqrt(np.sum((self.X_train[j]-X[i])**2))
                # 利用矩阵点乘自身
                # dists[i,j]=np.sqrt(np.dot(self.X_train[j]-X[i],self.X_train[j]-X[i]))
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            # 先将测试矩阵的每一行扩展(复制)为特征数×训练数据的数量的矩阵
            # 之后减去训练矩阵
            diffMat = np.tile(X[i], (num_train, 1)) - self.X_train
            # 将训练矩阵平方
            sq = diffMat ** 2
            # 相加
            sqDis = sq.sum(axis=1)
            # 再开方，获得一个测试样本的欧氏距离
            dists[i, :] = sqDis ** 0.5
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        # 无参数表示全部相加，axis＝0表示按列相加，axis＝1表示按行的方向相加
        # 这里就是将矩阵压缩成一维，num_train列的矩阵，得到a^2的累加
        dists += np.sum(self.X_train ** 2, axis=1).reshape(1, num_train)
        # 同理得到b^2的累加
        dists += np.sum(X ** 2, axis=1).reshape(num_test, 1)  # reshape for broadcasting
        # 这里减去2ab的累加
        dists -= 2 * np.dot(X, self.X_train.T)  # .T表示转置
        # 开方后得到结果
        dists = np.sqrt(dists)
        # 原理是（a-b）^2=a^2+b^2-2ab
        return dists

        '''
        >>> import numpy as np
        >>> a=np.sum([[0,1,2],[2,1,3]])
        >>> a
        9
        >>> a.shape()
        >>> a=np.sum([[0,1,2],[2,1,3]],axis=0)
        >>> a
        array([2, 2, 5])
        >>> a.shape
        (3,)
        >>> a=np.sum([[0,1,2],[2,1,3]],axis=1)
        >>> a
        array([3, 6])
        >>> a.shape
        (2,)
        '''

    def predict_labels(self, dists, k):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            # 通过np.argsort进行排序，返回的是顺序的列表
            order = np.argsort(dists[i])
            # 列表切片，获取前K个
            order = order[0:k]
            # 找到对应的Y(lable)
            for j in order:
                closest_y.append(self.y_train[j])
            # print(closest_y)
            # 定义一个全零的list刚好与类别数相同(10)
            l = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for o in closest_y:
                l[o] = l[o] + 1  # 是那个类，那个类加一
            y_pred[i] = np.argsort(l)[9]  # 排序得到最大的数的位置即9
            # print(y_pred[i])
        return y_pred
