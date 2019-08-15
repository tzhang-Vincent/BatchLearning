import numpy as np

# Generate train data dimension n=4 m=100
num_sample = 100
a = 10 * np.random.randn(num_sample, 1)
x1 = a + np.random.randn(num_sample, 1)
x2 = 1 * np.sin(a) + np.random.randn(num_sample, 1)
x3 = 5 * np.cos(5 * a) + np.random.randn(num_sample, 1)
x4 = 0.8 * x2 + 0.1 * x3 + np.random.randn(num_sample, 1)
x = np.hstack((x1, x2, x3, x4))


# Get Train data
def Get_train_datum(i, x):
    return x[i]



class ipca:
    def __init__(self, pc_num, dimemsion_of_datum,amnesic=2.0):
        self.pc_num = pc_num
        self.col = dimemsion_of_datum
        self.v = np.ones((self.pc_num, self.col))
        self.current_train=0  #The number of datum have trained
        self.amnesic = amnesic #Quantity in Incremental pca,determined by current_train
        self.mean = np.zeros((self.col))

    #fit one datum into eigenvectors
    def fit_partial(self,u,n):
        n=float(n)
        if n <= int(self.amnesic):
            w1 = float(n+1) / float(n + 2)
            w2 = float(1) / float(n + 2)
        else:
            w1 = float(n +2- self.amnesic) / float(n + 2)
            w2 = float(1 + self.amnesic) / float(n + 2)

        self.mean = w1 * self.mean + w2 * u
        u = u - self.mean

        for i in range(0, min(self.pc_num, int(n+1))):
            if i == n:
                self.v[i] = u
            else:
                self.v[i] = w1 * self.v[i] + w2 * np.dot(u, np.transpose(u)) * self.v[i] / np.linalg.norm(
                        self.v[i])
                u = u - np.dot(np.dot(np.transpose(u), self.v[i]),self.v[i] )/ (
                                np.linalg.norm(self.v[i]) * np.linalg.norm(self.v[i]))
        self.current_train += 1
        return

    #Train some specific number of datums
    def ipca_train(self,num_of_train):
        for n in range(0, num_of_train):
            train_datum = Get_train_datum(n, x)
            self.fit_partial(train_datum,n)


            self.explained_variance_ratio_ = np.sqrt(np.sum(self.v ** 2,
                                                            axis=1))  # `explained_variance_ratio_` : array, [n_components]. Percentage of variance explained by each of the selected components.
            # sort by explained_variance_ratio_
            idx = np.argsort(-self.explained_variance_ratio_)

            self.explained_variance_ratio_ = self.explained_variance_ratio_[idx]
            self.v = self.v[idx]
            # re-normalize
            self.explained_variance_ratio_ = (self.explained_variance_ratio_ / self.explained_variance_ratio_.sum())
            for r in range(0, self.pc_num):
                self.v[r] /= np.linalg.norm(self.v[r])


    #Q detection
    def ipca_detection(self,test_datum):
        test_datum=test_datum-self.mean
        I = np.eye(self.col, self.col)
        pp=np.dot(np.transpose(self.v),self.v)
        Q=np.dot(np.dot(test_datum,(I-pp)),np.transpose(test_datum))
        return Q


iiii=ipca(3,4)
iiii.ipca_train(100)




#test
a = 10*np.random.randn(num_sample,1)
x1 = a+np.random.randn(num_sample,1)
x2 = 1*np.sin(a)+np.random.randn(num_sample,1)
x3 = 5*np.cos(5*a)+np.random.randn(num_sample,1)
x4 = 0.8*x2+0.1*x3+np.random.randn(num_sample,1)
xx_test = np.hstack((x1,x2,x3,x4))
xx_test[50:,1] = xx_test[50:,1]+15*np.ones(50)


for i in range(0,20):
    print('t',iiii.ipca_detection(xx_test[i]))


