from sklearn.base import BaseEstimator
import numpy as np
import scipy.stats as stats

# For this assignment we will implement the Naive Bayes classifier as a
# a class, sklearn style. You only need to modify the fit and predict functions.
# Additionally, implement the Disparate Impact measure as the evaluateBias function.
class NBC(BaseEstimator):
    '''
    (a,b) - Beta prior parameters for the class random variable
    alpha - Symmetric Dirichlet parameter for the features
    '''

    def __init__(self, a=1, b=1, alpha=1):
        self.a = a
        self.b = b
        self.alpha = alpha
        self.__params = None
        
    def get_a(self):
        return self.a

    def get_b(self):
        return self.b

    def get_alpha(self):
        return self.alpha

    # you need to implement this function
    # nbTrain()
    def fit(self,X,y):
        '''
        This function does not return anything
        
        Inputs:
        X: Training data set (N x d numpy array)
        y: Labels (N length numpy array)
        '''
        a = self.get_a()
        b = self.get_b()
        alpha = self.get_alpha()
        self.__classes = np.unique(y)


        # compute the formulas of 5 and 6 ------ DONE
        n1 = 0
        n2 = 0
        for i in y:
            if i == 1:
                n1+=1 # count the number of n1
            if i == 2:
                n2+=1 # count the number of n2
        total = n1 + n2
        # the class probability when the Y == 1, refer to 5th formulas
        print("Y = 1 :" + str((n1 + a) / (total + a + b)))
        # the class probability when the Y == 1
        print("Y = 2 :" + str((n2 + a) / (total + a + b)))
        print("------------------------------------------------")
        
        # computes the formulas of 8 and 9 ------- NEEDS WORK

        # Return the count of repetitions of unique elements:
        # EXAMPLE:
        # First array: [5 2 6 2 7 5 6 8 2 9]
        # uniques:     [2 5 6 7 8 9]   <--Kj would be length of unique
        # count:       [3 2 2 1 1 1]

        XT = np.transpose(X)
        for j in XT: #iterating through each column of X instead of row since its transposed
            uniques,count = np.unique(j,return_counts = True) 
            print("Y = 1 : " + str((n1 + alpha) / (n1 + len(uniques) * b))
                   + "\nY = 2 :" + str((n2 + alpha) / (n2 + len(uniques) * b)))
            

        # for j in X:
        #     print("Y = 1 : " + str((n1 + alpha) / (n1 + j * b))
        #           + "\nY = 2 :" + str((n2 + alpha) / (n2 + j * b)))

        params = None
        self.__params = params
    
    # you need to implement this function
    # nbPredict
    def predict(self,Xtest):
        '''
        This function returns the predicted class for a given data set
        
        Inputs:
        Xtest: Testing data set (N x d numpy array)
        
        Output:
        predictions: N length numpy array containing the predictions
        '''
        params = self.__params
        a = self.get_a()
        b = self.get_b()
        alpha = self.get_alpha()

        #remove next line and implement from here
        predictions = np.random.choice(self.__classes,np.unique(Xtest.shape[0]))
        #do not change the line below
        return predictions
        
def evaluateBias(y_pred,y_sensitive):
    '''
    This function computes the Disparate Impact in the classification predictions (y_pred),
    with respect to a sensitive feature (y_sensitive).
    
    Inputs:
    y_pred: N length numpy array
    y_sensitive: N length numpy array
    
    Output:
    di (disparateimpact): scalar value
    '''
    #remove next line and implement from here
    di = 0
    
    #do not change the line below
    return di

def genBiasedSample(X,y,s,p,nsamples=1000):
    '''
    Oversamples instances belonging to the sensitive feature value (s != 1)
    
    Inputs:
    X - Data
    y - labels
    s - sensitive attribute
    p - probability of sampling unprivileged customer
    nsamples - size of the resulting data set (2*nsamples)
    
    Output:
    X_sample,y_sample,s_sample
    '''
    i1 = y == 1 # good
    i1 = i1[:,np.newaxis]
    i2 = y == 2 # bad
    i2 = i2[:,np.newaxis]
    
    sp = s == 1 #privileged
    sp = sp[:,np.newaxis]
    su = s != 1 #unprivileged
    su = su[:,np.newaxis]

    su1 = np.where(np.all(np.hstack([su,i1]),axis=1))[0]
    su2 = np.where(np.all(np.hstack([su,i2]),axis=1))[0]
    sp1 = np.where(np.all(np.hstack([sp,i1]),axis=1))[0]
    sp2 = np.where(np.all(np.hstack([sp,i2]),axis=1))[0]
    inds = []
    for i in range(nsamples):
        u = stats.bernoulli(p).rvs(1)
        if u == 1:
            #sample one bad instance with s != 1
            inds.append(np.random.choice(su2,1)[0])
            #sample one good instance with s == 1
            inds.append(np.random.choice(sp1,1)[0])
        else:
            #sample one good instance with s != 1
            inds.append(np.random.choice(su1,1)[0])
            #sample one bad instance with s == 1
            inds.append(np.random.choice(sp2,1)[0])
    X_sample = X[inds,:]
    y_sample = y[inds]
    s_sample = s[inds]
    
    return X_sample,y_sample,s_sample,inds
