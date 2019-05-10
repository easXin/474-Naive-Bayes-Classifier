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




        y1 = 0
        y2 = 0
        N = np.size(X,0)
        r = np.zeros((1,18))
        m = np.zeros((1,18))

        for i in y:
            if y[i] == 1:
                y1+=1
                s=X[i]
                r=np.vstack((r,s))
            else:
                y2 +=1
                n = X[i]
                m = np.vstack((m,n))
        r = np.delete(r,0,0)
        thetaOnejDict = {
            "feature 1": {},
            "feature 2": {},
            "feature 3": {},
            "feature 4": {},
            "feature 5": {},
            "feature 6": {},
            "feature 7": {},
            "feature 8": {},
            "feature 9": {},
            "feature 10": {},
            "feature 11": {},
            "feature 12": {},
            "feature 13": {},
            "feature 14": {},
            "feature 15": {},
            "feature 16": {},
            "feature 17": {},
            "feature 18": {},
        }

        for i in range(0,18):
            Kj = len(np.unique([row[i] for row in X]))
            for xj in range(1,Kj+1):
                NOnej = len([row[i] for row in r if row[i] ==xj])
                thetaOnej = (NOnej+alpha)/(y1+(Kj*alpha))
                thetaOnejDict["feature "+str(1+i)][str(xj)] = thetaOnej
        thetaTwojDict = {
            "feature 1": {},
            "feature 2": {},
            "feature 3": {},
            "feature 4": {},
            "feature 5": {},
            "feature 6": {},
            "feature 7": {},
            "feature 8": {},
            "feature 9": {},
            "feature 10": {},
            "feature 11": {},
            "feature 12": {},
            "feature 13": {},
            "feature 14": {},
            "feature 15": {},
            "feature 16": {},
            "feature 17": {},
            "feature 18": {},
        }
        for i in range(0,18):
            Kj = len(np.unique([row[i] for row in X]))
            for xj in range(1,Kj+1):
                NOnej = len([row[i] for row in r if row[i] ==xj])
                thetaTwoj = (NOnej+alpha)/(y2+(Kj*alpha))
                thetaTwojDict["feature "+str(1+i)][str(xj)] = thetaTwoj

        theta1 = (y1 +a)/((N)+a+b)
        theta2 = (y2 +a)/((N)+a+b)
        params = [theta1, theta2, thetaOnejDict, thetaTwojDict]
        self.__params = params



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
        product1 = 1
        for i in range(1, 19):#loop through all features 1-18
            for j in range(1, len(params[2]["feature " + str(i)]) + 1):
                product1 *= params[2]["feature " + str(i)][str(j)]
        product2 = 1
        for i in range(1, 19):#loop through all features 1-18
            for j in range(1, len(params[3]["feature " + str(i)]) + 1):
                product2 *= params[3]["feature " + str(i)][str(j)]

        print(product1)
        print("-----------------------------------")
        print(product2)
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
    top = 0
    bottom = 0
    for i in range(0,len(y_pred)):
        if y_pred[i] == 2 and y_sensitive[i] != 1:
            top += 1
        if y_pred[i] == 2 and y_sensitive[i] == 1:
            bottom += 1
    di = top/bottom

    
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
