Skip to content
 

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

        # remove next line and implement from here
        #total number of applicants
        N = X.shape[0]

        #number of features
        D = X.shape[1]

        #number of applicants w good credit
        N1 = 0

        #number of applicants w bad credit
        N2 = 0

        #matrix of Y==1
        Y1 = np.zeros((1,18))

        #matrix of Y ==2
        Y2 = np.zeros((1,18))

        #dividing the number of applicants w good or bad credit
        for i in range(N):
            if y[i] == 1:
                N1 = N1+1
                temp = X[i]
                Y1 = np.vstack((Y1, temp))
            else:
                N2 = N2 +1
                temp = X[i]
                Y2 = np.vstack((Y2, temp))

        #removing the first row
        Y1 = np.delete(Y1,0,axis=0)
        Y2 = np.delete(Y2,0,axis=0)

        #formula 5 & 6
        theta1 = (N1 + a)/(N + a + b)
        theta2 = 1 - theta1

        theta1_list = {
            "0": {},
            "1": {},
            "2": {},
            "3": {},
            "4": {},
            "5": {},
            "6": {},
            "7": {},
            "8": {},
            "9": {},
            "10": {},
            "11": {},
            "12": {},
            "13": {},
            "14": {},
            "15": {},
            "16": {},
            "17": {},
        }

        # theta1_list = np.zeros(18)
        # theta2_list = np.zeros(18)
        theta2_list = {
            "0": {},
            "1": {},
            "2": {},
            "3": {},
            "4": {},
            "5": {},
            "6": {},
            "7": {},
            "8": {},
            "9": {},
            "10": {},
            "11": {},
            "12": {},
            "13": {},
            "14": {},
            "15": {},
            "16": {},
            "17": {},
        }

        #finding the unique values for each feature and computing probability
        for i in range(18):
            temp = Y1[i]
            temp2 = Y2[i]
            # temp = Y1[i]
            # temp2 = Y2[i]
            Kj_2 = len(np.unique(temp2))
            Kj = len(np.unique(temp))

            for xj in range(Kj):
                N1_j = len([temp[i] for temp in Y1 if temp[i] == xj])
                theta1_j = (N1_j + alpha) / (N1 + (Kj * alpha))
                theta1_list[str(i)][str(xj)] = theta1_j

            for xj_2 in range(Kj_2):
                N2_j = len([temp2[i] for temp2 in Y2 if temp2[i] == xj_2])
                theta2_j = (N2_j + alpha) / (N2 + (Kj_2 * alpha))
                theta2_list[str(i)][str(xj_2)] = theta2_j


        #print(theta1_list["0"])

        params = [theta1,theta2,theta1_list,theta2_list,N1,N2]
        # do not change the line below
        self.__params = params
    
    # you need to implement this function
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

        theta1 = params[0]
        theta2 = params[1]
        theta1_list = params[2]
        theta2_list = params[3]
        N1 = params[4]
        N2 = params[5]
        #remove next line and implement from here
        #predictions = np.random.choice(self.__classes,np.unique(Xtest.shape[0]))

        N = Xtest.shape[0]
        predictions = np.zeros(N)
        prediction_class1 = np.zeros(N)
        prediction_class2 = np.zeros(N)
        #numerator_product1 = 1
        denominator1 = 0
        denominator2 = 0

        for i in range(N):
            temp = Xtest[i]
            numerator_product1 = theta1
            for j in range(len(temp)):
                current_feature = str(j)
                observed_feature = str(temp[j])
                temp_feature_values = theta1_list.get(current_feature)
                Kj = len(temp_feature_values)

                if observed_feature in temp_feature_values:
                    probability = float(temp_feature_values[observed_feature])
                    numerator_product1 = numerator_product1 * probability
                else:
                    temp_prob = alpha/(N1 + Kj*alpha)
                    numerator_product1 = numerator_product1 * temp_prob

            prediction_class1[i] = numerator_product1
            denominator1 = numerator_product1

            #numerator1 = theta1 * numerator_product1

            #numerator_product2 = 1
        for i in range(N):
            temp = Xtest[i]
            numerator_product2 = theta2

            for j in range(len(temp)):
                current_feature = str(j)
                observed_feature = str(temp[j])
                temp_feature_values = theta2_list.get(current_feature)
                Kj = len(temp_feature_values)

                if observed_feature in temp_feature_values:
                    probability = float(temp_feature_values[observed_feature])
                    numerator_product2 = numerator_product2 * probability
                else:
                    temp_prob = alpha / (N2 + Kj * alpha)
                    numerator_product2 = numerator_product2 * temp_prob

            prediction_class2[i] = numerator_product2
            denominator2 = numerator_product2

        for i in range(N):
            temp1 = prediction_class1[i]
            temp2 = prediction_class2[i]
            temp1 = temp1/(denominator1*theta1+denominator2*theta2)
            temp2 = temp2/(denominator1*theta1 + denominator2*theta2)
            if temp1 < temp2:
                predictions[i] = 2
            else:
                predictions[i] = 1


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
    # remove next line and implement from here
    top = 0
    bottom = 0
    for i in range(0, len(y_pred)):
        if y_pred[i] == 2 and y_sensitive[i] != 1:
            top += 1
        if y_pred[i] == 2 and y_sensitive[i] == 1:
            bottom += 1
    di = top / bottom

    # do not change the line below
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
