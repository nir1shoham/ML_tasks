import numpy as np

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.15, 
            (0, 1): 0.15,
            (1, 0): 0.2,
            (1, 1): 0.5
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.15,
            (0, 1): 0.15,
            (1, 0): 0.35,
            (1, 1): 0.35
        }  # P(X=x, C=y)

        self.Y_C = {
            (0, 0): 0.15,
            (0, 1): 0.15,
            (1, 0): 0.35,
            (1, 1): 0.35
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.045,
            (0, 0, 1): 0.045,
            (0, 1, 0): 0.105,
            (0, 1, 1): 0.105,
            (1, 0, 0): 0.105,
            (1, 0, 1): 0.105,
            (1, 1, 0): 0.245,
            (1, 1, 1): 0.245,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        for x, y in X_Y.keys():
            if not np.isclose(X[x] * Y[y], X_Y[(x, y)]):
                return True  # Return True immediately if any pair shows dependence
        return False  # Return False if all pairs are independent
    
    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        for x in self.X:
            for y in self.Y:
                for c in self.C:
                    # check if P(x,y|c) = P(x|c) * P(y|c). if one caculation is not equal, return false - dependent.
                    if not np.isclose((X_Y_C[(x, y, c)] / C[c]), (X_C[(x, c)] / C[c]) * (Y_C[(y, c)] / C[c])): 
                        return False
        return True

def poisson_log_pmf(k, rate): 
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    log_p = np.log(((rate ** k) * np.exp(-rate)) / np.math.factorial(k)) #log the caculation of poisson formula.
    return log_p


def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = np.zeros(len(rates))

    for i, rate in enumerate(rates): # build likelihoods table
        for k in samples:
            likelihoods[i] += poisson_log_pmf(k, rate)
    return likelihoods


def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """
    likelihoods = get_poisson_log_likelihoods(samples, rates)
    rate = rates[np.argmax(likelihoods)] # get the maximum likelihood 
    return rate

def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = np.mean(samples) # according to caculation, the mean of likelihoods.
    return mean

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    # Calculate the normal distribution pdf using the formula
    p = ((1 / np.sqrt(2 * np.pi * (std**2)))) * np.exp(-((x - mean)**2) / (2 * (std**2))) # by formula of nornal desnity function
    return p

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        self.dataset = dataset
        self.class_value = class_value
        self.class_data = dataset[dataset[:, -1] == class_value]  # Get data for this class
        self.mean = np.mean(self.class_data[:, :-1], axis=0)
        self.std = np.std(self.class_data[:, :-1], axis=0)
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = len(self.class_data) / len(self.dataset) 

        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood = 1.0

        for i in range(len(self.dataset.T) -1):
            likelihood *= normal_pdf(x[i], self.mean[i], self.std[i]) # according to naive , we multiply all probabilities.
        return likelihood
   
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_instance_likelihood(x) * self.get_prior() # by formula
        return posterior



class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods 
                     for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """

        posterior0 = self.ccd0.get_instance_posterior(x)
        posterior1 = self.ccd1.get_instance_posterior(x)
        
        return 0 if posterior0 > posterior1 else 1


def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.
    
    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """

    count_CorrectlyClassified = 0
    for instance in test_set:
        if map_classifier.predict(instance) == instance[-1]:
            count_CorrectlyClassified += 1 # if we predict right, add one to count

    acc = count_CorrectlyClassified / len(test_set)
    return acc

def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    d = len(mean)
    x_minus_mu = x[:-1] - mean
    cov_pinv = np.linalg.pinv(cov)
    
    # Calculate the normal distribution pdf using the formula
    coef = 1 / np.sqrt(((2 * np.pi) ** d) * np.linalg.det(cov))
    exponent = -0.5 * np.dot(np.dot(x_minus_mu.T, cov_pinv), x_minus_mu)
    pdf = coef * np.exp(exponent) # by formula we saw in class
    return pdf

    
class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        self.dataset = dataset
        self.class_value = class_value
        self.class_data = dataset[dataset[:, -1] == class_value]  # Get data for this class
        self.mean = np.mean(self.class_data[:, :-1], axis=0)
        self.cov = np.cov(self.class_data[:, :-1], rowvar=False)
        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = len(self.class_data) / len(self.dataset)
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        likelihood = multi_normal_pdf(x, self.mean, self.cov) # caculate likelihood using multi formula.
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        return posterior


class MaxPrior():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        
        return 0 if self.ccd0.get_prior() > self.ccd1.get_prior() else 1
        

class MaxLikelihood():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        
        return 0 if self.ccd0.get_instance_likelihood(x) > self.ccd1.get_instance_likelihood(x) else 1

EPSILLON = 1e-6 # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        self.dataset = dataset
        self.class_data = dataset[dataset[:, -1] == class_value]
        self.class_val = class_value
    
    def get_prior(self):
        """
        Returns the prior porbability of the class 
        according to the dataset distribution.
        """
        prior = len(self.class_data) / len(self.dataset)
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """
        likelihood = 1.0
        n_i = len(self.class_data)
    
        for j in range(len(x) - 1):  # Exclude the last element (class label)
            numberOfequalAttributes = np.sum(self.dataset[:,j] == x[j]) #check if this attribue of x' exists in dataset.
            if numberOfequalAttributes == 0:
                likelihood *= EPSILLON # if we didnt have this attribue in data, use epsillon for likelihood.
                continue

            # Count the number of instances with the class A_i and the value x_j in the relevant attribute
            n_ij = np.sum(self.class_data[:, j] == x[j])
        
            # Count the number of possible values of the relevant attribute
            V_j = len(np.unique(self.dataset[:, j]))
        
            # Calculate the likelihood for this feature value
            likelihood *= (n_ij + 1) / (n_i + V_j)


        return likelihood
        


    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance 
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        return posterior


class MAPClassifier_DNB():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        likelihood_c0 = self.ccd0.get_instance_posterior(x)
        likelihood_c1 = self.ccd1.get_instance_posterior(x)
        return 0 if likelihood_c0 > likelihood_c1 else 1

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        count_CorrectlyClassified = 0
        for instance in test_set:
            if self.predict(instance) == instance[-1]:
                count_CorrectlyClassified += 1
        acc = count_CorrectlyClassified / len(test_set)
        return acc

