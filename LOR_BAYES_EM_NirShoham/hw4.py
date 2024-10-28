import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def pearson_correlation(x, y):
    """
    Calculate the Pearson correlation coefficient for two given columns of data.

    Inputs:
    - x: An array containing a column of m numeric values.
    - y: An array containing a column of m numeric values. 

    Returns:
    - The Pearson correlation coefficient between the two columns.    
    """
    r = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    x_expectation = np.mean(x)
    y_expectation = np.mean(y)

    x_delta = x - x_expectation
    y_delta = y - y_expectation

    covariance_XY = np.sum(x_delta * y_delta)

    stdXMultStdY = np.sqrt(np.sum((x_delta)**2) * np.sum((y_delta)**2))

    r = covariance_XY/ stdXMultStdY

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return r

def feature_selection(X, y, n_features=5):
    """
    Select the best features using pearson correlation.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - best_features: list of best features (names - list of strings).  
    """
    best_features = [] 
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # Drop non-numeric columns
    X_numeric = X.select_dtypes(include=[np.number])    
    feature_names = X_numeric.columns
    
    # Calculate the Pearson correlation for each feature with y
    pearson_correlation_list = np.array([pearson_correlation(X[feature], y) for feature in feature_names])

    # Get the indices of the features with the highest absolute correlation values
    best_features_with_data= np.argsort(-np.abs(pearson_correlation_list))[:n_features]

    best_features = feature_names[best_features_with_data].tolist()
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return best_features


class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)

        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        XWithBias = np.insert(X,0,1,axis=1)
        self.theta = np.random.random(XWithBias.shape[1])
        self.thetas.append(self.theta)

        for i in range(self.n_iter):
          exponent = (-1) *np.dot(XWithBias, self.theta)
          hFunc = 1.0 / (1.0+ np.exp(exponent))
          errors = hFunc - y
          gradient = np.dot(XWithBias.T, errors)

          self.theta -= self.eta * gradient

          clipped_hFunc = np.clip(hFunc, 1e-10, 1 - 1e-10)
          cost = (-1.0 / len(y)) * (np.dot(y.T, np.log(clipped_hFunc)) + np.dot(1 - y, np.log(1 - clipped_hFunc)))
          self.Js.append(cost)
          self.thetas.append(self.theta)
          if len(self.Js) > 1 and abs(self.Js[i - 1] - self.Js[i]) < self.eps:
            break
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        XWithBias = np.insert(X,0,1,axis=1)
        val = np.dot(XWithBias,self.theta)
        sigmoid = 1.0/(1.0 + np.exp(-1.0 * val))
        preds = np.round(sigmoid).astype(int)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = 0.0

    # set random seed
    np.random.seed(random_state)

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    shuffled_indices = np.arange(X.shape[0])
    np.random.shuffle(shuffled_indices)
    X = X[shuffled_indices]
    y = y[shuffled_indices]
    
    group_size = X.shape[0] // folds
    accuracies = []


    for k in range(folds):
      start_index = k * group_size
      if k == folds - 1:  # Handle the last fold potentially having more elements
          end_index = len(X)
      else:
          end_index = start_index + group_size

      # Split data into training and testing sets
      X_test = X[start_index:end_index]
      y_test = y[start_index:end_index]
      X_training = np.concatenate((X[:start_index], X[end_index:]), axis=0)
      y_training = np.concatenate((y[:start_index], y[end_index:]), axis=0)

      algo.fit(X_training,y_training)

      predictions = algo.predict(X_test)

      accuracy = np.mean(predictions == y_test)
      accuracies.append(accuracy)


    cv_accuracy = np.mean(accuracies)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return cv_accuracy

def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    p = (1/(sigma* np.sqrt(2* np.pi))) * np.exp(-0.5 * (( data - mu ) / sigma) ** 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        self.weights = np.ones(self.k) / self.k
        n_samples = data.shape[0]
        indexes = np.random.choice(n_samples, self.k, replace=False)
        self.mus = data[indexes]
        self.sigmas = np.random.rand(self.k) + 0.01
        self.costs = []

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        N = data.shape[0]
        self.responsibilities = np.zeros((N, self.k))

        for sample_number in range(N):
            denominator = 0
            for gaus_num in range(self.k):
              dens = norm_pdf(data[sample_number], self.mus[gaus_num], self.sigmas[gaus_num]) * self.weights[gaus_num] #check the prob of the instance to be on this gaus
              denominator += dens
              self.responsibilities[sample_number,gaus_num] = dens
           
            self.responsibilities[sample_number, :] /= denominator

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def maximization(self, data):
        """ 
        M step - This function should calculate and update the distribution params
        """
        self.weights = np.mean(self.responsibilities, axis=0)
        self.mus = np.sum(self.responsibilities * data, axis=0) / np.sum(self.responsibilities, axis=0)
        var = np.mean(self.responsibilities * np.square(data - self.mus), axis=0)
        self.sigmas = np.sqrt(var / self.weights)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        self.init_params(data)  # Initialize parameters
        self.costs = []  # Store costs for convergence check
        prev_cost = None

        for i in range(self.n_iter):
            self.expectation(data)  # E-step
            self.maximization(data)  # M-step
            cost = self.calculate_cost(data)  # Calculate current cost
            self.costs.append(cost)  # Store current cost

            # Check for convergence
            if i > 0 and abs(cost - prev_cost) < self.eps:
                break
            prev_cost = cost
            
    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas
    
    def calculate_cost(self, data):
      """
      Calculate the negative log likelihood cost function.
      """
      N = data.shape[0]
      cost = 0.0
      for i in range(N):
          pdf_sum = 0.0
          for k in range(self.k):
              pdf_sum += self.weights[k] * norm_pdf(data[i], self.mus[k], self.sigmas[k])
          cost -= np.log(pdf_sum)
      return cost

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pdf = np.sum(weights * norm_pdf(data.reshape(-1,1) , mus , sigmas), axis=1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None
        self.labels = []
        self.gaussian = {}

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        self.prior = {}
        self.labels = np.unique(y)
        for label in self.labels:
            self.gaussian[label] = []
            self.prior[label] = np.mean(y == label)
            label_data = X[np.where(y == label)]
            for feature_num in range(label_data.shape[1]):
                feature = label_data[:, feature_num].reshape(-1, 1)
                em = EM(k=self.k, random_state=self.random_state)
                em.fit(feature)
                self.gaussian[label].append(em)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        preds = []
        for instance in X:
            max_prob = -float('inf')
            pred_label = None
            for label in self.labels:
                likelihood = 1
                for feature_num, em in enumerate(self.gaussian[label]):
                    # Retrieve parameters from EM object
                    weights, mus, sigmas = em.get_dist_params()
                    # Calculate the probability density for each feature of the instance
                    likelihood *= gmm_pdf(instance[feature_num], weights, mus, sigmas)
                prob = self.prior[label] * likelihood
                if prob > max_prob:
                    max_prob = prob
                    pred_label = label
            preds.append(pred_label)
        return preds
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

# Function for ploting the decision boundaries of a model
def plot_decision_regions(X, y, classifier, resolution=0.01, title=""):
  # setup marker generator and color map
  markers = ('.', '.')
  colors = ['blue', 'red']
  cmap = ListedColormap(colors[:len(np.unique(y))])
  # plot the decision surface
  x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                        np.arange(x2_min, x2_max, resolution))
  Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
  Z = np.array(Z).reshape(xx1.shape)
  # Z = Z.reshape(xx1.shape)
  plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
  plt.xlim(xx1.min(), xx1.max())
  plt.ylim(xx2.min(), xx2.max())

  for idx, cl in enumerate(np.unique(y)):
      plt.title(title)
      plt.scatter(x=X[y == cl, 0], 
                  y=X[y == cl, 1],
                  alpha=0.8, 
                  c=colors[idx],
                  marker=markers[idx], 
                  label=cl, 
                  edgecolor='black')
  plt.show()

def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    lor_model = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    lor_model.fit(x_train, y_train)
    prediction_lor_train = lor_model.predict(x_train)
    prediction_lor_test= lor_model.predict(x_test)
    lor_train_acc = np.mean(y_train == prediction_lor_train)
    lor_test_acc = np.mean(y_test == prediction_lor_test)
    plt.figure()
    plot_decision_regions(x_train, y_train, classifier=lor_model, title="Logistic Regression Decision Regressor")
    # Naive Bayes with EM
    bayes_model = NaiveBayesGaussian(k=k)
    bayes_model.fit(x_train, y_train)
    bayes_train_predictions = bayes_model.predict(x_train)
    bayes_test_predictions = bayes_model.predict(x_test)
    bayes_train_acc = np.count_nonzero(bayes_train_predictions == y_train) / len(y_train  )
    bayes_test_acc = np.count_nonzero(bayes_test_predictions == y_test) / len(y_test)
    print("Lor train acc = " + str(lor_train_acc) + "\n")
    print("Lor test acc = " + str(lor_test_acc) + "\n")
    print("bayes train acc = " + str(bayes_train_acc) + "\n")
    print("bayes test acc = " + str(bayes_test_acc) + "\n")
    plt.figure()
    plot_decision_regions(x_train, y_train, classifier=bayes_model, title="Naive Bayes Decision Regressor")
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(lor_model.Js)), lor_model.Js)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost vs Iteration for Logistic Regression')
    plt.grid(True)
    plt.show()
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

def   generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    ###########################################################################
    np.random.seed(42)

    # Dataset A (Naive Bayes performs better)
    mus_a_class00 = np.array([5, 5, 5])
    sigmas_a_class00 = np.diag([1, 1, 1])
    mus_a_class01 = np.array([-5, -5, -5])
    sigmas_a_class01 = np.diag([1, 1, 1])

    samples1, samples2 = multivariate_normal.rvs(mean=mus_a_class00, cov=sigmas_a_class00, size=300), multivariate_normal.rvs(mean=mus_a_class01, cov=sigmas_a_class01, size=300)
    data_a_class0 = np.vstack([samples1, samples2])
    labels_a_class0 = np.zeros(600)

    mus_a_class1 = np.array([0, 0, 0])
    sigmas_a_class1 = np.diag([1, 1, 1])

    data_a_class1 = multivariate_normal.rvs(mean=mus_a_class1, cov=sigmas_a_class1, size=400)
    labels_a_class1 = np.ones(400)

    dataset_a_features = np.concatenate((data_a_class0, data_a_class1), axis=0)
    dataset_a_labels = np.concatenate((labels_a_class0, labels_a_class1), axis=0)

    # Dataset B (Logistic Regression performs better)
    mus_b_class0 = np.array([0, 0, 0])
    sigmas_b_class0 = np.array([[2.0, 0.1, 0.1], [0.1, 2.0, 0.1], [0.1, 0.1, 2.0]])
    data_b_class0 = multivariate_normal.rvs(mean=mus_b_class0, cov=sigmas_b_class0, size=500)
    labels_b_class0 = np.zeros(500)

    mus_b_class1 = np.array([1, 1, 0])
    sigmas_b_class1 = np.array([[2.0, 0.2, 0.3], [0.2, 2.0, 0.4], [0.3, 0.4, 2.0]])

    data_b_class1 = multivariate_normal.rvs(mean=mus_b_class1, cov=sigmas_b_class1, size=500)
    labels_b_class1 = np.ones(500)

    dataset_b_features = np.concatenate((data_b_class0, data_b_class1), axis=0)
    dataset_b_labels = np.concatenate((labels_b_class0, labels_b_class1), axis=0)
        ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }

def visualize_datasets(dataset_features, dataset_labels):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dataset_features[:, 0], dataset_features[:, 1], dataset_features[:, 2], c=dataset_labels, cmap='viridis')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    plt.show()

# Generate datasets
datasets = generate_datasets()

# Plot dataset B in 3D
visualize_datasets(datasets['dataset_a_features'], datasets['dataset_a_labels'])

visualize_datasets(datasets['dataset_b_features'], datasets['dataset_b_labels'])


