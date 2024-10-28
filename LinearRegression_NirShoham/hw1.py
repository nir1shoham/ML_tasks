###### Your ID ######
# ID1: 322657073
# ID2: 211741921
#####################

# imports 
import numpy as np
import pandas as pd


def preprocess(X, y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    ###########################################################################
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X_mean = np.mean(X, axis=0)
    y_min = np.min(y, axis=0)
    y_max = np.max(y, axis=0)
    y_mean = np.mean(y, axis=0)

    X = (X-X_mean)/(X_max-X_min)
    y = (y-y_mean)/(y_max-y_min)
    ###########################################################################
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X, y


def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    ###########################################################################
    ###########################################################################
    ones = np.ones(len(X),)
    X = np.c_[ones, X]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X


def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    
    J = 0  # We use J for the cost.
    ###########################################################################
    ###########################################################################
    model = X.dot(theta)
    J = np.sum((model-y)**2) / (2 * len(X))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy()  # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the cost value in every iteration
    ###########################################################################
    ###########################################################################
    m = len(X)
    for i in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y

        gradient = np.dot(X.T, errors) / m

        theta -= alpha * gradient

        cost = compute_cost(X, y, theta)

        J_history.append(cost)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history


def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []
    ###########################################################################
    ###########################################################################
    pinv = np.linalg.inv(np.dot(X.T, X)).dot(X.T)
    pinv_theta = pinv.dot(y)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pinv_theta


def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy()  # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the cost value in every iteration
    ###########################################################################
    ###########################################################################
    m = len(X)
    for i in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = np.dot(X.T, errors) / m

        theta -= alpha * gradient

        cost = compute_cost(X, y, theta)
        J_history.append(cost)
        if i > 0 and (J_history[i-1] - J_history[i] < 1e-8):
            break
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history


def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}  # {alpha_value: validation_loss}
    ###########################################################################
    ###########################################################################
    for alpha in alphas:
        np.random.seed(42)
        length = X_train.shape[1]
        init_theta = np.random.random(size=length)
        theta = efficient_gradient_descent(X_train, y_train, init_theta, alpha, iterations)[0]
        alpha_dict[alpha] = compute_cost(X_val, y_val, theta)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return alpha_dict


def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    ###########################################################################
    ###########################################################################

    for i in range(5):
        feature_to_add = None
        init_theta = np.random.random(size=i + 2)
        np.random.seed(42)
        minimal_lost = float("inf")
        for feature in range(X_train.shape[1]):
            if feature not in selected_features:

                test_selected_features = selected_features + [feature]
                X_train_bias = apply_bias_trick(X_train[:, test_selected_features])
                X_val_bias = apply_bias_trick(X_val[:, test_selected_features])

                theta, cost = efficient_gradient_descent(X_train_bias, y_train, init_theta, best_alpha, iterations)
                cost_val = compute_cost(X_val_bias, y_val, theta)

                if cost_val < minimal_lost:
                    minimal_lost = cost_val
                    feature_to_add = feature

        selected_features.append(feature_to_add)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return selected_features


def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    ###########################################################################
    ###########################################################################

    # Iterate over each pair of features
    new_columns = []  # To store new columns

    # Iterate over each pair of features
    for col1 in df.columns:
        col1_loc = df.columns.get_loc(col1)  # Correctly getting the index of col1
        for col2 in df.columns[col1_loc:]:  # Iterate from col1 to the end of columns
            if col1 == col2:
                new_col = df_poly[col1] ** 2
                name = "{}^2".format(col1)
                new_col.name = name
                new_columns.append(new_col)
            else:
                new_col = df_poly[col1] * df_poly[col2]
                name = "{}*{}".format(col1, col2)
                new_col.name = name
                new_columns.append(new_col)

    # Concatenate new columns to the original dataframe
    df_poly = pd.concat([df_poly] + new_columns, axis=1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return df_poly
