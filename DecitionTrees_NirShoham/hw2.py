import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    ###########################################################################                                         #
    ###########################################################################
    column = data[:, -1]

    # Count unique values
    values, counts = np.unique(column, return_counts=True)
    num_rows = data.shape[0]
    calc = np.sum((counts/num_rows)**2)
    gini = 1 - calc
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    column = data[:, -1]

    # Count unique values
    values, counts = np.unique(column, return_counts=True)
    num_rows = data.shape[0]
    entropy = -1 * np.sum((counts/num_rows)*np.log2(counts/num_rows))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy

class DecisionNode:

    
    def __init__(self, data, impurity_func, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.impurity_func = impurity_func  
        self.gain_ratio = gain_ratio
        self.feature_importance = 0
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        ###########################################################################
        values, counts = np.unique(self.data[:,-1], return_counts=True)
        pred = values[np.argmax(counts)]
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        ###########################################################################
        ###########################################################################
        self.children.append(node)
        self.children_values.append(val)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        
    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """
        ###########################################################################
        ###########################################################################
        # Calculate the proportion of data samples containing the feature
        proportion_of_sampels = len(self.data) / n_total_sample     
        # Multiply the proportion by the effectiveness of the feature's split to get feature importance
        self.feature_importance = proportion_of_sampels * (self.goodness_of_split(self.feature)[0])
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting 
                  according to the feature values.
        """
        goodness = 0
        groups = {} # groups[feature_value] = data_subset
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        values = np.unique(self.data[:, feature])  # Get unique values in the specified column

        # Split data into groups based on the feature values and store in the dictionary
        for value in values:
            group_data = self.data[self.data[:, feature] == value]
            groups[value] = group_data
        
        # Calculate the total impurity of the dataset before the split
        total_impurity = self.impurity_func(self.data)
        
        # Calculate the weighted impurity of each group and sum them
        weighted_impurity = np.sum((group_data.shape[0] / self.data.shape[0]) * 
                                self.impurity_func(group_data) for group_data in groups.values())
    
        goodness = total_impurity - weighted_impurity
        
        # If gain ratio is to be used, compute split information and adjust goodness
        if self.gain_ratio:
            split_information = -1 * sum((group.shape[0] / self.data.shape[0]) * np.log2(group.shape[0] / self.data.shape[0])
                             for group in groups.values() if group.size > 0)
            
            # Adjust goodness by the split information (handle zero to avoid division by zero)    
            goodness = goodness / split_information if split_information != 0 else 0
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return goodness, groups
    
    def pure(self):
        """
        Check if all elements in the last column of the dataset are the same.

        Returns:
        - Boolean: True if all elements are the same (dataset is pure), False otherwise.
        """        
        
        return np.unique(self.data[:, -1]).size == 1
    
    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        ###########################################################################
        ###########################################################################
        # Check if the current depth has reached the maximum allowed depth for pruning
        if self.max_depth is not None and self.depth >= self.max_depth:
            return  # Stop further splitting if the maximum depth is reached
    
        # Find the best feature to split on and its corresponding child nodes data
        best_feature_nodes = self.find_best_feature()
    
        # If a valid feature is found and the split passes the chi-square test for significance
        if (self.feature != -1 and self.calc_chi_square(best_feature_nodes)):
            # Create a child node for each group derived from the best feature
            for val in best_feature_nodes:
                new_node = DecisionNode(data = best_feature_nodes[val], impurity_func = self.impurity_func,
                                        depth = self.depth + 1, chi = self.chi, max_depth = self.max_depth, 
                                        gain_ratio = self.gain_ratio)
                # Add each new node as a child to the current node
                self.add_child(new_node, val)
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def find_best_feature(self):
       """
        Finds the best feature to split the data on based on impurity reduction.
    
        Returns:
        - Dictionary of the groups of data corresponding to the best split feature
        """
       number_of_features = self.data.shape[1] - 1  # Total number of features excluding the target variable
       best_goodness = 0  # Best goodness of split found so far
       best_feature = -1  # Index of the best feature
       best_groups = {}  # Best groups from the split of the best feature

        # Iterate over all possible features to find the one with the highest goodness of split
       for feature in range(number_of_features):
            goodness, groups = self.goodness_of_split(feature)
            # Update the best split if this feature gives a better goodness and passes the chi-square test
            if goodness > best_goodness and self.calc_chi_square(groups):
                best_goodness = goodness
                best_feature = feature
                best_groups = groups        

        # Update the class attribute with the best feature found
       self.feature = best_feature
        # Calculate feature importance based on the best feature found
       self.calc_feature_importance(self.data.shape[0] - 1)

       return best_groups

    def calc_chi_square(self, subset):
        """
        Calculate the chi-square statistic for a given subset of data to determine if the split
        made by a feature is statistically significant.
    
        Input:
        - subset: Dictionary of data subsets grouped by feature values.

        Returns:
        - Boolean: True if the chi-square statistic is greater than the critical value (significant), False otherwise.
        """
        # If self.chi is set to 1, always return True (ignores the chi-square test)
        if self.chi == 1:
            return True

        # Calculate the expected probabilities for each class label ('e' and 'p')
        e_prob = np.count_nonzero(self.data[:, -1] == "e") / self.data.shape[0]  # Probability of 'e'
        p_prob = np.count_nonzero(self.data[:, -1] == "p") / self.data.shape[0]  # Probability of 'p'
        chi_square = 0  # Initialize chi-square statistic

        # Iterate over each group in the subset to calculate its contribution to the chi-square statistic
        for feature_values in subset.values():
            D_f = len(feature_values)  # Total instances in this group
            p_f = np.count_nonzero(feature_values[:, -1] == 'e')  # Count of 'e' in this group
            n_f = np.count_nonzero(feature_values[:, -1] == 'p')  # Count of 'p' in this group
            E_e = D_f * e_prob  # Expected count of 'e'
            E_p = D_f * p_prob  # Expected count of 'p'

            # Compute the chi-square value for each class and add to the total chi-square statistic
            chi_square += (n_f - E_p)**2 / E_p + (p_f - E_e)**2 / E_e

        # Calculate the number of degrees of freedom for the chi-square distribution
        num_of_values = len(subset) - 1
        # Compare the calculated chi-square statistic with the critical value from the chi-square distribution table
        # 'chi_table' should be a predefined table or function that gives the critical chi-square value for a given degrees of freedom and significance level
        return chi_square > chi_table[num_of_values][self.chi]

                    
class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data # the relevant data for the tree
        self.impurity_func = impurity_func # the impurity function to be used in the tree
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio #
        self.root = None # the root node of the tree
        
    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = None
        ###########################################################################
        ###########################################################################
        # Create a root node with the full dataset and other parameters
        self.root = DecisionNode(data=self.data, impurity_func=self.impurity_func, chi=self.chi,
                             max_depth=self.max_depth, gain_ratio=self.gain_ratio)
    
        # Start with a queue that includes only the root node
        queue = [self.root]

        # Process nodes in the queue until it's empty
        while(queue):
            # Pop the first node from the queue
            node = queue.pop(0)

            # Check if the current node's depth exceeds the maximum allowed depth and break the loop if true
            if node.depth >= node.max_depth:
                break
        
            # Check if the current node is pure (all data have the same class label)
            elif node.pure():
                # Mark this node as a terminal node (no further splitting)
                node.terminal = True
        
            # If the node is not pure and can still be split
            else:
                # Perform the split operation, which also creates child nodes
                node.split()
                # Extend the queue with the child nodes created from the split
                queue += node.children

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    
    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: an row vector from the dataset. Note that the last element 
                    of this vector is the label of the instance.
     
        Output: the prediction of the instance.
        """
        pred = None
        ###########################################################################
        ###########################################################################
        if self.root is None:
            return None  # If no tree exists, no prediction can be made
    
        # Start at the root of the tree
        current_node = self.root

        # Continue traversing the tree until a terminal node is reached
        while not current_node.terminal:
            # Get the feature value of the current node to decide which child node to go to next
            feature_value = instance[current_node.feature]

        # Check if the current feature value has a corresponding child node
            if feature_value in current_node.children_values:
            # Move to the appropriate child node based on the feature value
                current_node = current_node.children[current_node.children_values.index(feature_value)]
            else:
            # If no child node matches the feature value, break the loop (e.g., handling missing or unseen feature values)
                break
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return current_node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset 
     
        Input:
        - dataset: the dataset on which the accuracy is evaluated
     
        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        good_prediction  = 0
        for instance in dataset:
            if instance[-1] == self.predict(instance):
                good_prediction+=1 # if predict is correct, add one
        accuracy = (good_prediction/len(dataset))
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return accuracy
        
    def depth(self):
        return self.root.depth()



def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy 
    as a function of the max_depth. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation  = []
    
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        ###########################################################################
        ###########################################################################
        # create root with given max depth
        tree_root = DecisionTree(data = X_train, impurity_func = calc_entropy, max_depth = max_depth, gain_ratio = True) 
        tree_root.build_tree() # build tree
        training.append(tree_root.calc_accuracy(X_train)) # append accuracy on train data
        validation.append(tree_root.calc_accuracy(X_validation)) # append accuracy on validation data
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    return training, validation


def chi_pruning(X_train, X_test):

    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc  = []
    depth = []
    ###########################################################################
    ###########################################################################
    # check every chi value given in table
    for chi in [1, 0.5, 0.25, 0.1, 0.05, 0.0001]:
        tree = DecisionTree(data = X_train, impurity_func = calc_entropy, chi = chi , gain_ratio = True) # create root with given chi
        tree.build_tree() # build tree
        chi_training_acc.append(tree.calc_accuracy(X_train)) # append accuracy on train data
        chi_validation_acc.append(tree.calc_accuracy(X_test)) # append accuracy on validation data
        depth.append(max_depth(tree.root)) # append depth of tree to depth
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
        
    return chi_training_acc, chi_validation_acc, depth

def max_depth(node):
    """
    Find the max depth under the given node
    Input:
    - node: a node in the tree
    Output: 
    int: the max depth under the given node
    """
    depth = node.depth # init depth
    if (node.terminal): # if its a leaf, return its depth
        return depth
    for child in node.children: 
        depth = max(depth, max_depth(child)) # return max between this node depth and recursive call.
    return depth

def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    ###########################################################################
    ###########################################################################
    if node == None: #check if node exist
        return 0 
    count = 1
    if (node.terminal): #if it a leaf, return 1
        return 1
    for child in node.children: ## add nodes under node's children
        count += count_nodes(child)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return count






