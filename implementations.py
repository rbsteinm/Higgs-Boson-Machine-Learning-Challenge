# -*- coding: utf-8 -*-
from helpers import *
from proj1_helpers import *
########################################################
########################################################
#
# Data cleaning and preparation
#
########################################################
########################################################

def data_cleaning(tX):
    """This function handles outliers(with quartiles) and unnasigned values.
    It also standardizes the data."""
    print("shape of tX before standardizing:",tX.shape)
    for col in tX.T:
        q1 = np.percentile(col,25)
        q3 = np.percentile(col,75)
        interq = q3-q1

        col_cleaned = col[abs(col)!=999]
        col_cleaned = col_cleaned[col_cleaned<=q3+interq]
        col_cleaned = col_cleaned[col_cleaned>=q1-interq]
        #print(col_cleaned.shape)
        mean = np.mean(col_cleaned)

        col[abs(col)==999] = mean
        col[col>q3+interq] = mean
        col[col<q1-interq] = mean

    tX,_,_ = standardize(tX)
    print("shape of tX before standardizing:",tX.shape)
    print("data cleaning completed")
    return tX

def PCA(tX):
    """PCA is a method that allows us to get rid of features that
    don't give enough information, ie features that are too correlated
    with other ones. The threshold indicates severity."""
    print("Previous number of features in tX:", tX.shape[1])
    threshold = 0.95
    C = 1/tX.shape[0]*tX.T.dot(tX)

    eigenvalue, eigenvector = np.linalg.eig(C)
    SUM = np.sum(eigenvalue)

    idx = np.argsort(eigenvalue)[::-1]
    eigenvector = eigenvector[:,idx]
    eigenvalue = eigenvalue[idx]

    F = 0
    k = 0
    while F < threshold:
        F += eigenvalue[k]/SUM
        k = k+1   

    eigenvector=eigenvector[:, :k]

    tX = np.dot(eigenvector.T, tX.T).T
    print("New number of features in tX:",tX.shape[1])
    print("PCA completed")
    return tX




########################################################
########################################################
#
# Helper functions
#
########################################################
########################################################

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N = len(y)
    e = y - tx.dot(w)
    return (-1/N)*(tx.T).dot(e)

def compute_loss(y, tx, w):
    """Compute the cost by MSE"""
    N = len(y)
    e = y - tx.dot(w)
    return (1/(2*N))*((e.T).dot(e))

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # temporarily get rid of the column of ones that's already there from standardization
    if(np.all(x[:, 0] == 1.0)):
        x = x[:, 1:x.shape[1]]
    result = np.ones((x.shape[0], 1))
    for i in range(1, degree+1):
       result = np.concatenate((result, x ** i), axis=1)
    return result

def split_data(x, y, ratio, seed=0):
    """Randomly splits the data given in input into two subsets (test/train).
    The ratio determines the size of the training set."""
    np.random.seed(seed)
    size = y.shape[0]
    # randomly permutes array of intergers from 0 to size-1
    indexes = np.random.permutation(size)
    tr_size = int(np.floor(ratio * size))
    # get (randomly generated) indexes of training/testing set
    tr_indexes = indexes[0:tr_size]
    te_indexes = indexes[tr_size:]
    # split x (resp. y) into two subarrays x_tr, x_te (resp. y_tr, y_te)
    x_tr = x[tr_indexes]
    y_tr = y[tr_indexes]
    x_te = x[te_indexes]
    y_te = y[te_indexes]
    return [x_tr, y_tr, x_te, y_te]

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_, degree, method_to_use):
    """Performs one round of cross-validation. For this round, the kth subset
    is the testing set, all the other ones together form the training set.This 
    method is called by the upper-layer cross validation method (cross_validation_error
    for example) each time with a different k and then a mean is calculated."""
    # get k'th subgroup in test, others in train
    x_tr = np.empty([0,31])
    x_te = np.empty([0,31])
    y_tr = np.array([])
    y_te = np.array([])
    
    for i in range(len(k_indices)):
        subgroup_x = x[k_indices[i]]
        subgroup_y = y[k_indices[i]]
        if(i != k):
            x_tr = np.concatenate((x_tr, subgroup_x))
            y_tr = np.append(y_tr, subgroup_y)
        else:
            x_te = np.concatenate((x_te, subgroup_x))
            y_te = np.append(y_te, subgroup_y)

    # form data with polynomial degree
    poly_basis_tr = build_poly(x_tr, degree)
    poly_basis_te = build_poly(x_te, degree)

    # ridge regression
    w_tr, mse_tr = method_to_use(y_tr, poly_basis_tr,lambda_)
    y_est = predict_labels(w_tr,poly_basis_te)
    fitting = error(y_te, y_est)
    
    # calculate the loss for train and test data
    #print("x shape:",poly_basis_te.shape,"y shape:",y_te.shape)
    mse_te = compute_loss(y_te, poly_basis_te, w_tr)
    loss_tr = np.sqrt(2*mse_tr)
    loss_te = np.sqrt(2*mse_te)
    return loss_tr, loss_te, fitting

def cross_validation_error(x, y, degree, lambda_, method_to_use, k_fold):
    """This method runs cross validation on the given method with 
    the given parameters, calculates the error and outputs the 
    percentage of prediction data on real data fitting."""
    seed = 1
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    k = 0
    fitting_mean = 0
    for k in range(k_fold):
        _,_,fitting = cross_validation(y, x, k_indices, k, lambda_, degree, method_to_use)
        fitting_mean += fitting
        k+=1
    fitting_mean /= k_fold
    return fitting_mean

def sigmoid_scal(t):
    """apply sigmoid function on scalar value t."""
    if(t < 0):
        return np.exp(t)/(1+np.exp(t))
    else:
        return 1/(1+np.exp(-t))
"""This allows us to call the sigmoid function element-wise on a vector"""
sigmoid = np.vectorize(sigmoid_scal)
    

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    loss = 0
    for index, row in enumerate(tx):
        loss += np.log(1+np.exp(row.dot(w)))-y[index]*(row).dot(w)
    return loss

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return tx.T.dot(sigmoid(tx.dot(w))-y)

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """   
    grad = calculate_gradient(y,tx,w)
    loss = calculate_loss(y, tx, w)
    w -= gamma*grad
    return w, loss, gamma

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    S = sigmoid(tx.dot(w))*(1-sigmoid(tx.dot(w)))
    S = np.reshape(S, [len(S)])
    return tx.T.dot(np.diag(S)).dot(tx)

def learning_by_newton_method(y, tx, w, gamma):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    loss, gradient, hessian = get_LGH(y, tx, w)
    w -= gamma * np.linalg.inv(hessian).dot(gradient)
    return w, loss

def learning_by_penalized_gradient_descent(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss, grad, hessian = get_LGH(y, tx, w)
    grad += 2*lambda_*w
    loss += lambda_*(np.linalg.norm(w)**2)
    hessian += np.diag(2*lambda_*np.ones(len(hessian)))
    w -= gamma*np.linalg.inv(hessian).dot(grad)
    return w, loss

def get_LGH(y, tx, w):
    """return the loss, gradient, and hessian."""
    loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    hessian = calculate_hessian(y, tx, w)
    return loss, gradient, hessian

def error(y,y_est):
    """Computes the percentage of right values in the 
    regression result compared to our test data"""
    diff = np.count_nonzero(y-y_est)/len(y)
    return (1-diff)*100

def compute_AMS(w, true_y, tX_te):
    """AMS is an estimator of the precision. We want to maximize it.
    true_y and y_pred should be in the form {0,1}"""
    y_pred = predict_labels(w, tX_te)
    s = sum(true_y*y_pred)
    b = sum((true_y==0)*(y_pred==1))
    return np.sqrt(2*((s+b)*np.log(1+s/(b+10))-s))

def minus_one_to_zero(y):
    """The domain of the y vector changes from {-1,1} to {0,1}.
    This is useful for logistic regression."""
    assert(np.any(y==-1))
    return (y+1)/2

def zero_to_minus_one(y):
    """The domain of the y vector changes from {0,1} to {-1,1}.
    This is useful for logistic regression."""
    assert(np.any(y==0))
    return 2*y-1

########################################################
########################################################
#
# Functions we were asked to implement
#
########################################################
########################################################


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """performs linear regression using gradient descent algorithm. 
    Returns two arrays containing weights and loss values 
    at each step of the algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        #update w
        w = w - gamma * gradient
        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """performs linear regression using stochastic gradient descent algorithm. 
    Returns two arrays containing weights and loss values 
    at each step of the algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    batch_size = 800 #try changing batch size
    
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            #compute new loss and w
            loss = compute_loss(y, tx, w) # add one loss per minibatch (compute mean)
            w = w - gamma * gradient
            # store loss and w in arrays
            ws.append(w)
            losses.append(loss)
                 
    return w, loss

def least_squares(y, tx):
    """performs linear regression by calculating 
    the least squares solution using normal equations.
    returns loss and optimal wieghts."""
    opt_w = np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y)
    #computes the loss using MSE
    mse = compute_loss(y, tx, opt_w)
    return opt_w, mse
    
def ridge_regression(y, tx, lambda_):
    """implement ridge regression. This calculates the MSE while taking in 
    accout a regularizer that is determined by lambda. This has for effect to
    penalize/avoid large weights in order to avoid overfitting."""
    # tx is the polynomial basis
    opt_w = (np.linalg.inv(tx.T.dot(tx)+lambda_*2*len(y)*np.identity(tx.shape[1])).dot(tx.T)).dot(y)
    mse = compute_loss(y, tx, opt_w)
    return opt_w, mse

def logistic_regression(y, tx, initial_w, max_iter, gamma):
    threshold = 1e-20
    losses = np.array([0,1])
    w = initial_w
    
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        w, loss, gamma = learning_by_gradient_descent(y, tx, w, gamma)
        losses[1] = losses[0]
        losses[0] = loss
        if iter % 1 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # Condition d'arrêt:
        if abs(losses[0]-losses[1]) < abs(losses[1]*threshold):
            break
            
    print("The loss={l}".format(l=calculate_loss(y, tx, w)))
    return w, loss
    
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iter, gamma):
    threshold = 1e-350
    losses = np.array([0,1])
    w = initial_w

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        w, loss = learning_by_penalized_gradient_descent(y, tx, w, gamma, lambda_)
        losses[1] = losses[0]
        losses[0] = loss
        if iter % 1 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # Condition d'arrêt: 
        if abs(losses[0]-losses[1]) < abs(losses[1]*threshold):
            break

    print("The loss={l}".format(l=calculate_loss(y, tx, w)))
    return w, loss
    
def newton_logistic_regression(y, tx, max_iter, gamma):
    threshold = 1e-10
    losses = np.array([0,1])
    w = np.zeros(tx.shape[1])
    
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        w, loss = learning_by_newton_method(y, tx, w, gamma)
        losses[1] = losses[0]
        losses[0] = loss
        if iter % 10 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # Condition d'arrêt:
        if abs(losses[0]-losses[1]) < abs(losses[1]*threshold):
            break
            
    print("The loss={l}".format(l=calculate_loss(y, tx, w)))
    return w, loss

########################################################
########################################################
#
# Some demo methods
#
########################################################
########################################################


def polynomial_regression():
    """For each degree, constructs the polynomial basis function expansion of the data
       and stores the corresponding RMSE in an array. At the end we chose the degree that
       generated the smallest RMSE. Of course we cannot test all degrees so this is not
       optimal but it helps us having a good idea of the optimal degree value."""
    # define parameters
    degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # for each degree we store the corresponding RMSE in this array
    rmse_array = np.array([])

    for ind, degree in enumerate(degrees):
        # form the data to do polynomial regression:
        polynomial_basis = build_poly(tX, degree)
        
        # least square and calculate rmse:
        weight, mse = least_squares(y, polynomial_basis)
        rmse = np.sqrt(2*mse)
        rmse_array = np.append(rmse_array, rmse)
        print("RMSE for degree", degree, ":", rmse)
    
    # plot the RMSE in function of the degree
    plt.plot(degrees, rmse_array)
    plt.xlabel('degree')
    plt.ylabel('RMSE')
    
    #compute the best degree
    best_degree = degrees[np.argmin(rmse_array)]
    print("The best degree among those we tested is", best_degree, ".")
    
    
def ridge_regression_demo(x, y, ratio, seed):
    """Calculate polyomial basis tX from x with given degree,
    splits the data according to given ratio and then run
    ridge regression on tX, y with different lambda values.
    At the end we plot the RMSEs of training/testing set in
    function of lambda in order to determine the best lambda value"""
    # define parameter
    lambdas = np.logspace(-5, 0, 15)
    degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    # split the data, and return train and test data:
    x_tr, y_tr, x_te, y_te = split_data(x, y, ratio, seed)
    
    #calculate test/train RMSE for each lambda and store them in lists
    rmse_list_tr = np.empty([len(degrees), len(lambdas)])
    rmse_list_te = np.empty([len(degrees), len(lambdas)])
    errors = np.empty([len(degrees), len(lambdas)])
    optimal_weights_err = np.empty([np.shape(tX)[1]])
    optimal_weights_rmse_te = np.empty([np.shape(tX)[1]])
    best_err = 0
    best_RMSE = 10e10
    for i, degree in enumerate(degrees):
        # for each lambda, store the best RMSE and the degree that generated it
        for j, lambd in enumerate(lambdas):
            # compute polynomial basis from given degree
            poly_basis_tr = build_poly(x_tr, degree)
            poly_basis_te = build_poly(x_te, degree)
            # compute training and testing (R)MSE for current lambda/degree
            w_tr, mse_tr = ridge_regression(y_tr, poly_basis_tr,lambd)
            mse_te = compute_loss(y_te, poly_basis_te, w_tr)
            rmse_tr = np.sqrt(2*mse_tr)
            rmse_te = np.sqrt(2*mse_te)
            err = error(y_te,predict_labels(w_tr,poly_basis_te))
            #print("Training RMSE for lambda =", lambd, "and degree", degree, ":", rmse_tr, "\n")
            #print("Testing RMSE for lambda =", lambd, "and degree", degree, ":", rmse_te, "\n")
            #print("error for lambda = ", lambd, "and degree", degree, ":", err)
            # Store RMSEs in arrays
            rmse_list_tr[i][j] = rmse_tr
            rmse_list_te[i][j] = rmse_te
            errors[i][j] = err
            # we do this to get optimal weights according to the error
            if(best_err < err):
                best_err = err
                optimal_weights_err = w_tr
            # get the optimal weights according to the testing rmse
            if(best_RMSE > rmse_te):
                best_RMSE = rmse_te
                optimal_weights_rmse_te = w_tr
        # plot figures
        plt.figure(i)
        plot_train_test(rmse_list_tr[i, :], rmse_list_te[i, :], lambdas, degree)
        # TODO we need to compute the error for each (d, lambda) value, store all in array and print best for each degree
        #err = error(y_te,predict_labels(w_tr,poly_basis_te))
        plt.title(("RR degree", degree, ".Best Error: ", best_err))
    
    # get best degree, lambda according to the testing RMSE
    degree_index_te, lambd_index_te = np.where(rmse_list_te == rmse_list_te.min())
    degree_index_te, lambd_index_te = (degree_index_te[0],lambd_index_te[0])
    best_rmse_te = rmse_list_te[degree_index_te][lambd_index_te]
    print("Best testing RMSE is", best_rmse_te, "with degree", degrees[degree_index_te], "and lambda=", lambdas[lambd_index_te], "\n")
    # get best degree and lambda according to the error
    degree_index_err, lambd_index_err = np.where(errors == errors.max())
    degree_index_err, lambd_index_err = (degree_index_err[0],lambd_index_err[0])
    best_error = errors[degree_index_err][lambd_index_err]
    print("Best fitting is", best_error, "% with degree", degrees[degree_index_err], "and lambda=", lambdas[lambd_index_err], "\n")
    #return optimal_weights, best_RMSE
    return optimal_weights_rmse_te, best_rmse_te


    
def cross_validation_demo(x, y, k_fold, degrees, lambdas, method_to_use):
    seed = 1
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    fittings = np.empty([len(degrees), len(lambdas)])
    best_fitting = 0
    best_lambda = 0
    best_degree = 0
    for i, degree in enumerate(degrees):
        for j,lambda_ in enumerate(lambdas):
                # we specify the name of the method we want to use with cross-validation
                fitting = cross_validation_error(x, y, degree, lambda_, method_to_use, k_fold)
                fittings[i,j] = fitting
                if(fitting > best_fitting):
                    best_fitting = fitting
                    best_degree = degree
                    best_lambda = lambda_
                print("fitting for degree",degree,"and lambda",lambda_,":",fitting)
        plt.figure(i)
        cross_validation_visualization(lambdas, fittings[i,:],degree)
        plt.title(("cross validation degree",degree))
    print("max fitting for lambda =",best_lambda,"degree=",best_degree,"->",best_fitting)



