# Useful starting lines
import numpy as np
from proj1_helpers import *
from helpers import *
from implementations import *
print("Importation complete")
DATA_TRAIN_PATH = '../data/train.csv'
DATA_TEST_PATH = '../data/test.csv'

y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
print("training data is loaded")
tX = data_cleaning(tX)

_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
print("testing data is loaded")
tX_test = data_cleaning(tX_test)

# test ridge regression with poly basis of degree 7
# This is the best solution we found (with cleaning and no PCA)
print("computing weights...")
lambda_ = 4.64158883361e-06
degree = 9
poly_basis_te = build_poly(tX_test, degree)
poly_basis_tr = build_poly(tX, degree)
w, loss = ridge_regression(y, poly_basis_tr, lambda_)

print("computing predictions...")
OUTPUT_PATH = '../data/submissionData/predictions.csv'
y_pred = predict_labels(w, poly_basis_te)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
assert(y_pred.shape[0]==568238)
print("done")