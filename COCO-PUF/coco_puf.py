import numpy as np
from sklearn.svm import LinearSVC
from scipy.linalg import khatri_rao

def my_fit( X_train, y0_train, y1_train ):
  # Use this method to train your models using training CRPs
	# X_train has 32 columns containing the challenge bits
	# y0_train contains the values for Response0
	# y1_train contains the values for Response1

    # Map training data to a higher dimensional space
    X_transformed = my_map(X_train)

    # Train SVM for the first response
    model_0 = LinearSVC()
    model_0.fit(X_transformed, y0_train)

    # Train SVM for the second response
    model_1 = LinearSVC()
    model_1.fit(X_transformed, y1_train)

    # Extract the weights and intercepts
    w0 = model_0.coef_.ravel()
    b0 = model_0.intercept_[0]

    w1 = model_1.coef_.ravel()
    b1 = model_1.intercept_[0]

    # THE RETURNED MODELS SHOULD BE TWO VECTORS AND TWO BIAS TERMS
	# If you do not wish to use a bias term, set it to 0
    return w0, b0, w1, b1


def my_map( X ):


  # Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points
 
    num_features = X.shape[1]
    num_samples = X.shape[0]
    num_mapped_features = num_features ** 2

    # Initialize matrix for mapped features
    feat = np.zeros((num_samples, num_mapped_features))

    # Compute the Khatri-Rao product for each sample
    for i in range(num_samples):
        row = X[i, :].reshape(1, -1)
        kr_product = khatri_rao(row.T, row.T)
        feat[i, :] = kr_product.flatten()

    return feat

def validate():
    # Example file paths
    train_file = "public_trn.txt"
    test_file = "public_tst.txt"

    # Load data
    X_train, y0_train, y1_train = load_data(train_file)

    # Fit the model
    w0, b0, w1, b1 = my_fit(X_train, y0_train, y1_train)

    # Print the results
    print("Weights and bias for Response0:")
    print(w0, b0)
    print("Weights and bias for Response1:")
    print(w1, b1)

    # Load test data (assuming we want to validate on the same data format)
    X_test, y0_test, y1_test = load_data(test_file)

    # Map the test data
    X_test_mapped = my_map(X_test)

    # Predict responses
    y0_pred = np.sign(X_test_mapped @ w0 + b0)
    y1_pred = np.sign(X_test_mapped @ w1 + b1)

    # Calculate accuracy
    accuracy0 = np.mean((y0_pred > 0) == y0_test)
    accuracy1 = np.mean((y1_pred > 0) == y1_test)

    print("Test accuracy for Response0:", accuracy0)
    print("Test accuracy for Response1:", accuracy1)

def load_data(file_path):
    # Load training data
    data = np.loadtxt(file_path, delimiter=" ")
    X = data[:, :32]
    y0 = data[:, 32]
    y1 = data[:, 33]
    return X, y0, y1

if __name__ == "__main__":
    validate()
