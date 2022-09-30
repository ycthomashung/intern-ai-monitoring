import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Generate a classification dataset and visualize it with scatter plot
X, y = datasets.make_classification(n_samples=1000, n_features=5, n_informative=3, n_redundant=1, n_repeated=0, random_state=0)
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=y)

# Train-test split and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape) # Check the size of the training/testing dataset
decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree.fit(X_train, y_train)
print("Testing score (not shifted): {}".format(decision_tree.score(X_test, y_test))) # 0.884

# Trial on covariate shift: change in the distribution of X for X->Y problems
np.random.seed(0)
X_shift, y_shift = datasets.make_classification(n_samples=1000, n_features=5, n_informative=3, n_redundant=1, n_repeated=0, shift=np.random.normal(size=(5,)), random_state=0)
plt.subplot(122)
plt.scatter(X_shift[:, 0], X_shift[:, 1], c=y_shift)

X_test_s1, X_test_s2, y_test_s1, y_test_s2 = train_test_split(X_shift, y_shift, test_size=0.5, random_state=1) # s1 is used for detection, s2 is for evaluation after correction
print("Testing score (shifted): {}".format(decision_tree.score(X_test_s1, y_test_s1))) # 0.626, much lower than the non-shifted score

# plt.show() # Works the best for data with 2-dim only

# Trial on detecting a covariate shift: model-based approach
# The data that should be tested on are (X_train, X_test_s1), with (X_train, X_test) used as a control
# For the resulting classification score, the higher the score, the more likely there is a dataset shift.
# This is because the classification model giving a higher score can split the old/new data easily, indicating a possible shift.
X_detect = np.vstack((X_train, X_test_s1))
y_detect = np.hstack((np.ones(X_train.shape[0]), np.zeros(X_test_s1.shape[0])))
# print(X_detect.shape, y_detect.shape) # Check if they have the correct shape
detection_model = DecisionTreeClassifier(max_depth=2, random_state=0) # A decision tree with restricted depth
detection_model.fit(X_detect, y_detect)
print("Classification score for drift detection: {}".format(detection_model.score(X_detect, y_detect))) # 0.926

X_control = np.vstack((X_train, X_test))
y_control = np.hstack((np.ones(X_train.shape[0]), np.zeros(X_test.shape[0])))
# print(X_control.shape, y_control.shape) # Check if they have the correct shape
control_model = DecisionTreeClassifier(max_depth=2, random_state=0) # A decision tree with restricted depth
control_model.fit(X_control, y_control)
print("Classification score for drift control: {}".format(control_model.score(X_control, y_control))) # 0.56