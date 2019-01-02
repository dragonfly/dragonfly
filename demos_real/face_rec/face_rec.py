"""
  ===================================================
  Faces recognition example using eigenfaces and SVMs
  ===================================================

  Adapted from scikit-learn examples.
  http://scikit-learn.org/

  -- kvysyara@andrew.cmu.edu
"""

from __future__ import print_function
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC

def face_rec(x):
  n_components = int(np.floor(x[0]))
  C = x[1]
  gamma = x[2]

  # #############################################################################
  # Download the data, if not already on disk and load it as numpy arrays
  
  lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
  
  # introspect the images arrays to find the shapes (for plotting)
  n_samples, h, w = lfw_people.images.shape
  
  # for machine learning we use the 2 data directly (as relative pixel
  # positions info is ignored by this model)
  X = lfw_people.data
  n_features = X.shape[1]
  
  # the label to predict is the id of the person
  y = lfw_people.target
  target_names = lfw_people.target_names
  n_classes = target_names.shape[0]
  
  # #############################################################################
  # Split into a training set and a validation set using a stratified k fold
  
  # split into a training and validation set
  X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25,
                                                        random_state=42)
  
  # #############################################################################
  # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
  # dataset): unsupervised feature extraction / dimensionality reduction
  
  pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
  
  X_train_pca = pca.transform(X_train)
  X_valid_pca = pca.transform(X_valid)
  
  # #############################################################################
  # Train a SVM classification model
  
  clf = SVC(kernel='rbf', class_weight='balanced', C=C, gamma=gamma)
  clf = clf.fit(X_train_pca, y_train)
  
  # #############################################################################
  # Quantitative evaluation of the model quality on the validation set
  
  y_valid_pred = clf.predict(X_valid_pca)
  cf_valid = confusion_matrix(y_valid, y_valid_pred, labels=range(n_classes))
  
  y_train_pred = clf.predict(X_train_pca)
  cf_train = confusion_matrix(y_train, y_train_pred, labels=range(n_classes))
  
  train_result = (np.trace(cf_train))/float(y_train.size)
  valid_result = (np.trace(cf_valid))/float(y_valid.size)

  return valid_result


def objective(x):
  """ Main objective. """
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  return face_rec(x)


def main(x):
  return objective(x)

