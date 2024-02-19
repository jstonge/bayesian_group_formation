
from modAL.models import ActiveLearner
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Assuming X_pool and y_pool are your unlabeled dataset and labels respectively
# Initialize your learner
learner = ActiveLearner(
    estimator=RandomForestClassifier(),
    X_training=X_initial, y_training=y_initial
)

# Select data points for querying
query_idx, query_instance = learner.query(X_pool, n_instances=5)

# Manually label these instances or retrieve their labels
# For demonstration, assuming manual labeling is done here
y_new = np.array([label for label in manually_labeled_data])

# Teach the learner with the new labels
learner.teach(X_pool[query_idx], y_new)

# Remove labeled instances from the pool
X_pool = np.delete(X_pool, query_idx, axis=0)
