# importing all the required libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.metrics import accuracy_score, PCA, load_digits
from sklearn.metrics import confusion_matrix
from scipy.stats import mode

# loading the dataset into variable
digits = load_digits()

# use the k-means algorithm to estimate clusters
estimate = KMeans(n_clusters=10)
clusters = estimate.fit_predict(digits.data)
estimate.cluster_centers_.shape

# printing the images
fig = plt.figure(figsize=(8,3))
for i in range(10):
    ax = fig.add_subplot(2,5,1+i,xticks=[],yticks=[])
    # display
    ax.imshow(estimate.cluster_centers_[i].reshape((8,8)), cmap=plt.cm.binary)

# assigning actual labels to obtained cluster labels
labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]


# visualising clusters using PCA (Principle Component Analysis) 
X = PCA(2).fit_transform(digits.data)

# setting arguments
kwargs = dict(cmap = plt.cm.get_cmap('rainbow', 10), edgecolor='none', alpha=0.6)

# subplots
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].scatter(X[:, 0], X[:, 1], c=labels, **kwargs)
ax[0].set_title('Learned cluster labels')

ax[1].scatter(X[:, 0], X[:, 1], c=digits.target, **kwargs)
ax[1].set_title('True labels');

# printing accuracy of cluster predictions
accuracy_score(digits.target, labels)

# confusion matrix
print(confusion_matrix(digits.target, labels))

# plotting the confusion matrix
plt.imshow(confusion_matrix(digits.target, labels),cmap='Greens', interpolation='nearest')
plt.colorbar()
plt.grid(False)
plt.ylabel('true')
plt.xlabel('predicted');