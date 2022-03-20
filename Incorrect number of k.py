from sklearn.datasets import make_blobs
import matplotlib.pyplot as ply

# Generating the dataset

X, y = make_blobs(n_samples=1000, random_state=random_state)

# Incorrect clusters
y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)

plt.scatter(X[:,0],X[:,1],c=y)
plt.title("Globular data")
plt.show()

plt.scatter(X[:,0],X[:,1],c=y_pred)
plt.title("Incorrect number of classes")
plt.show()

# Incorrect clusters
y_pred = KMeans(n_clusters=5, random_state=random_state).fit_predict(X)

# Plotting the cluster predictions
plt.scatter(X[:,0],X[:,1],c=y_pred)
plt.title("Incorrect number of classes")
plt.show()

