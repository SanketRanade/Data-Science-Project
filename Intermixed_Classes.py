import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

#Generating highed intermixed test data
#Define means and covariances 
mean1=[0,0]
mean2=[0,1]
mean3=[1,0]
mean4=[1,1]
cov=[[0.3,0],[0,0.3]]
sample1 = np.random.multivariate_normal(mean1, cov, 30) # Generating 4 sample distributions for each mean
sample2 = np.random.multivariate_normal(mean2, cov, 30)
sample3 = np.random.multivariate_normal(mean3, cov, 30)
sample4 = np.random.multivariate_normal(mean4, cov, 30)
X1 = np.vstack((sample1,sample4))    # label = 1  data
X2 = np.vstack((sample2,sample3))    # label = 0  data
Y1 = np.ones((60,1))
Y2 = np.zeros((60,1))
X = np.vstack((X1,X2))
Y = np.vstack((Y1,Y2))

y = Y
# Clusters prediction
y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)


# Plotting test data
plt.scatter(X1[:,0],X1[:,1],c='green')
plt.scatter(X2[:,0],X2[:,1],c='orange')
plt.title("Two Test Classes")
plt.legend(["label 1","label -1"])
plt.show()

# Plotting clustered data
plt.subplot()
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Predicted Clusters")
plt.show()

