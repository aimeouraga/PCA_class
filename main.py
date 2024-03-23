from sklearn.datasets import load_iris
from pca_class import PCA

iris = load_iris()
X = iris['data']
y = iris['target']


n_samples, n_features = X.shape

print('Number of samples:', n_samples)
print('Number of features:', n_features)

pca = PCA(2)
# X_pro = pca.transform()

if __name__ == '__main__':
    pca.fit(X)
    pca.transform()
    X_pro = pca.transform()
    pca.n_features
    pca.covariance()
    pca.plot(X_pro, y)
