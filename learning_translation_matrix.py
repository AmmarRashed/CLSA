import numpy as np, pickle

def mikolov(X, Y, W):
    result = 0
    for score in range(len(X)):
        result += np.linalg.norm(W.dot(X[score]) - Y[score])**2
    return result

np.random.seed(42)

def learn_translation_matrix(X,Y, iterations=500, alpha=0.1):
    W = np.random.random((300, 300))
    for i in range(iterations):
        gradient = np.zeros(300)
        for score in range(len(X)):
            error = X[score].dot(W) - Y[score]
            gradient += alpha * np.gradient(error)
        W += gradient
        if i == 2000:
            alpha /= 100

        if i%1000 == 0:
            alpha *= 0.8
            print("Mikolov distance: {}".format(mikolov(X, Y, W)))

X, Y = pickle.load(open("testing_vecs.pickle", "rb"))
W = learn_translation_matrix(X, Y, iterations=30000, alpha=0.0001)