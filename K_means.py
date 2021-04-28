from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


def main(Nsamples: int, Nclusters: int):
    
    n_samples= Nsamples
    n_clusters=Nclusters
    centers =5
    #this function generates isotropic Gaussian samples
    X, Y =  make_blobs(n_samples=n_samples , centers = centers, cluster_std=[1.0, 1.6, 0.5 , 1.3, 0.7])
    #the center function did not work for me, so I find the true centers here
    C = np.zeros((centers,2))
    for i in range(len(X)):
        label = Y[i]
        C[label]+= X[i]

    for j in range(centers):
        length = np.count_nonzero(Y ==j)
        C[j]/=length
    #initialize K means class
    KM = KMeans(n_clusters=n_clusters)
    #fit K means
    y_pred = KM.fit_predict(X)
    #generating plot
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.show
    plt.savefig('K_means_results.png')
    print('True centers :' +
         C)
    print('predicted centers :' +
         KM.cluster_centers_)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("-Ns", "--Nsample", type=int, default=1500,
                        help="Number of samples for each experiment.")
    
    parser.add_argument("-Nc", "--Nclusters", type=int, default=5,
                        help="Number of clusters for K means algorithm.")


    main(**parser.parse_args().__dict__)
   
