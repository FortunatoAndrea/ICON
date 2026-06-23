from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt

def findElbow(dataset):
    # Scelta numero cluster (metodo del gomito)
    inertia = []
    maxK = 10
    for i in range(1, maxK):
        km = KMeans(n_clusters=i, n_init=5, init='random')
        km.fit(dataset)
        inertia.append(km.inertia_)

    kl = KneeLocator(range(1, maxK), inertia, curve='convex', direction='decreasing')
    k_ideale = kl.elbow
    #visualizeElbow(maxK, inertia)
    return k_ideale


def visualizeElbow(maxK, inertia):
    plt.plot(range(1, maxK), inertia, marker='o')
    plt.title('Metodo del Gomito')
    plt.xlabel('Numero di Cluster (k)')
    plt.ylabel('Inerzia')

    plt.show()
    return