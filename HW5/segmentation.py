import numpy as np
import random
from scipy.spatial.distance import squareform, pdist
from skimage.util import img_as_float

### Clustering Methods
def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)

    for n in range(num_iters):
        ### YOUR CODE HERE
        nums=np.zeros(k,dtype=int)
        s=np.zeros((k,D));
        ds=np.zeros(k)
        for vec in range(N):
            for j in range(k):
                ds[j]=np.linalg.norm(features[vec]-centers[j])
            idx=np.argmin(ds)
            assignments[vec]=idx
            s[idx]+=features[vec]
            nums[idx]+=1
        for j in range(k):
            centers[j]=s[j]/nums[j]
        assignments.astype(int)
        ### END YOUR CODE

    return assignments

def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find np.repeat and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)

    for n in range(num_iters):
        ### YOUR CODE HERE
        nums=np.zeros(k,dtype=int)
        s=np.zeros((k,D));
        ds=np.zeros(k)
        for vec in range(N):
            for j in range(k):
                ds[j]=np.linalg.norm(features[vec]-centers[j])
            idx=np.argmin(ds)
            assignments[vec]=idx
            s[idx]+=features[vec]
            nums[idx]+=1
        for j in range(k):
            centers[j]=s[j]/nums[j]
        assignments=assignments.astype(int)
        ### END YOUR CODE

    return assignments



def hierarchical_clustering(features, k):
    """ Run the hierarchical agglomerative clustering algorithm.

    The algorithm is conceptually simple:

    Assign each point to its own cluster
    While the number of clusters is greater than k:
        Compute the distance between all pairs of clusters
        Merge the pair of clusters that are closest to each other

    We will use Euclidean distance to defeine distance between two clusters.

    Recomputing the centroids of all clusters and the distances between all
    pairs of centroids at each step of the loop would be very slow. Thankfully
    most of the distances and centroids remain the same in successive
    iterations of the outer loop; therefore we can speed up the computation by
    only recomputing the centroid and distances for the new merged cluster.

    Even with this trick, this algorithm will consume a lot of memory and run
    very slowly when clustering large set of points. In practice, you probably
    do not want to use this algorithm to cluster more than 10,000 points.

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """



    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Assign each point to its own cluster
    assignments = np.arange(N)
    centers = np.copy(features)
    n_clusters = N

    while n_clusters > k:
        ### YOUR CODE HERE
        dmin=np.linalg.norm(centers[0]-centers[1]);
        d1=0
        d2=1
        for i in range(n_clusters-1):
            for j in range(i+1,n_clusters):
                ds=np.linalg.norm(centers[i]-centers[j])
                if(ds<dmin):
                    dmin=ds
                    d1=i
                    d2=j
        centers[d1]=(centers[d1]+centers[d2])/2
        centers=np.delete(centers,d2,0)
        for i in range(N):
            if(assignments[i]>d2-1):
                if(assignments[i]==d2):
                    assignments[i]=d1
                else:
                    assignments[i]-=1
        n_clusters-=1
        ### END YOUR CODE

    return assignments


### Pixel-Level Features
def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H*W, C))

    ### YOUR CODE HERE
    for i in range(H):
        for j in range(W):
            features[i*W+j]=img[i,j]
    ### END YOUR CODE

    return features

def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).
    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))

    ### YOUR CODE HERE
    mc=np.mean(color)
    sc=np.std(color)
    l1=np.array([i for i in range(H)])
    l2=np.array([i for i in range(W)])
    m1=np.mean(l1)
    m2=np.mean(l2)
    s1=np.std(l1)
    s2=np.std(l2)

    for i in range(H):
        for j in range(W):
            for k in range(C+2):
                if(k<C):
                    features[i*W+j,k]=(color[i,j,k]-mc)/sc
                elif(k==C):
                    features[i*W+j,k]=(i-m1)/s1
                else:
                    features[i*W+j,k]=(j-m2)/s2
    ### END YOUR CODE

    return features

def my_features(img):
    """ Implement your own features

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    features = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return features
    

### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    ### YOUR CODE HERE
    h,w=np.shape(mask)
    c=0
    for i in range(h):
        for j in range(w):
            if(mask_gt[i,j]==mask[i,j]):
                c+=1
    accuracy=c/(h*w)
    ### END YOUR CODE

    return accuracy

def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments. 
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy
