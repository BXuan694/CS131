import numpy as np
from skimage import color


def energy_function(image):
    """Computes energy of the input image.

    For each pixel, we will sum the absolute value of the gradient in each direction.
    Don't forget to convert to grayscale first.

    Hint: you can use np.gradient here

    Args:
        image: numpy array of shape (H, W, 3)

    Returns:
        out: numpy array of shape (H, W)
    """
    H, W, _ = image.shape
    out = np.zeros((H, W))

    ### YOUR CODE HERE
    G_x=np.zeros((H,W))
    G_y=np.zeros((H,W))
    img_gray=color.rgb2gray(image)
    for i in range(H):
        for j in range(W):
            if(i==0):
                if(j==0):
                    G_x[i,j]=(img_gray[i,j+1]-img_gray[i,j])
                    G_y[i,j]=(img_gray[i+1,j]-img_gray[i,j])
                   
                elif(j==W-1):
                    G_x[i,j]=(img_gray[i,j]-img_gray[i,j-1])
                    G_y[i,j]=(img_gray[i+1,j]-img_gray[i,j])
                else:
                    G_x[i,j]=(img_gray[i,j+1]-img_gray[i,j-1])/2
                    G_y[i,j]=(img_gray[i+1,j]-img_gray[i,j])
            elif(i==H-1):
                if(j==0):
                    G_x[i,j]=(img_gray[i,j+1]-img_gray[i,j])
                    G_y[i,j]=(img_gray[i,j]-img_gray[i-1,j])
                elif(j==W-1):
                    G_x[i,j]=(img_gray[i,j]-img_gray[i,j-1])
                    G_y[i,j]=(img_gray[i,j]-img_gray[i-1,j])
                else:
                    G_x[i,j]=(img_gray[i,j+1]-img_gray[i,j-1])/2
                    G_y[i,j]=(img_gray[i,j]-img_gray[i-1,j])
            else:
                if(j==0):
                    G_x[i,j]=(img_gray[i,j+1]-img_gray[i,j])
                    G_y[i,j]=(img_gray[i+1,j]-img_gray[i-1,j])/2
                elif(j==W-1):
                    G_x[i,j]=(img_gray[i,j]-img_gray[i,j-1])
                    G_y[i,j]=(img_gray[i+1,j]-img_gray[i-1,j])/2
                else:
                    G_x[i,j]=(img_gray[i,j+1]-img_gray[i,j-1])/2
                    G_y[i,j]=(img_gray[i+1,j]-img_gray[i-1,j])/2
    out=np.abs(G_x)+np.abs(G_y);
    ### END YOUR CODE

    return out


def compute_cost(image, energy, axis=1):
    """Computes optimal cost map (vertical) and paths of the seams.

    Starting from the first row, compute the cost of each pixel as the sum of energy along the
    lowest energy path from the top.
    We also return the paths, which will contain at each pixel either -1, 0 or 1 depending on
    where to go up if we follow a seam at this pixel.

    Make sure your code is vectorized because this function will be called a lot.
    You should only have one loop iterating through the rows.

    Args:
        image: not used for this function
               (this is to have a common interface with compute_forward_cost)
        energy: numpy array of shape (H, W)
        axis: compute cost in width (axis=1) or height (axis=0)

    Returns:
        cost: numpy array of shape (H, W)
        paths: numpy array of shape (H, W) containing values -1, 0 or 1
    """
    energy = energy.copy()

    if axis == 0:
        energy = np.transpose(energy, (1, 0))

    H, W = energy.shape

    cost = np.zeros((H, W))
    paths = np.zeros((H, W), dtype=np.int)

    # Initialization
    cost[0] = energy[0]
    paths[0] = 0  # we don't care about the first row of paths

    ### YOUR CODE HERE
    for i in range(1,H):
        for j in range(W):
            tmp=cost[i-1,max([0,j-1]):min([W,j+2])]
            idx=np.where(tmp==tmp.min())[0][0];
            cost[i,j]=energy[i,j]+min(tmp)
            if(j==0):
                paths[i,j]=idx;
            else:
                paths[i,j]=idx-1;
    ### END YOUR CODE

    if axis == 0:
        cost = np.transpose(cost, (1, 0))
        paths = np.transpose(paths, (1, 0))

    # Check that paths only contains -1, 0 or 1
    assert np.all(np.any([paths == 1, paths == 0, paths == -1], axis=0)), \
           "paths contains other values than -1, 0 or 1"

    return cost, paths


def backtrack_seam(paths, end):
    """Backtracks the paths map to find the seam ending at (H-1, end)

    To do that, we start at the bottom of the image on position (H-1, end), and we
    go up row by row by following the direction indicated by paths:
        - left (value -1)
        - middle (value 0)
        - right (value 1)

    Args:
        paths: numpy array of shape (H, W) containing values -1, 0 or 1
        end: the seam ends at pixel (H, end)

    Returns:
        seam: np.array of indices of shape (H,). The path pixels are the (i, seam[i])
    """
    H, W = paths.shape
    seam = np.zeros(H, dtype=np.int)

    # Initialization
    seam[H-1] = end

    ### YOUR CODE HERE
    for i in range(H-1):
        base=seam[H-1-i]
        seam[H-2-i]=base+paths[H-1-i,base]
    ### END YOUR CODE

    # Check that seam only contains values in [0, W-1]
    assert np.all(np.all([seam >= 0, seam < W], axis=0)), "seam contains values out of bounds"

    return seam


def remove_seam(image, seam):
    """Remove a seam from the image.

    This function will be helpful for functions reduce and reduce_forward.

    Args:
        image: numpy array of shape (H, W, C) or shape (H, W)
        seam: numpy array of shape (H,) containing indices of the seam to remove

    Returns:
        out: numpy array of shape (H, W-1, C) or shape (H, W-1)
    """

    # Add extra dimension if 2D input
    
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)

    out = None
    H, W, C = image.shape
    ### YOUR CODE HERE
    out=np.zeros((H,W-1,C),dtype=type(image[0,0,0]));
    for i in range(H):
        for j in range(W-1):
            if((j<seam[i])):
                out[i,j]=image[i,j]
            else:
                out[i,j]=image[i,j+1]

    ### END YOUR CODE
    out = np.squeeze(out)  # remove last dimension if C == 1
    return out


def reduce(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Reduces the size of the image using the seam carving process.

    At each step, we remove the lowest energy seam from the image. We repeat the process
    until we obtain an output of desired size.
    Use functions:
        - efunc
        - cfunc
        - backtrack_seam
        - remove_seam

    Args:
        image: numpy array of shape (H, W, 3)
        size: size to reduce height or width to (depending on axis)
        axis: reduce in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, 3) if axis=0, or (H, size, 3) if axis=1
    """

    out = np.copy(image)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert W > size, "Size must be smaller than %d" % W

    assert size > 0, "Size must be greater than zero"

    ### YOUR CODE HERE
    for iteration in range(W-size):
        cost,path=cfunc(out,efunc(out))
        end = np.argmin(cost[-1])
        indx=backtrack_seam(path, end)
        out=remove_seam(out, indx)
    ### END YOUR CODE

    assert out.shape[1] == size, "Output doesn't have the right shape"

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def duplicate_seam(image, seam):
    """Duplicates pixels of the seam, making the pixels on the seam path "twice larger".

    This function will be helpful in functions enlarge_naive and enlarge.

    Args:
        image: numpy array of shape (H, W, C)
        seam: numpy array of shape (H,) of indices

    Returns:
        out: numpy array of shape (H, W+1, C)
    """

    H, W, C = image.shape
    out = np.zeros((H, W + 1, C))
    ### YOUR CODE HERE
    for i in range(H):
        for j in range(W+1):
            if(j<=seam[i]):
                out[i,j]=image[i,j]
            else:
                out[i,j]=image[i,j-1]
    ### END YOUR CODE

    return out


def enlarge_naive(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Increases the size of the image using the seam duplication process.

    At each step, we duplicate the lowest energy seam from the image. We repeat the process
    until we obtain an output of desired size.
    Use functions:
        - efunc
        - cfunc
        - backtrack_seam
        - duplicate_seam

    Args:
        image: numpy array of shape (H, W, C)
        size: size to increase height or width to (depending on axis)
        axis: increase in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, C) if axis=0, or (H, size, C) if axis=1
    """

    out = np.copy(image)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]
    #print("size,w:",size,W)

    assert size > W, "size must be greather than %d" % W

    ### YOUR CODE HERE
    for iteration in range(size-W):
        cost,path=cfunc(out,efunc(out))
        end = np.argmin(cost[-1])
        indx=backtrack_seam(path, end)
        out=duplicate_seam(out,indx)
    ### END YOUR CODE

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def find_seams(image, k, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Find the top k seams (with lowest energy) in the image.

    We act like if we remove k seams from the image iteratively, but we need to store their
    position to be able to duplicate them in function enlarge.

    We keep track of where the seams are in the original image with the array seams, which
    is the output of find_seams.
    We also keep an indices array to map current pixels to their original position in the image.

    Use functions:
        - efunc
        - cfunc
        - backtrack_seam
        - remove_seam

    Args:
        image: numpy array of shape (H, W, C)
        k: number of seams to find
        axis: find seams in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        seams: numpy array of shape (H, W)
    """

    image = np.copy(image)
    if axis == 0:
        image = np.transpose(image, (1, 0, 2))

    H, W, C = image.shape
    assert W > k, "k must be smaller than %d" % W

    # Create a map to remember original pixel indices
    # At each step, indices[row, col] will be the original column of current pixel
    # The position in the original image of this pixel is: (row, indices[row, col])
    # We initialize `indices` with an array like (for shape (2, 4)):
    #     [[1, 2, 3, 4],
    #      [1, 2, 3, 4]]
    indices = np.tile(range(W), (H, 1))  # shape (H, W)

    # We keep track here of the seams removed in our process
    # At the end of the process, seam number i will be stored as the path of value i+1 in `seams`
    # An example output for `seams` for two seams in a (3, 4) image can be:
    #    [[0, 1, 0, 2],
    #     [1, 0, 2, 0],
    #     [1, 0, 0, 2]]
    seams = np.zeros((H, W), dtype=np.int)
    # Iteratively find k seams for removal
    for i in range(k):
        # Get the current optimal seam
        energy = efunc(image)
        cost, paths = cfunc(image, energy)
        end = np.argmin(cost[H - 1])
        seam = backtrack_seam(paths, end)

        # Remove that seam from the image
        image = remove_seam(image, seam)
        
        # Store the new seam with value i+1 in the image
        # We can assert here that we are only writing on zeros (not overwriting existing seams)
        assert np.all(seams[np.arange(H), indices[np.arange(H), seam]] == 0), \
            "we are overwriting seams"
        seams[np.arange(H), indices[np.arange(H), seam]] = i + 1
        # We remove the indices used by the seam, so that `indices` keep the same shape as `image`
        indices = remove_seam(indices, seam)
    if axis == 0:
        seams = np.transpose(seams, (1, 0))
    return seams


def enlarge(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Enlarges the size of the image by duplicating the low energy seams.

    We start by getting the k seams to duplicate through function find_seams.
    We iterate through these seams and duplicate each one iteratively.

    Use functions:
        - find_seams
        - duplicate_seam

    Args:
        image: numpy array of shape (H, W, C)
        size: size to reduce height or width to (depending on axis)
        axis: enlarge in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, C) if axis=0, or (H, size, C) if axis=1
    """

    out = np.copy(image)
    # Transpose for height resizing
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H, W, C = out.shape

    assert size > W, "size must be greather than %d" % W

    assert size <= 2 * W, "size must be smaller than %d" % (2 * W)

    ### YOUR CODE HERE
    k=size-W
    seams=find_seams(out, k, 1, efunc, cfunc)
    for i in range(k):
        seam=np.where(seams==(i+1))[:][1]
        out=duplicate_seam(out,seam)
        Hs, Ws = np.shape(seams)
        seams1 = np.zeros((Hs, Ws + 1))
        for si in range(Hs):
            for sj in range(Ws+1):
                if((sj<=seam[si])):
                    seams1[si,sj]=seams[si,sj]
                else:
                    seams1[si,sj]=seams[si,sj-1]
        seams=seams1
    ### END YOUR CODE

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def compute_forward_cost(image, energy):
    """Computes forward cost map (vertical) and paths of the seams.

    Starting from the first row, compute the cost of each pixel as the sum of energy along the
    lowest energy path from the top.
    Make sure to add the forward cost introduced when we remove the pixel of the seam.

    We also return the paths, which will contain at each pixel either -1, 0 or 1 depending on
    where to go up if we follow a seam at this pixel.

    Args:
        image: numpy array of shape (H, W, 3) or (H, W)
        energy: numpy array of shape (H, W)

    Returns:
        cost: numpy array of shape (H, W)
        paths: numpy array of shape (H, W) containing values -1, 0 or 1
    """

    image = color.rgb2gray(image)
    H, W = image.shape

    cost = np.zeros((H, W))
    paths = np.zeros((H, W), dtype=np.int)

    # Initialization
    cost[0] = energy[0]
    for j in range(W):
        if j > 0 and j < W - 1:
            cost[0, j] += np.abs(image[0, j+1] - image[0, j-1])
    paths[0] = 0  # we don't care about the first row of paths

    ### YOUR CODE HERE
    for i in range(1,H):
        for j in range(W):
            if (j == 0):
                cost[i,j]=energy[i,j]+min([cost[i-1,j],cost[i-1,j+1]+np.abs(image[i,j+1]-image[i-1,j])])
            elif(j == W-1):
                cost[i,j]=energy[i,j]+min([cost[i-1,j-1]+np.abs(image[i-1,j]-image[i,j-1]),cost[i-1,j]])
            else:
                cost[i,j]=energy[i,j]+np.abs(image[i,j+1]-image[i,j-1])+min([cost[i-1,j-1]+np.abs(image[i-1,j]-image[i,j-1]),cost[i-1,j],cost[i-1,j+1]+np.abs(image[i,j+1]-image[i-1,j])])
            tmp=cost[i-1,max([0,j-1]):min([W,j+2])]
            idx=np.where(tmp==tmp.min())[0][0];
            if(j==0):
                paths[i,j]=idx;
            else:
                paths[i,j]=idx-1;        
    ### END YOUR CODE

    # Check that paths only contains -1, 0 or 1
    assert np.all(np.any([paths == 1, paths == 0, paths == -1], axis=0)), \
           "paths contains other values than -1, 0 or 1"

    return cost, paths


def reduce_fast(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Reduces the size of the image using the seam carving process. Faster than `reduce`.

    Use your own implementation (you can use auxiliary functions if it helps like `energy_fast`)
    to implement a faster version of `reduce`.

    Hint: do we really need to compute the whole cost map again at each iteration?

    Args:
        image: numpy array of shape (H, W, C)
        size: size to reduce height or width to (depending on axis)
        axis: reduce in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, C) if axis=0, or (H, size, C) if axis=1
    """

    out = np.copy(image)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert W > size, "Size must be smaller than %d" % W

    assert size > 0, "Size must be greater than zero"

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    assert out.shape[1] == size, "Output doesn't have the right shape"

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def remove_object(image, mask):
    """Remove the object present in the mask.

    Returns an output image with same shape as the input image, but without the object in the mask.

    Args:
        image: numpy array of shape (H, W, 3)
        mask: numpy boolean array of shape (H, W)

    Returns:
        out: numpy array of shape (H, W, 3)
    """
    out = np.copy(image)

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out
