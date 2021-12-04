## CODE

#### Imports & Global Settings

import numpy as np
from scipy import signal
from scipy.sparse import spdiags, csc_matrix
from scipy.sparse.linalg import cg, spsolve, spilu, LinearOperator
from PIL import Image
from datetime import datetime

#### Array Helper Functions

def diff(x, axis=0):
    if axis==0:
        return x[1:,:]-x[:-1,:]
    else:
        return x[:,1:]-x[:,:-1]


def cyclic_diff(x,axis=0):
    if axis==0:
        return x[0,:]-x[-1,:]
    else:
        return (x[:,0]-x[:,-1])[None,:].T


def flatten_by_cols(x):
    return x.T.reshape(np.prod(x.shape), -1).flatten()


def flatten_by_rows(x):
    return x.reshape(-1, np.prod(x.shape)).flatten()


def geometric_mean(image):
    try:
        assert image.ndim == 3, 'Warning: Expected a 3d-array.  Returning input as-is.'
        return np.power(np.prod(image, axis=2), 1/3)
    except AssertionError as msg:
        print(msg)
        return image

def normalize_array(array):
    if array.ndim==3:
        array_ = array - array.min(axis=2).min(axis=1).min(axis=0)#.astype(np.float32)
        return array_ / array_.max(axis=2).max(axis=1).max(axis=0)
    
    if array.ndim==2:
        array_ = array - array.min(axis=1).min(axis=0)#.astype(np.float32)
        return array_ / array_.max(axis=1).max(axis=0)

    if array.ndim==1:
        array_ = array - array.min(axis=0)#.astype(np.float32)
        return array_ / array_.max(axis=0)


def imresize(image, scale=-1, size=(-1,-1)):
    ''' image: numpy array with shape (n, m) or (n, m, 3)
       scale: mulitplier of array height & width (if scale > 0)
       size: (num_rows, num_cols) 2-tuple of ints > 0 (only used if scale <= 0)'''
    
    if image.ndim==2:
        im = Image.fromarray(image)
        if scale > 0:
            width, height = im.size
            newsize = (int(width*scale), int(height*scale))
        else:
            newsize = (size[1],size[0])    #         numpy index convention is reverse of PIL

        return np.array(im.resize(newsize))

    if scale > 0:
        height = np.max(1,image.shape[0])
        width = np.max(1,image.shape[1])
        newsize = (int(width*scale), int(height*scale))
    else:
        newsize = (size[1],size[0])    

    tmp = np.zeros((newsize[1],newsize[0],3))
    for i in range(3):
        im = Image.fromarray(image[:,:,i])
        tmp[:,:,i] = np.array(im.resize(newsize))

    return tmp


#### Texture Functions

def delta(x):   
    
    dt0_v = np.vstack([diff(x, axis=0),cyclic_diff(x,axis=0)])
    dt0_h = np.hstack([diff(x,axis=1),cyclic_diff(x,axis=1)])
    return dt0_v, dt0_h

def kernel(dt0_v, dt0_h, sigma):
    try:
        assert sigma%2==1, 'Warning: sigma should be odd'
    except AssertionError as warning:
        print('***** '+warning+' *****')
    n_pad = int(sigma/2)
    
    kernel_v = signal.convolve(dt0_v,  np.ones((sigma,1)), method='fft')[n_pad:n_pad+dt0_v.shape[0],:]
    kernel_h = signal.convolve(dt0_h,  np.ones((1,sigma)), method='fft')[:,n_pad:n_pad+dt0_h.shape[1]]
    
    return kernel_v, kernel_h

def textures(dt0_v, dt0_h, kernel_v, kernel_h, sharpness):
    #return array from center with shape same as input array 

    W_v = 1/(np.abs(kernel_v) * np.abs(dt0_v) + sharpness)
    W_h = 1/(np.abs(kernel_h) * np.abs(dt0_h) + sharpness)

    return W_v, W_h


#### Illumination Map Function 

def construct_map(wx, wy, lamda,):
    
    r, c = wx.shape        
    k = r * c
    
    dx = -lamda * flatten_by_cols(wx) 
    dy = -lamda * flatten_by_cols(wy)
    
    wx_permuted_cols = np.roll(wx,1,axis=1) # tmp_x
    dx_permuted_cols = -lamda * flatten_by_cols(wx_permuted_cols)  # dxa
    
    wy_permuted_rows = np.roll(wy,1,axis=0) # tmp_y
    dy_permuted_rows = -lamda * flatten_by_cols(wy_permuted_rows)   # dya

    D = 1 - (dx + dy + dx_permuted_cols + dy_permuted_rows)
        
    wx_permuted_cols_head = np.zeros_like(wx_permuted_cols) 
    wx_permuted_cols_head[:,0] = wx_permuted_cols[:,0]   # tmp_xx             LAST COLUMN OF WX MOVED TO 1ST COLUMN, ALL BUT 1ST COLUMN IS THEN SET TO 0
    dx_permuted_cols_head = -lamda * flatten_by_cols(wx_permuted_cols_head)  # dxd1
    
    wy_permuted_rows_head = np.zeros_like(wy_permuted_rows)
    wy_permuted_rows_head[0,:] = wy_permuted_rows[0,:]    # tmp_yy            LAST ROW OF WY MOVED TO 1ST ROW, ALL BUT 1ST ROW IS THEN SET TO 0
    dy_permuted_rows_head = -lamda * flatten_by_cols(wy_permuted_rows_head)    # dyd1

    wx_no_tail = np.zeros_like(wx)  #  NO PERMUTATION
    wx_no_tail[:,:-1] = wx[:,:-1]  #  wxx      LAST COLUMN OF WX IS 0
    dx_no_tail = -lamda * flatten_by_cols(wx_no_tail)   # dxd2

    wy_no_tail = np.zeros_like(wy)  #  NO PERMUTATION
    wy_no_tail[:-1,:] = wy[:-1,:]  #  wyy       LAST ROW OF WY IS 0
    dy_no_tail = -lamda * flatten_by_cols(wy_no_tail)   # dyd2
    
    Ax = spdiags([dx_permuted_cols_head, dx_no_tail], [-k+r, -r], k, k)  
    
    Ay = spdiags([dy_permuted_rows_head, dy_no_tail], [-r+1,-1],  k, k)
    
    d = spdiags(D, 0, k, k)
    
    A = Ax + Ay
    A = A + A.T + d

    return A


#### Sparse solver function


def solver_sparse(A, B, method='direct', CG_prec='ILU', CG_TOL=0.1, LU_TOL=0.015, MAX_ITER=50, FILL=50):
    """
    Solves for x = b/A  [[b is vector(B)]]
    A can be sparse (csc or csr) or dense
    b must be dense
    
   """
    N = A.shape[0]
    b = B.flatten(order='F')
    if method == 'cg':
        if CG_prec == 'ILU':
            # Find ILU preconditioner (constant in time)
            A_ilu = spilu(A.tocsc(), drop_tol=LU_TOL, fill_factor=FILL)  
            M = LinearOperator(shape=(N, N), matvec=A_ilu.solve)
        else:
            M = None
        x0 = np.random.random(N) # Start vector is uniform
        c, info = cg(A, b, x0=x0, tol=CG_TOL, maxiter=MAX_ITER, M=M)
        if info==MAX_ITER:
            print(f'Warning: cg max iterations ({MAX_ITER}) reached without convergence')
        
    elif method == 'direct':
        c = spsolve(A, b)       
                
    return c

def solve_linear_equation(G, A, method='cg', CG_prec='ILU', CG_TOL=0.1, LU_TOL=0.015, MAX_ITER=50, FILL=50):

    r, c = G.shape
    G_ = flatten_by_cols(G)
    g = solver_sparse(A,G_, method, CG_prec, CG_TOL, LU_TOL, MAX_ITER, FILL)
    
    return g.reshape(c,r).T

    
#### Exposure Functions

def applyK(G, k, a=-0.3293, b=1.1258, verbose=False, clip=False):

    if k==1.0:
        return G

    gamma = k**a
    beta = np.exp((1-gamma)*b)

    if verbose:
        print(f'a: {a:.4f}, b: {b:.4f}, k: {k:.4f}, gamma: {gamma:.4f}, beta: {beta}.  ----->  output = {beta:.4} * image^{gamma:.4f}')

    G_adjusted = np.power(G,gamma)*beta  #mod 20211129 0035
    
    if clip:
        G_adjusted = np.where(G_adjusted>1,1,G_adjusted)
        G_adjusted = np.where(G_adjusted<0,0,G_adjusted)

    return G_adjusted


def entropy(array, normalize=True, nbins=100):
    #a = np.real(array).flatten()  #20211128 2356
    a = array.flatten()  # reverted on 20211201
    a = a.astype(np.float32)
    
    if normalize:
        a = normalize_array(a)
        lo = 0.
        hi = 1.
    else:
        lo = min(a) - 0.001
        hi = max(a) + 0.001
        
    n_bins = complex(0,nbins)
    bins = np.r_[lo:hi:n_bins]
    hist = np.histogram(a,bins=bins)
    counts = hist[0][hist[0]>0]
    frequencies = counts / counts.sum()
    return (-1* np.dot(frequencies, np.log2(frequencies)))

#    h01 = hist[0] / np.prod(a.shape)
#    h = h01[h01>0]
#    return (-1* h* np.log2(h)).sum()


def get_dim_pixels(image,dim_pixels,dim_size=(50,50)):
    
    dim_pixels_reduced = imresize(dim_pixels,size=dim_size)

    image_reduced = imresize(image,size=dim_size)
    image_reduced = np.where(image_reduced>0,image_reduced,0)
#    Y = geometric_mean(np.real(image_reduced)) # possibly complex?
    Y = geometric_mean(image_reduced) # possibly complex?
    Y = Y[dim_pixels_reduced]
    return Y


def optimize_exposure_ratio(array, a, b, lo=1, hi=7, npoints=20, clip=True, normalize=True, nbins=100):
  
    if sum(array.shape)==0:
        return 1.0

    sample_ratios = np.r_[lo:hi:np.complex(0,npoints)].tolist()
    entropies = np.array(list(map(lambda k: entropy(applyK(array, k, a, b, clip=clip), normalize=normalize, nbins=nbins), sample_ratios)))
    optimal_index = np.argmax(entropies)
    return sample_ratios[optimal_index]
      

def bimef(image, exposure_ratio=-1, enhance=0.5, 
          a=-0.3293, b=1.1258, lamda=0.5, 
          sigma=5, scale=0.3, sharpness=0.001, 
          dim_threshold=0.5, dim_size=(50,50), 
          solver='cg', CG_prec='ILU', CG_TOL=0.1, LU_TOL=0.015, MAX_ITER=50, FILL=50, 
          clip=False, normalize=True, nbins=100, lo=1, hi=7, npoints=20,
           verbose=False, print_info=True):
    
    ''' parameters (clip, normalize, nbins) are used only by optimize_exposure_ratio()'''
  
    tic = datetime.now()

    if image.ndim == 3: 
        image = image[:,:,:3]
        image_maxRGB = image.max(axis=2)
    else: image_maxRGB = image
        
    if (scale <= 0) | (scale >= 1) : image_maxRGB_reduced = image_maxRGB
    else: image_maxRGB_reduced = imresize(image_maxRGB, scale)
        
    image_maxRGB_reduced_01 = normalize_array(image_maxRGB_reduced)
    
    ############ TEXTURE MAP  ###########################
    dt0_v, dt0_h = delta(image_maxRGB_reduced_01)
    kernel_v, kernel_h = kernel(dt0_v, dt0_h, sigma)
    wx, wy = textures(dt0_v, dt0_h, kernel_v, kernel_h, sharpness)
    ######################################################

    ############ ILLUMINATION MAP  ###########################
    illumination_map = construct_map(wx, wy, lamda) 
    ######################################################

    ############ SOLVE LINEAR EQUATION:  ###########################
    image_maxRGB_reduced_01_latent = solve_linear_equation(image_maxRGB_reduced_01, illumination_map, method=solver, CG_prec=CG_prec, CG_TOL=CG_TOL, LU_TOL=LU_TOL, MAX_ITER=MAX_ITER, FILL=FILL)
    ######################################################

    ############ RESTORE REDUCED SIZE LATENT MATRIX TO FULL SIZE:  ###########################
    if scale <=0.: image_maxRGB_01_latent = image_maxRGB_reduced_01_latent
    else: image_maxRGB_01_latent = imresize(image_maxRGB_reduced_01_latent, size=image_maxRGB.shape)
    ######################################################
    
    ############# CALCULATE WEIGHTS ###############################
    weights = np.power(image_maxRGB_01_latent, enhance)  
    weights = np.expand_dims(weights, axis=2)
    weights  = np.where(weights>1,1,weights)
    ######################################################
    
    image_01 = normalize_array(image)
    dim_pixels = np.zeros_like(image_maxRGB_01_latent)
    
    if exposure_ratio==-1:
        dim_pixels = image_maxRGB_01_latent<dim_threshold
        Y = get_dim_pixels(image_01, dim_pixels, dim_size=dim_size) 
        exposure_ratio = optimize_exposure_ratio(Y, a, b, lo=lo, hi=hi, npoints=npoints, clip=clip, normalize=normalize, nbins=nbins)
    
    image_exposure_adjusted = applyK(image_01, exposure_ratio, a, b, verbose=verbose, clip=clip) 
    image_exposure_adjusted_clipped = np.where(image_exposure_adjusted>1,1,image_exposure_adjusted)    
    
    ############ Final Result:  ###########################
    enhanced_image =  image_01 * weights + image_exposure_adjusted_clipped * (1 - weights)   
    ##################################################
    
    toc = datetime.now()

    if print_info:
        print(f'[{datetime.now().isoformat()}] exposure_ratio: {exposure_ratio:.4f}, enhance: {enhance:.4f}, lamda: {lamda:.4f}, scale: {scale:.4f}, runtime: {(toc-tic).total_seconds():.4f}s')
        
    return enhanced_image