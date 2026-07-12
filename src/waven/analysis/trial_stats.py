"""Trial statistics, retinotopy maps, and variance metrics."""
from .common import *
from .receptive_fields import *
from .nonlinear_models import *

def circular_variance(angles, responses):
    """Compute doubled-angle preferred orientation and circular variance.

    Args:
        angles: One-dimensional orientation-bin centers in degrees.
        responses: Response matrix with shape ``(n_angles, n_cells)``. Columns
            with a zero total response receive a zero resultant vector instead
            of producing a divide-by-zero warning.

    Returns:
        tuple[np.ndarray, np.ndarray]: Preferred orientations in ``[0, 180)``
        degrees and circular variance, where zero is maximally selective.
    """
    # responses should be of shape n_angles, n_cells
    # angles is of shape n_angles IN DEGREES
    angles_radians = np.deg2rad(angles)[:, np.newaxis]

    numerator = (responses * np.exp(angles_radians * 2j)).sum(axis=0)
    denominator = responses.sum(axis=0)
    resultant = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=complex),
        where=denominator != 0,
    )

    circular_variance = 1 - np.abs(resultant)

    preferred_angles = np.rad2deg(np.angle(resultant))
    preferred_angles = preferred_angles / 2 + 90
    preferred_angles = np.mod(preferred_angles + 90, 180)

    return preferred_angles, circular_variance


def compute_signal_related_variance(resp_a, resp_b, mean_center=True):
    '''
    compute the fraction of signal-related variance for each neuron,
    as per Stringer et al Nature 2019. Cross-validated by splitting
    responses into two halves. Note, this only is "correct" if resp_a
    and resp_b are *not* averages of many trials.

    Args:
        resp_a (ndarray): n_stimuli, n_cells
        resp_b (ndarray): n_stimuli, n_cells

    Returns:
        fraction_of_stimulus_variance: 0-1, 0 is non-stimulus-caring, 1 is only-stimulus-caring neurons
        stim_to_noise_ratio: ratio of the stim-related variance to all other variance
    '''
    if len(resp_a.shape) > 2:
        # if the stimulus is multi-dimensional, flatten across all stimuli
        resp_a = resp_a.reshape(-1, resp_a.shape[-1])
        resp_b = resp_b.reshape(-1, resp_b.shape[-1])
    ns, nc = resp_a.shape
    if mean_center:
        # mean-center the activity of each cell
        resp_a = resp_a - resp_a.mean(axis=0)
        resp_b = resp_b - resp_b.mean(axis=0)

    # compute the cross-trial stimulus covariance of each cell
    # dot-product each cell's (n_stim, ) vector from one half
    # with its own (n_stim, ) vector on the other half

    covariance = (resp_a * resp_b).sum(axis=0) / ns

    # compute the variance of each cell across both halves
    resp_a_variance = (resp_a ** 2).sum(axis=0) / ns
    resp_b_variance = (resp_b ** 2).sum(axis=0) / ns
    total_variance = (resp_a_variance + resp_b_variance) / 2

    # compute the fraction of the total variance that is
    # captured in the covariance
    fraction_of_stimulus_variance = covariance / total_variance

    # if you want, you can compute SNR as well:
    stim_to_noise_ratio = fraction_of_stimulus_variance / (1 - fraction_of_stimulus_variance)

    return fraction_of_stimulus_variance, stim_to_noise_ratio

def split_trials(n_stim, n_rep, n_split = 2):
    # n_stim, n_rep = respmat.shape[:2]
    """Function for split trials.

    Args:
        n_stim: Input value for this operation.
        n_rep: Input value for this operation.
        n_split: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    n_trial_split = n_rep // n_split
    trial_idxs = np.zeros((n_split, n_stim, n_rep), bool)

    for st_idx in range(n_stim):
        trial_order = np.random.permutation(n_rep)
        for split_idx in range(n_split):
            split_trials = trial_order[split_idx*n_trial_split : (split_idx+1) * n_trial_split]
            trial_idxs[split_idx, st_idx][split_trials] = 1

    return trial_idxs.astype(bool)

def stimresp_matrix(stimuli, responses, n_responses_per_stim = None):
        '''
        Make a stim-response matrix from a sequence of stimuli and responses

        Args:
            stimuli (ndarray): IDs of stimuli presented in sequence, of size n_trials
            responses (ndarray): Responses to each trial, of size n_trials, n_cells, n_response_window
            n_responses_per_stim (int, optional): Ignore stimuli with less repeats than this number

        Returns:
            respmat: n_unique_stim, n_repeats, n_cells, n_response_window
        '''
        unique_stim = np.unique(stimuli)
        if n_responses_per_stim is None:
            n_responses_per_stim = (stimuli  == unique_stim[0]).sum()
        n_unique_stim = len(unique_stim)
        n_cells = responses.shape[1]
        stim_ids = []
        respmat = []
        for idx, stim_id in enumerate(unique_stim):
            idxs = np.where(stimuli == stim_id)[0][:n_responses_per_stim]
            if len(idxs) < n_responses_per_stim:
                print("Stim %d only has %d repeats" % (stim_id, len(idxs)))
                continue
            respmat.append(responses[idxs])
            stim_ids.append(stim_id)
        return np.array(respmat), stim_ids


def lowess(x, y, f=1./3.):
    """
    Basic LOWESS smoother with uncertainty.
    Note:
        - Not robust (so no iteration) and
             only normally distributed errors.
        - No higher order polynomials d=1
            so linear smoother.
    """
    # get some paras
    xwidth = f*(x.max()-x.min()) # effective width after reduction factor
    N = len(x) # number of obs
    # Don't assume the data is sorted
    order = np.argsort(x)
    # storage
    y_sm = np.zeros_like(y)
    y_stderr = np.zeros_like(y)
    # define the weigthing function -- clipping too!
    tricube = lambda d : np.clip((1- np.abs(d)**3)**3, 0, 1)
    # run the regression for each observation i
    for i in range(N):
        dist = np.abs((x[order][i]-x[order]))/xwidth
        w = tricube(dist)
        # form linear system with the weights
        A = np.stack([w, x[order]*w]).T
        b = w * y[order]
        ATA = A.T.dot(A)
        ATb = A.T.dot(b)
        # solve the syste
        sol = np.linalg.solve(ATA, ATb)
        # predict for the observation only
        yest = A[i].dot(sol)# equiv of A.dot(yest) just for k
        place = order[i]
        y_sm[place]=yest
        sigma2 = (np.sum((A.dot(sol) -y [order])**2)/N )
        # Calculate the standard error
        y_stderr[place] = np.sqrt(sigma2 *
                                A[i].dot(np.linalg.inv(ATA)
                                                    ).dot(A[i]))
    return y_sm, y_stderr

def getHVA(signMap, neuron_pos, thresh=0.3, sign=1):
    """Function for getHVA.

    Args:
        signMap: Input value for this operation.
        neuron_pos: Input value for this operation.
        thresh: Input value for this operation.
        sign: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    if sign==1:
        signMap_binary=signMap>0
    elif sign==0:
        signMap_binary = signMap < 0
    kernel = np.ones((100, 100), np.uint8)
    opening = cv.morphologyEx(signMap_binary.astype('uint8'), cv.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, thresh * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 1] = 0

    sm=signMap*255*signMap_binary
    markers2 = cv.watershed(cv2.merge((sm.astype('uint8'),sm.astype('uint8'),sm.astype('uint8'))), markers)
    markers2_neurons=np.array([markers2[np.maximum(0, int(neuron_pos[i, 1])-1), np.maximum(0, int(neuron_pos[i, 0])-1)] for i in range(neuron_pos.shape[0])])#np.zeros_like(maxes[0, :])
    return markers2, markers2_neurons

def filter_nan_gaussian_conserving2(arr, sigma, mode='reflect'):
    """Apply a gaussian filter to an array with nans.

    Intensity is only shifted between not-nan pixels and is hence conserved.
    The intensity redistribution with respect to each single point
    is done by the weights of available pixels according
    to a gaussian distribution.
    All nans in arr, stay nans in gauss.
    """
    nan_msk = np.isnan(arr)

    loss = np.zeros(arr.shape)
    loss[nan_msk] = 1
    loss = ndimage.gaussian_filter(
            loss, sigma=sigma, mode=mode, cval=1)

    gauss = arr / (1-loss)
    gauss[nan_msk] = 0
    gauss = ndimage.gaussian_filter(
            gauss, sigma=sigma, mode=mode, cval=0)
    gauss[nan_msk] = np.nan

    return gauss

def visualSignMap(phasemap1, phasemap2):
    """
    calculate visual sign map from two orthogonally oriented phase maps
    """

    if phasemap1.shape != phasemap2.shape:
        raise LookupError("'phasemap1' and 'phasemap2' should have same size.")

    gradmap1 = np.gradient(phasemap1)
    gradmap2 = np.gradient(phasemap2)

    graddir1 = np.zeros(np.shape(gradmap1[0]))

    graddir2 = np.zeros(np.shape(gradmap2[0]))

    for i in range(phasemap1.shape[0]):
        for j in range(phasemap2.shape[1]):
            graddir1[i, j] = math.atan2(gradmap1[1][i, j], gradmap1[0][i, j])
            graddir2[i, j] = math.atan2(gradmap2[1][i, j], gradmap2[0][i, j])

    vdiff = np.multiply(np.exp(1j * graddir1), np.exp(-1j * graddir2))

    areamap = np.sin(np.angle(vdiff))

    return areamap

def getSignMap(neuron_pos, maxes, plotting=False):
    """Function for getSignMap.

    Args:
        neuron_pos: Input value for this operation.
        maxes: Input value for this operation.
        plotting: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    import scipy as sp
    x_pos=np.arange(0, np.max(neuron_pos[:, 0]))
    y_pos=np.arange(0, np.max(neuron_pos[:, 1]))
    grid=np.meshgrid(x_pos, y_pos)
    grid=np.array(grid).reshape(2, -1).T
    tree_A = cKDTree(neuron_pos)
    tree_B = cKDTree(grid)
    neighbourhood = tree_B.query_ball_tree(tree_A, 100)
    newx = np.ones_like(grid[:, 0]) * np.NaN
    newy = np.ones_like(grid[:, 1]) * np.NaN
    
    # Removed the wrapping 'for i in range(len(neighbourhood)):' here
    # Direct boolean evaluation is vastly faster than try/except
    for i, n in enumerate(neighbourhood):
        if n:  # If the neighborhood list is not empty
            newx[i] = np.nanmedian(maxes[0, n])
            newy[i] = np.nanmedian(maxes[1, n])
        else:
            print('no neighbour')

    newx2d = newx.reshape(y_pos.shape[0], x_pos.shape[0])
    newy2d = newy.reshape(y_pos.shape[0], x_pos.shape[0])

    newx2d_blur = filter_nan_gaussian_conserving2(newx2d, sigma=50)
    newy2d_blur = filter_nan_gaussian_conserving2(newy2d, sigma=50)

    signMap = visualSignMap(newx2d_blur, newy2d_blur)
    signMap_blur = filter_nan_gaussian_conserving2(signMap, sigma=15)
    sign_map_neurons=np.array([signMap_blur[np.maximum(0, int(neuron_pos[i, 1])-1), np.maximum(0, int(neuron_pos[i, 0])-1)] for i in range(neuron_pos.shape[0])])#np.zeros_like(maxes[0, :])

    if plotting:
        plt.figure()
        plt.imshow(signMap, cmap='coolwarm')

        plt.figure()
        plt.imshow(signMap_blur, vmin=-np.max(signMap_blur), vmax=np.max(signMap_blur), cmap='coolwarm')
        plt.colorbar()

        plt.figure()
        plt.imshow(newx2d, cmap='jet')

        plt.figure()
        plt.imshow(newy2d, cmap='jet')

    return signMap, sign_map_neurons

from skimage.color import lab2rgb

def TwoDimColorMap(X, Y, plotting=False):
    """Function for TwoDimColorMap.

    Args:
        X: Input value for this operation.
        Y: Input value for this operation.
        plotting: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    az = np.arange(0, 6)  # azimuths (assumes one screen and a half)
    el = np.arange(-1.5, 1.5)
    # I chose squares of size 8 just to illustrate the point:
    # to map neurons I would use 1 deg squares
    # to map neuropil I would use 10 or 15 deg squares

    azMean = np.mean(az)
    elMean = np.mean(el)
    azRange = np.ptp(az)
    elRange = np.ptp(el)

    azMat, elMat = np.meshgrid(az, el)

    # scale them to be L,a,b coordinates

    aMat = 2 * (azMat - azMean) / azRange * 100  # red-green
    bMat = 2 * (elMat - elMean) / elRange * 100  # blue-yellow

    Lmat = 65 * np.ones(np.shape(azMat))  # brightness (from 0 to 100)
    # where zero should be black but weirdly it is not
    # I would use transparency to encode strength of responses

    # convert them to RGB

    rgbImage = lab2rgb(np.dstack((Lmat, aMat, bMat)))
    rgbImage[rgbImage < 0] = 0
    rgbImage=skimage.transform.resize(rgbImage, (11, 27),mode='edge', order=0, anti_aliasing=True, preserve_range=True)

    if plotting:
        plt.figure()
        plt.imshow(rgbImage)
        plt.plot([0, 26], [5, 5], 'k--')
        plt.plot([0, 0], [0, 10], 'k--')
        plt.axis('image')
        plt.xlabel('Azimuth')
        plt.ylabel('Elevation')
        plt.title('Two-dimensional colormap of the screen')

    col=rgbImage[Y, X]
    return col, rgbImage


def rescale_to_minus_a_plus_a(arr, a=1.0):
    """Function for rescale to minus a plus a.

    Args:
        arr: Input value for this operation.
        a: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max == arr_min:
        return np.zeros_like(arr)  # éviter division par zéro
    arr_scaled = 2 * a * (arr - arr_min) / (arr_max - arr_min) - a
    return arr_scaled


