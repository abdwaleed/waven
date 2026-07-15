"""Nonlinear Gabor model helper functions and tuning plots."""
from .common import *
from .receptive_fields import *
from .trial_stats import circular_variance
from ..runtime.performance import cpu_threadpool_scope, cpu_worker_count, resolve_compute_device


def spikeTrig(spk, w_i, w_r, w_c, ran):
    """Function for spikeTrig.

    Args:
        spk: Input value for this operation.
        w_i: Input value for this operation.
        w_r: Input value for this operation.
        w_c: Input value for this operation.
        ran: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    spk_sub = spk[ran:]
    spk_sum = np.sum(spk_sub)
    spk_t = spk_sub.T
    
    mu_sin = np.empty(ran)
    mu_cos = np.empty(ran)
    mu_complex = np.empty(ran)
    
    for i, dt1 in enumerate(range(ran)[::-1]):
        mu_sin[i] = np.sum(spk_t * w_i[ran-dt1:w_i.shape[0]-dt1]) / spk_sum
        mu_cos[i] = np.sum(spk_t * w_r[ran-dt1:w_r.shape[0]-dt1]) / spk_sum
        mu_complex[i] = np.sum(spk_t * w_c[ran-dt1:w_c.shape[0]-dt1]) / spk_sum

    return mu_sin, mu_cos, mu_complex


def compute_stc(a_t, b_t, mu_b_t, ran, dt1, dt2):
    """Function for compute stc.

    Args:
        a_t: Input value for this operation.
        b_t: Input value for this operation.
        mu_b_t: Input value for this operation.
        ran: Input value for this operation.
        dt1: Input value for this operation.
        dt2: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    with torch.no_grad():
        # Keep spikes fixed in time. Shape becomes (T_sub, 1) for broadcasting
        a_sub = a_t[ran:].reshape(-1, 1) 
        
        # Shift stimuli and subtract means
        b_sub1 = b_t[ran - dt1 : b_t.shape[0] - dt1] - mu_b_t[mu_b_t.shape[0] - 1 - dt1]
        b_sub2 = b_t[ran - dt2 : b_t.shape[0] - dt2] - mu_b_t[mu_b_t.shape[0] - 1 - dt2]
        
        # Multiply spikes with the first stimulus across the time dimension
        # a_sub (T, 1) * b_sub1 (T, N) -> broadcasts to (T, N)
        weighted_b1 = a_sub * b_sub1
        
        # Matmul transposes the time dimension to compute the spatial covariance
        # (N, T) @ (T, N) -> (N, N)
        c = torch.matmul(weighted_b1.T, b_sub2) / torch.sum(a_sub)
        
        return c.cpu().numpy().astype('float16')


def CovspikeTrig(spk, w, mu, ran):
    """Function for CovspikeTrig.

    Args:
        spk: Input value for this operation.
        w: Input value for this operation.
        mu: Input value for this operation.
        ran: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    ran_val = np.max(np.abs(np.array(ran)))
    Css = np.zeros((ran_val, 54*135, ran_val, 54*135), dtype='float16')
    
    device = resolve_compute_device(prefer_gpu=True)
    with cpu_threadpool_scope(), torch.no_grad():
        spk_t = torch.as_tensor(spk, device=device, dtype=torch.float32)
        w_t = torch.as_tensor(w, device=device, dtype=torch.float32)
        mu_t = torch.as_tensor(mu, device=device, dtype=torch.float32)
        
        for dt1 in range(ran_val):
            for dt2 in range(ran_val):
                print(dt1, dt2)
                Css[dt1, :, dt2, :] = compute_stc(spk_t, w_t, mu_t, ran_val, dt1, dt2)
        
        # Clear main tensors and force PyTorch to dump VRAM
        del spk_t, w_t, mu_t
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
                
    return Css


@njit(parallel=True, fastmath=True)
def CovspikeTrigC(spk, w_i, w_r, mu_i, mu_r, ran):
    """Function for CovspikeTrigC.

    Args:
        spk: Input value for this operation.
        w_i: Input value for this operation.
        w_r: Input value for this operation.
        mu_i: Input value for this operation.
        mu_r: Input value for this operation.
        ran: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    ran_val = np.max(np.abs(np.array(ran)))
    Css = np.zeros((ran_val, ran_val))
    
    spk_sub = spk[ran_val:]
    spk_sum = np.sum(spk_sub)
    
    # prange distributes the outer loop across available CPU cores automatically
    for dt1 in prange(ran_val):
        for dt2 in range(ran_val):
            w_i_sub = w_i[ran_val-dt1:w_i.shape[0]-dt1] - mu_i[dt1]
            w_r_sub = w_r[ran_val-dt2:w_r.shape[0]-dt2] - mu_r[dt2]
            
            dot_prod = np.sum(spk_sub * w_i_sub * w_r_sub, axis=0)
            Css[dt1, dt2] = dot_prod / spk_sum

    return Css


def getSVDPolar(idx, spk, ncut, args, plotting=False, more_smooth=False, smoothing_size=5):
    """Function for getSVDPolar.

    Args:
        idx: Input value for this operation.
        spk: Input value for this operation.
        ncut: Input value for this operation.
        args: Input value for this operation.
        plotting: Input value for this operation.
        more_smooth: Input value for this operation.
        smoothing_size: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    rho, phi, dd, ns, zz, hanz, dx, dy, dp, dn= args
    hanz=hanz[::-1]
    f = interpolate.LinearNDInterpolator(np.stack((rho.flatten(), phi.flatten(), dd.flatten())).T, zz.flatten().T)
    if more_smooth:
        s=ncut
        rhospace=np.linspace(0, np.max(rho), s)
        phispace=np.linspace(0, np.max(phi), s)
        dphispace = np.linspace(np.min(dd), np.max(dd), s)
        nspace = np.linspace(np.min(ns), np.max(ns), s)

        xx, yy, tt = np.meshgrid(rhospace, phispace, dphispace)
        znew=f(xx.flatten(), yy.flatten(), tt.flatten()).reshape((s, s, s))
        mask = [~np.isnan(znew)][0]
        x = xx[mask].reshape(-1)
        y = yy[mask].reshape(-1)
        t = tt[mask].reshape(-1)
        points = np.array([x, y, t]).T
        values = znew[mask].reshape(-1)

        # generate interpolated grid data
        interp_grid = griddata(points, np.nan_to_num(values), (xx, yy, tt), method='nearest')
        interp_grid=hanningconv3d(interp_grid, smoothing_size)
        if plotting:
            plt.figure()
            plt.pcolormesh(rhospace, phispace, np.nanmean(interp_grid, axis=2)[:-1, :-1],  cmap='coolwarm')
            plt.plot(rhospace, phispace, color='k', ls='none')
            plt.colorbar()
            plt.title('interp_grid')


        f = interpolate.LinearNDInterpolator(np.stack((xx.flatten(), yy.flatten(), tt.flatten())).T, interp_grid.flatten().T)

        u, s, v = svds(np.nanmean(interp_grid, axis=2), 2)
        u1, s, v1 = svds(np.nan_to_num(interp_grid.reshape(-1, ncut)), 2)
        if plotting:
            plt.figure()
            plt.plot(rhospace, abs(v[1]))
            plt.title('rho tuning curve')
            plt.figure()
            plt.plot(phispace, abs(u[:, 1]))
            plt.title('phi tuning curve')



            plt.figure()
            plt.plot(dphispace, abs(v1[1]))
            plt.title('d phi / dt tuning curve')
            plt.figure()
            plt.imshow(abs(u1[:, 1]).reshape(ncut, ncut))


        if plotting:
            return v1[1], u1[:, 1], hanz, s[1], f #
        else:
            return (dphispace, abs(v1[1])), (phispace, abs(u[:, 1])), hanz, s[1], f
    else:
        return f, hanz


def nonvis(spks, idx):
    """Function for nonvis.

    Args:
        spks: Input value for this operation.
        idx: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    spk=np.mean(spks[[0, 2, 4], :, idx], axis=0)
    s_non_vis = np.array([spks[i, :, idx] - spk for i in range(5)])
    return s_non_vis


def deconvolve_avg_pop(spks, idx):
    """Function for deconvolve avg pop.

    Args:
        spks: Input value for this operation.
        idx: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    pc_mean_pop=np.mean(spks, axis=2)

    s_non_vis=nonvis(spks, idx)
    Z=[]
    for i in range(5):
        y = s_non_vis.T[:, i]
        x = pc_mean_pop.T[:, i]
        h = ifft(fft(y) / fft(x))
        z = np.convolve(x, h)[:9000].reshape(-1, 1)
        Z.append(z)
    zt = np.stack(Z, axis=1)
    zt = np.real(zt.reshape(9000, 5).T)
    zt=np.array([zt[rep] - np.mean(zt[rep]) for rep in range(5)])
    return zt


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def getNonLinearModel(idx, spks, x, y, o, s, w_i_downsampled, w_r_downsampled, ncut):
    """Function for getNonLinearModel.

    Args:
        idx: Input value for this operation.
        spks: Input value for this operation.
        x: Input value for this operation.
        y: Input value for this operation.
        o: Input value for this operation.
        s: Input value for this operation.
        w_i_downsampled: Input value for this operation.
        w_r_downsampled: Input value for this operation.
        ncut: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    sin_w = w_i_downsampled[6:, x, y, o, s]
    cos_w = w_r_downsampled[6:, x, y, o, s]
    rho, phi, z, theta1, dx, dy= SinCosPlot(idx, spks[:, 6:, :], x, y, o, s, w_i_downsampled[6:], w_r_downsampled[6:], ncut)
    from scipy.stats import binned_statistic

    mean_rho = binned_statistic(np.nan_to_num(rho.flatten()), np.nan_to_num(z.flatten()),
                                statistic='max',
                                bins=10)
    mean_phi = binned_statistic(np.nan_to_num(phi.flatten()), np.nan_to_num(z.flatten()),
                                statistic='max',
                                bins=10)
    plt.figure()
    plt.plot(mean_rho[1][1:], mean_rho[0])
    plt.figure()
    plt.plot(mean_phi[1][1:], mean_phi[0])

    rp = np.array([cart2pol([cos_w[i]], [sin_w[i]]) for i in range(8994)])
    rh = rp[:, 0, 0, 0]
    ph = rp[:, 1, 0, 0]
    R = np.interp(rh, mean_rho[1][1:], mean_rho[0])
    P = np.interp(ph, mean_phi[1][1:], mean_phi[0])

    resp = R * P
    return resp


from scipy.stats import binned_statistic


def getNonLinearModel2(idx, spk, w_i, w_r, dphi, noise, ncut, smoothing_size, plotting=False, more_smooth=False):

    """Function for getNonLinearModel2.

    Args:
        idx: Input value for this operation.
        spk: Input value for this operation.
        w_i: Input value for this operation.
        w_r: Input value for this operation.
        dphi: Input value for this operation.
        noise: Input value for this operation.
        ncut: Input value for this operation.
        smoothing_size: Input value for this operation.
        plotting: Input value for this operation.
        more_smooth: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    rr, pp, dd, nn, zz, hanz, dx, dy, dp, dn= SinCosPlot2(idx, spk, w_i, w_r,dphi,noise,  ncut,smoothing_size, plotting=plotting)

    args=(rr, pp,dd, nn, zz, hanz, dx, dy, dp, dn)

    print(plotting)
    arg2 =getSVDPolar(idx, spk, ncut, args, plotting=plotting, more_smooth=more_smooth, smoothing_size=smoothing_size)#
    print(arg2)
    if plotting:
        print(plotting)
        v, u, m_dp, s, f=arg2
        return v, u, m_dp, s, f, (dp, zz)  #

    else:
        print('no plotting')
        v, u, m_dp, s, f=arg2

        return v, u, m_dp, s, f


def computeNonlin(f, rho, phi, dphi):
    """Function for computeNonlin.

    Args:
        f: Input value for this operation.
        rho: Input value for this operation.
        phi: Input value for this operation.
        dphi: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    nonlinres=np.nan_to_num(f(rho, phi, dphi))#*dP.reshape(-1))
    return nonlinres


def computeNonlinMultiplicative(rh, ph, mean_rho, mean_phi, dphi,dp, m_dphi, ncut):
    """Function for computeNonlinMultiplicative.

    Args:
        rh: Input value for this operation.
        ph: Input value for this operation.
        mean_rho: Input value for this operation.
        mean_phi: Input value for this operation.
        dphi: Input value for this operation.
        dp: Input value for this operation.
        m_dphi: Input value for this operation.
        ncut: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    rhospace = np.linspace(0, np.max(rh), ncut)
    phispace = np.linspace(0, np.max(ph), ncut)

    R = np.interp(rh, rhospace, mean_rho).reshape(-1)
    P = np.interp(ph, phispace, mean_phi).reshape(-1)
    dP= np.interp(dphi, dp[::-1], m_dphi[::-1]).reshape(-1)
    nonlinres = R * P * dP

    return nonlinres


from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings


def sigmoid(X1, *args):#a, b, w1, w2, w3): # Sigmoid A With Offset
    """Function for sigmoid.

    Args:
        X1: Input value for this operation.
        args: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    a=args[0]
    b=args[1]
    c=args[2]
    w=np.array([args[i] for i in range(3, len(args))]).reshape(1, -1)
    x=np.dot(w, X1.T)
    return  (c / (1.0 + np.exp(-a * (x-b)))).reshape(-1)


def relu(X1,*args): # Sigmoid A With Offset
    """Function for relu.

    Args:
        X1: Input value for this operation.
        args: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    a = args[0]
    w = np.array([args[i] for i in range(1, len(args))]).reshape(1, -1)
    x = np.dot(w, X1.T)
    return np.clip(x,a, 10000).reshape(-1)


from sklearn.metrics import r2_score, explained_variance_score

def fitnonlin(X1, y_train, func):
    """Function for fitnonlin.

    Args:
        X1: Input value for this operation.
        y_train: Input value for this operation.
        func: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    def sumOfSquaredError(parameterTuple):
        """Function for sumOfSquaredError.

        Args:
            parameterTuple: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        warnings.filterwarnings("ignore")  # do not print warnings by genetic algorithm
        val = func(X1, *parameterTuple)
        return np.sum((y_train - val) ** 2.0)

    def generate_Initial_Parameters(nb_params):
        """Function for generate Initial Parameters.

        Args:
            nb_params: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        maxX = np.max(X1)
        minX = np.min(X1)
        maxY = np.max(y_train)
        minY = np.min(y_train)

        parameterBounds = []
        if func==sigmoid:
            print('sigmoid')
            parameterBounds.append([0, 2])  # search bounds for a
            parameterBounds.append([0, 100])  # search bounds for b
            parameterBounds.append([0, 100]) # search bounds for c
            for n in range(nb_params):
                parameterBounds.append([-1e1, 1e1])
        elif func==relu:
            print('relu')
            parameterBounds.append([-2 * maxX, 2 * maxX])  # search bounds for a
            for n in range(nb_params):
                parameterBounds.append([-1e1, 1e1])
        print(parameterBounds)
        
        # OPTIMIZED: workers=-1 utilizes all available CPU threads for the genetic algorithm
        result = differential_evolution(sumOfSquaredError, parameterBounds, seed=2, workers=-1, updating='deferred')
        return result.x

    geneticParameters = generate_Initial_Parameters(X1.shape[1])
    print('geneticParameters')
    print(geneticParameters)

    # curve fit the test data
    fittedParameters, pcov = curve_fit(func, X1, y_train, geneticParameters, method='trf')

    print('Parameters', fittedParameters)

    modelPredictions = func(X1, *fittedParameters)
    if func==sigmoid:
        dx=np.linspace(-1, 5, 100)

    absError = modelPredictions - y_train

    SE = np.square(absError)  # squared errors
    MSE = np.mean(SE)  # mean squared errors
    RMSE = np.sqrt(MSE)  # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (np.var(absError) / np.var(y_train))
    print('RMSE:', RMSE)
    print('R-squared:', Rsquared)

    return fittedParameters, pcov, modelPredictions


def PlotR2scoreAnalysis(path, neuron_pos, respcorr):
    """Function for PlotR2scoreAnalysis.

    Args:
        path: Input value for this operation.
        neuron_pos: Input value for this operation.
        respcorr: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    r=np.load(path)
    pearsons=r[:, 1]
    r2=r[:, 0]
    print(np.mean(r2), np.mean(pearsons))
    print(np.mean(r2[respcorr>=0.2]), np.mean(pearsons[respcorr>=0.2]))
    print(np.median(r2), np.median(pearsons))
    plt.figure()
    plt.rcParams['axes.facecolor']='none'
    plt.scatter(neuron_pos[:, 1], neuron_pos[:, 0], s=5, c=r2, vmax=0.1, vmin=0, cmap='Greys')
    plt.colorbar()
    plt.title('R2')
    plt.figure()
    plt.rcParams['axes.facecolor']='none'
    plt.scatter(neuron_pos[:, 1], neuron_pos[:, 0], s=5, c=pearsons, vmax=0.3, vmin=0, cmap='Greys')
    plt.colorbar()
    plt.title('pearsons')

    plt.figure()
    plt.rcParams['axes.facecolor']='none'
    plt.hist(pearsons, bins=30)
    plt.hist(pearsons[respcorr>=0.2], bins=30)
    plt.title('pearsons')
    plt.tight_layout()

    plt.figure()
    plt.rcParams['axes.facecolor']='none'
    plt.hist(r2, bins=30)
    plt.hist(r2[respcorr >= 0.2], bins=30)
    plt.title('r2')
    plt.tight_layout()

    return r, np.where(r2>=0.2)


def calculate_spike_triggered_covariance(spikes, stimulus, tau):
    """Function for calculate spike triggered covariance.

    Args:
        spikes: Input value for this operation.
        stimulus: Input value for this operation.
        tau: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    X, T = stimulus.shape

    # Boolean indexing to avoid massive for-loop iteration over non-spikes
    spike_indices = np.where(spikes[tau:T] != 1)[0] + tau
    num_segments = len(spike_indices)
    
    if num_segments == 0:
        return np.zeros((X * tau, X * tau))

    # Pre-allocate array to avoid np.stack RAM spike
    segments = np.empty((num_segments, X * tau), dtype=np.float32)
    
    for i, t in enumerate(spike_indices):
        segments[i] = (spikes[t] * stimulus[:, t-tau:t]).reshape(-1)

    mean_segment = np.mean(segments, axis=0)
    centered_segments = segments - mean_segment

    covariance_matrix = np.zeros((X * tau, X * tau), dtype=np.float32)
    
    # TUNE THIS: Increase slice_size to push your GPU harder. 
    # 2000-5000 is usually a great sweet spot for modern GPUs to max out cores without OOM.
    slice_size = 2000 
    
    device = resolve_compute_device(prefer_gpu=True)
    with cpu_threadpool_scope(), torch.no_grad():
        # Keep main tensor in pinned CPU memory. This allows max-speed async transfers to GPU 
        # without overflowing VRAM by trying to load the whole dataset at once.
        cent_seg_t = torch.as_tensor(centered_segments, dtype=torch.float32)
        if device == "cuda":
            cent_seg_t = cent_seg_t.pin_memory()
        
        for start_i in range(0, X * tau, slice_size):
            print(f"Processing row block: {start_i}")
            end_i = min(start_i + slice_size, X * tau)
            
            # Move only the specific chunk to GPU as fast as possible
            slice_i = cent_seg_t[:, start_i:end_i].to(device, non_blocking=device == "cuda")

            for start_j in range(start_i, X * tau, slice_size):
                end_j = min(start_j + slice_size, X * tau)
                slice_j = cent_seg_t[:, start_j:end_j].to(device, non_blocking=device == "cuda")

                # GPU Matrix Math
                partial_cov = slice_i.t().mm(slice_j) / (num_segments - 1)
                cov_cpu = partial_cov.cpu().numpy()

                covariance_matrix[start_i:end_i, start_j:end_j] = cov_cpu
                if start_i != start_j:
                    covariance_matrix[start_j:end_j, start_i:end_i] = cov_cpu.T
                
                # Strict RAM/VRAM management: explicitly delete the inner chunk
                del slice_j 

            # Strict RAM/VRAM management: explicitly delete the outer chunk
            del slice_i 

    return covariance_matrix


def on_pick(event):
    """Function for on pick.

    Args:
        event: Input value for this operation.
    """
    artist = event.artist
    xmouse, ymouse = event.mouseevent.xdata, event.mouseevent.ydata
    x, y = artist.get_xdata(), artist.get_ydata()
    ind = event.ind
    print ('idx:', event.artist)


def gaussian(x, mu, sig):
    """Function for gaussian.

    Args:
        x: Input value for this operation.
        mu: Input value for this operation.
        sig: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )

def sigma_func(x, a, b):
    """Function for sigma func.

    Args:
        x: Input value for this operation.
        a: Input value for this operation.
        b: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    return (1.0 / (1.0 + np.exp(-a * (x - b))))


def create_fake_cell( w_i_downsampled, w_r_downsampled, pos_angle_scale, phase_t_shift, phase=np.pi/3, thresh=0.075, ncut=30, dt=5400, plotting=False):
    """Function for create fake cell.

    Args:
        w_i_downsampled: Input value for this operation.
        w_r_downsampled: Input value for this operation.
        pos_angle_scale: Input value for this operation.
        phase_t_shift: Input value for this operation.
        phase: Input value for this operation.
        thresh: Input value for this operation.
        ncut: Input value for this operation.
        dt: Input value for this operation.
        plotting: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    x, y, o, s=pos_angle_scale
    w_i = w_i_downsampled[:dt, x, y, o, s]  
    w_r = w_r_downsampled[:dt, x, y, o, s]  

    # SUPER-VECTORIZED: Bypassing the pure python for-loop and cart2pol function entirely
    # This computes polar coords over the whole array instantly in C.
    rho = np.hypot(w_r, w_i)
    phi = np.arctan2(w_i, w_r)
    
    dphi = np.diff(phi, prepend=0)
    dphi[abs(dphi) >= 3] = np.nan
    nans, _x = nan_helper(dphi)
    dphi[nans] = np.interp(_x(nans), _x(~nans), dphi[~nans])

    rhospace = np.linspace(0, 0.2, ncut)
    phispace = np.linspace(0, 2*np.pi, ncut)
    dphispace = np.linspace(-180, 180, ncut)

    interp_grid=(((1+np.cos(phispace+phase)).reshape(-1, 1)*sigma_func(rhospace, 100, thresh).reshape(1, -1)).reshape(-1, 1)*(2.5*gaussian(dphispace, phase_t_shift, 150)).reshape(1,-1)).reshape(ncut, ncut, ncut)
    interp_grid=np.clip(interp_grid, 0, None)
    xx, yy, tt = np.meshgrid(rhospace, phispace, dphispace)

    if plotting:
        plt.figure()
        plt.pcolormesh(rhospace, phispace, np.nanmean(interp_grid, axis=2)[:-1, :-1], cmap='coolwarm')
        plt.plot(rhospace, phispace, color='k', ls='none')
        plt.colorbar()

    try:
        f = interpolate.LinearNDInterpolator(np.stack((xx.flatten(), yy.flatten(), tt.flatten())).T,
                                             interp_grid.flatten().T)
    except ValueError as e:
        print(e)

    nonlinres = computeNonlin(f, rho, phi, dphi)
    return nonlinres


def PlotSelfCorrelation(w_c_downsampled, neuron_pos, pos_ori, visual_coverage, screen_ratio, sigmas, frequencies, ns=4, nf=1, nx=None, ny=None, n_orientations=None, nx_full=None, ny_full=None):
    """Function for PlotSelfCorrelation.

    Args:
        w_c_downsampled: Input value for this operation.
        neuron_pos: Input value for this operation.
        pos_ori: Input value for this operation.
        visual_coverage: Input value for this operation.
        screen_ratio: Input value for this operation.
        sigmas: Input value for this operation.
        frequencies: Input value for this operation.
        ns: Input value for this operation.
        nf: Input value for this operation.
        nx: Input value for this operation.
        ny: Input value for this operation.
        n_orientations: Input value for this operation.
        nx_full: Input value for this operation.
        ny_full: Input value for this operation.
    """
    x, y, o, s = pos_ori
    if nx is None or ny is None:
        nx, ny = coarse_grid_dimensions(w_c_downsampled.shape[1], w_c_downsampled.shape[2])
    if n_orientations is None:
        n_orientations = w_c_downsampled.shape[3]

    rfs = PearsonCorrelationPinkNoise(
        w_c_downsampled.reshape(w_c_downsampled.shape[0], -1),
        w_c_downsampled[:, x, y, o, s].reshape(w_c_downsampled.shape[0], -1),
        neuron_pos, 
        nx, 
        ny, 
        ns,
        nf,
        visual_coverage,
        screen_ratio,
        sigmas,
        frequencies,
        n_orientations=n_orientations,
    )
    
    rfs_idx = rfs[0][:, :, :, :, 0] if rfs[0].ndim > 5 else rfs[0]
    
    fig, ax = plt.subplots(n_orientations, ns)
    if n_orientations == 1 and ns == 1:
        ax = np.array([[ax]])
    elif n_orientations == 1 or ns == 1:
        ax = ax.reshape(n_orientations, ns)

    if nx_full is None:
        nx_full = w_c_downsampled.shape[1] * 5
    if ny_full is None:
        ny_full = w_c_downsampled.shape[2] * 5
    r = skimage.transform.resize(
        rfs_idx,
        (nx_full, ny_full, n_orientations, ns),
        anti_aliasing=True,
    )
    vmax = 1
    vmin = -vmax
    
    for i in range(n_orientations):
        for j in range(ns):
            ax[i, j].imshow(r[:, :, i, j].T, vmin=vmin, vmax=vmax, cmap='coolwarm')


def Plot_RF(rfs_idx, ns=4, title='', n_orientations=None):


    """Function for Plot RF.

    Args:
        rfs_idx: Input value for this operation.
        ns: Input value for this operation.
        title: Input value for this operation.
        n_orientations: Input value for this operation.
    """
    if n_orientations is None:
        n_orientations = rfs_idx.shape[2]
    fig, ax = plt.subplots(n_orientations, ns)
    if n_orientations == 1 and ns == 1:
        ax = np.array([[ax]])
    elif n_orientations == 1 or ns == 1:
        ax = ax.reshape(n_orientations, ns)
    plt.title(title)
    cc = 0
    vmax = np.max(rfs_idx)
    vmin = -vmax
    for i in range(n_orientations):
        for j in range(ns):
            ax[i, j].imshow(rfs_idx[:, :, i, j].T, vmin=vmin, vmax=vmax, cmap='coolwarm')
            ax[i, j].set_aspect("equal")
            ax[i, j].set_xticks(np.linspace(0, rfs_idx.shape[0] - 1, 3).astype(int))
            ax[i, j].set_yticks(np.linspace(0, rfs_idx.shape[1] - 1, 3).astype(int))


def gaus(x,a,x0,sigma, offset):
    """Function for gaus.

    Args:
        x: Input value for this operation.
        a: Input value for this operation.
        x0: Input value for this operation.
        sigma: Input value for this operation.
        offset: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    return (a*exp(-(x-x0)**2/(2*sigma**2))) + offset


def fit_gaussian_params(x_m_phi, plotting=False):
    """Function for fit gaussian params.

    Args:
        x_m_phi: Input value for this operation.
        plotting: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    x=x_m_phi[0]
    y=abs(x_m_phi[1])
    n = len(x)  # the number of data
    mean = sum(x * y) / n  # note this correction
    sigma = sum(y * (x - mean) ** 2) / n  # note this correction
    popt, pcov = curve_fit(gaus, x, y, p0=[1, mean, sigma, 0])
    if plotting:
        plt.figure()
        plt.plot(x, y)
        plt.plot(x,gaus(x,*popt))
    return popt


import scipy
import scipy.cluster.hierarchy as sch


def cluster_corr(corr_array, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly
    correlated variables are next to eachother

    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max() / 2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold,
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)

    if not inplace:
        corr_array = corr_array.copy()

    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx], idx_to_cluster_array


def FEVE(gt, pred):
    """Function for FEVE.

    Args:
        gt: Input value for this operation.
        pred: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    sig=np.mean(np.var(gt, axis=0))
    absError = pred - np.mean(gt, axis=0)
    SE = np.square(absError)  # squared errors
    MSE = np.mean(SE)
    num=MSE-np.square(sig)
    denom=np.var(gt)-np.square(sig)
    feve=1-(num/denom)
    return feve


def rolling_avg(arr, win):
    """Function for rolling avg.

    Args:
        arr: Input value for this operation.
        win: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    import scipy.signal as sig
    kernal = np.ones(win, dtype=('float'))
    padsize = arr.shape[0] + win * 2
    mov_pad = np.zeros([padsize], dtype=('float'))
    mov_pad[win:(padsize-win)] = arr
    mov_ave = sig.fftconvolve(mov_pad, kernal) / win
    return mov_ave


def hanningconvnd(interp_grid, n):
    """Function for hanningconvnd.

    Args:
        interp_grid: Input value for this operation.
        n: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    kern = np.hanning(n).reshape(-1, 1)
    kern = kern * kern.T
    kern=kern[:, :, np.newaxis]* kern.T
    kern /= kern.sum()  # normalize the kernel weights to sum to 1
    hanning = ndimage.convolve(interp_grid, kern)
    return hanning


from scipy.interpolate import NearestNDInterpolator,LinearNDInterpolator
from scipy.ndimage import gaussian_filter


def interpolateDatand(z, dx, dy, dp, dx_h, dy_h, dp_h, ncut, smooth=True):
    """Function for interpolateDatand.

    Args:
        z: Input value for this operation.
        dx: Input value for this operation.
        dy: Input value for this operation.
        dp: Input value for this operation.
        dx_h: Input value for this operation.
        dy_h: Input value for this operation.
        dp_h: Input value for this operation.
        ncut: Input value for this operation.
        smooth: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    x_grid, y_grid, p_grid,xh_grid, yh_grid, ph_grid = np.meshgrid(dx, dy, dp,dx_h, dy_h, dp_h)

    # get known values to set the interpolator
    mask = [~np.isnan(z)][0]
    x = x_grid[mask].reshape(-1)
    y = y_grid[mask].reshape(-1)
    p = p_grid[mask].reshape(-1)
    xh = xh_grid[mask].reshape(-1)
    yh = yh_grid[mask].reshape(-1)
    ph = ph_grid[mask].reshape(-1)
    points = np.array([x, y, p, xh, yh, ph]).T
    values = z[mask].reshape(-1)

    # generate interpolated grid data
    interp = NearestNDInterpolator(list(zip(x, y, p, xh, yh, ph)), values)

    interp_grid = interp(x_grid, y_grid, p_grid, xh_grid, yh_grid, ph_grid)
    if smooth:
        interp_grid=gaussian_filter(interp_grid, sigma=ncut)
    interp = LinearNDInterpolator(list(zip(x_grid.reshape(-1), y_grid.reshape(-1), p_grid.reshape(-1), xh_grid.reshape(-1), yh_grid.reshape(-1), ph_grid.reshape(-1))), interp_grid)

    return interp_grid, interp


def SinCosPlot3( spk, w_i, w_r, dphi, w_i_inhib, w_r_inhib, dphi_inhib, ncut, smoothing_size, plotting=True):
    """Function for SinCosPlot3.

    Args:
        spk: Input value for this operation.
        w_i: Input value for this operation.
        w_r: Input value for this operation.
        dphi: Input value for this operation.
        w_i_inhib: Input value for this operation.
        w_r_inhib: Input value for this operation.
        dphi_inhib: Input value for this operation.
        ncut: Input value for this operation.
        smoothing_size: Input value for this operation.
        plotting: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    sin=w_i.reshape(-1,1)
    cos=w_r.reshape(-1,1)
    dphi=dphi.reshape(-1, 1)

    sin_h = w_i_inhib.reshape(-1, 1)
    cos_h = w_r_inhib.reshape(-1, 1)
    dphi_h = dphi_inhib.reshape(-1, 1)


    Y = spk.reshape(-1, 1)
    Y_past=np.zeros(Y.shape)
    Y_past[:-5]=Y[5:]

    nCut=ncut
    data=np.concatenate([cos, sin, dphi, sin_h, cos_h, dphi_h], axis=1)#, columns=['sin', 'cos', 'sig'])
    histo, edges=np.histogramdd(data, bins=ncut, density=False, weights=Y[:, 0])
    histo_, edges_ = np.histogramdd(data, bins=ncut)
    z=histo/histo_
    dx=(edges[0][:-1] + edges[0][1:]) / 2
    dy=(edges[1][:-1] + edges[1][1:]) / 2
    dp = (edges[2][:-1] + edges[2][1:]) / 2
    dx_h = (edges[3][:-1] + edges[3][1:]) / 2
    dy_h =(edges[4][:-1] + edges[4][1:]) / 2
    dp_h = (edges[5][:-1] + edges[5][1:]) / 2
    interp_grid, interp=interpolateDatand(z, dx, dy, dp, dx_h, dy_h, dp_h)

    return interp_grid,z, dx, dy, dp, dx_h, dy_h, dp_h


def getNonLinearModel3(idx, spk, w_i, w_r, dphi, noise, ncut, smoothing_size, plotting=False, more_smooth=False):

    """Function for getNonLinearModel3.

    Args:
        idx: Input value for this operation.
        spk: Input value for this operation.
        w_i: Input value for this operation.
        w_r: Input value for this operation.
        dphi: Input value for this operation.
        noise: Input value for this operation.
        ncut: Input value for this operation.
        smoothing_size: Input value for this operation.
        plotting: Input value for this operation.
        more_smooth: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    interp_grid,z, dx, dy, dp, dx_h, dy_h, dp_h= SinCosPlot3(idx, spk, w_i, w_r,dphi,noise,  ncut,smoothing_size, plotting=plotting)
    x_grid, y_grid, p_grid, xh_grid, yh_grid, ph_grid = np.meshgrid(dx, dy, dp, dx_h, dy_h, dp_h)
    f = interpolate.LinearNDInterpolator(np.stack((x_grid.flatten(), y_grid.flatten(), p_grid.flatten(), xh_grid.flatten(), yh_grid.flatten(), ph_grid.flatten())).T, interp_grid.flatten().T)
    return f


def getmetrics(x, y, n, frames_per_minute=None):
    """Function for getmetrics.

    Args:
        x: Input value for this operation.
        y: Input value for this operation.
        n: Input value for this operation.
        frames_per_minute: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    if frames_per_minute is None:
        raise ValueError(
            "var_exp requires frames_per_minute from the configured stimulus "
            "frame rate."
        )
    frames_per_minute = int(frames_per_minute)
    ev = explained_variance_score(
        np.mean(y.reshape(n, frames_per_minute), axis=0),
        x,
        multioutput='uniform_average',
    )
    feve = FEVE(y.reshape(n, frames_per_minute), x)
    cc = np.corrcoef(np.mean(y.reshape(n, frames_per_minute), axis=0), x)
    return feve, ev,cc

def getpolar(cos, sin):
    """Function for getpolar.

    Args:
        cos: Input value for this operation.
        sin: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    rho = np.sqrt(cos ** 2 + sin ** 2)
    temp_phi = np.arctan2(sin, cos)
    shift = (2 * np.pi) * (temp_phi < 0)
    phi = np.arctan2(sin, cos) + shift
    return rho, phi

import scipy.fftpack
from scipy.signal import find_peaks


def gaussian_smooth(y, sigma = 2):
    """Smooth a one-dimensional model factor with a normalized Gaussian kernel.

    Args:
        y: One-dimensional numeric factor sampled on a model-feature axis.
        sigma: Gaussian width in factor samples. Must be positive.

    Returns:
        np.ndarray: Smoothed factor with the same intended axis length as ``y``.
    """
    kernel_size = 2 * int(3 * sigma) + 1
    gaussian_kernel = np.exp(-0.5 * (np.linspace(-3, 3, kernel_size) / sigma) ** 2)
    gaussian_kernel /= gaussian_kernel.sum()  # Normalisation

    # Convolution (conserve intensity)
    y_smooth = np.convolve(y, gaussian_kernel, mode='full')
    y_smooth=y_smooth[int((gaussian_kernel.shape[0]-1)/2)-1:-int((gaussian_kernel.shape[0]-1)/2)-1]
    return y_smooth

def approx_Matrix(X, plotting=False):
    """Fit a rank-one non-negative approximation used by the response model.

    Args:
        X: Non-negative three-dimensional response histogram.
        plotting: Whether to create diagnostic Matplotlib figures.

    Returns:
        tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]: The
        rank-one reconstruction and one factor for each input axis.
    """
    X=np.clip(X, 0, None)
    if not np.any(X):
        factors = tuple(np.zeros(length, dtype=float) for length in X.shape)
        return np.zeros_like(X, dtype=float), factors
    model1 = NMF(n_components=1, init='random', random_state=42)
    U1 = model1.fit_transform(X.reshape(X.shape[0], -1))[:, 0]

    model2 = NMF(n_components=1, init='random', random_state=42)
    U2 = model2.fit_transform(X.transpose(1, 0, 2).reshape(X.shape[1], -1))[:, 0]

    model3 = NMF(n_components=1, init='random', random_state=42)
    U3 = model3.fit_transform(X.transpose(2, 0, 1).reshape(X.shape[2], -1))[:, 0]
    if plotting:
        plt.figure()
        plt.plot(U1)
        plt.figure()
        plt.plot(U2)
        plt.figure()
        plt.plot(U3)

    U1=gaussian_smooth(U1, 1)
    U2 = gaussian_smooth(U2, 1)
    U3 = gaussian_smooth(U3, 1)

    if plotting:
        plt.figure()
        plt.plot(U1)
        plt.figure()
        plt.plot(U2)
        plt.figure()
        plt.plot(U3)
    # Approximation
    X_approx = np.einsum('i,j,k -> ijk', U1, U2, U3)

    return X_approx, (U1, U2, U3)

def approx_Matrix2(X, smoothing_factor=0.75, plotting=False):
    """Fit a rank-one non-negative CP approximation with safe empty handling.

    Args:
        X: Non-negative three-dimensional response histogram.
        smoothing_factor: Optional Gaussian width applied to returned factors.
            Pass ``None`` to preserve raw CP factors.
        plotting: Reserved compatibility flag for legacy callers.

    Returns:
        tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]: The
        reconstruction and one-dimensional factors for each input axis.
    """
    from tensorly.decomposition import non_negative_parafac

    X = np.clip(np.asarray(X, dtype=float), 0.0, None)
    if not np.any(X):
        factors = tuple(np.zeros(length, dtype=float) for length in X.shape)
        return np.zeros_like(X), factors

    weights, factors = non_negative_parafac(X, rank=1, init='random', normalize_factors=False)
    U1, U2, U3 = factors

    U1=U1.reshape(-1)
    U2=U2.reshape(-1)
    U3=U3.reshape(-1)

    if smoothing_factor!=None:
        U1 = gaussian_smooth(U1, smoothing_factor)
        U2 = gaussian_smooth(U2, smoothing_factor)
        U3 = gaussian_smooth(U3, smoothing_factor)

    X_approx = np.einsum('i,j,k -> ijk', U1, U2, U3)
    return X_approx,  (U1, U2, U3)
    

def getPhiRho(spk, w_i, w_r, dphi, w_i_inhib, w_r_inhib, dphi_inhib, ncut=20, plotting=True, sigma=7):
    """Estimate a smoothed firing-response surface in rho/phase/drift space.

    Args:
        spk: Trial-by-frame neural responses with shape ``(trials, frames)``.
        w_i: Imaginary selected-wavelet trace, ``(frames,)`` or ``(frames, 1)``.
        w_r: Real selected-wavelet trace with the same frame axis as ``w_i``.
        dphi: Instantaneous phase-drift trace in radians per frame.
        w_i_inhib: Imaginary inhibitory-wavelet trace.
        w_r_inhib: Real inhibitory-wavelet trace.
        dphi_inhib: Inhibitory phase-drift trace.
        ncut: Number of bins on each response-surface axis.
        plotting: Whether to construct legacy diagnostic figures.
        sigma: Histogram-smoothing width in bins.

    Returns:
        tuple: Smoothed response surfaces, bin centers, histogram data, and
        interpolation helpers consumed by the nonlinear model. Empty histogram
        cells are represented safely as zero response rather than NaN.
    """
    sin = w_i.reshape(-1, 1)
    cos = w_r.reshape(-1, 1)
    dphi = dphi.reshape(-1, 1)
    rho, phi = getpolar(sin, cos)

    sin_h = w_i_inhib.reshape(-1, 1)
    cos_h = w_r_inhib.reshape(-1, 1)
    dphi_h = dphi_inhib.reshape(-1, 1)
    rho_h, phi_h = getpolar(sin_h, cos_h)

    a = abs(max(rho.min(), rho.max()))
    c = abs(max(phi.min(), phi.max()))
    if a == 0:
        a = 0.3
    b = abs(max(dphi.min(), dphi.max()))
    d = abs(max(cos.min(), cos.max()))
    e = abs(max(sin.min(), sin.max()))
    d = max(d, e)
    if b == 0:
        b = 1.0
        
    E = [np.linspace(0, a, ncut + 1), np.linspace(0, c, ncut + 1), np.linspace(-b, b, ncut + 1)]
    Ecs = [np.linspace(-d, d, ncut + 1), np.linspace(-d, d, ncut + 1)]
    
    # ---------------------------------------------------------
    # OPTIMIZATION 1: Multithreaded histogram loops
    # ---------------------------------------------------------
    n_spk = spk.shape[0]
    Hcs = np.empty((n_spk, ncut, ncut))
    Hcs_ = np.empty((n_spk, ncut, ncut))
    H = np.empty((n_spk, ncut, ncut, ncut))
    H_ = np.empty((n_spk, ncut, ncut, ncut))

    data = np.concatenate([rho, phi, dphi], axis=1)
    datacs = np.concatenate([cos, sin], axis=1)

    # Worker function to run histograms outside GIL constraints
    def compute_hist_1(i):
        """Function for compute hist 1.

        Args:
            i: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        Y = spk[i].reshape(-1, 1)
        _hcs, _ = np.histogramdd(datacs, bins=Ecs, density=False, weights=Y[:, 0])
        _hcs_, _ = np.histogramdd(datacs, bins=Ecs)
        _h, _ = np.histogramdd(data, bins=E, density=False, weights=Y[:, 0])
        _h_, _ = np.histogramdd(data, bins=E)
        return i, _hcs, _hcs_, _h, _h_

    # Dynamically allocate threads based on your CPU
    threads = cpu_worker_count(cap=min(8, n_spk))
    with cpu_threadpool_scope(threads), ThreadPoolExecutor(max_workers=threads) as executor:
        for i, _hcs, _hcs_, _h, _h_ in executor.map(compute_hist_1, range(n_spk)):
            Hcs[i] = _hcs
            Hcs_[i] = _hcs_
            H[i] = _h
            H_[i] = _h_

    Hcs = np.nanmean(Hcs, axis=0)
    Hcs_ = np.nanmean(Hcs_, axis=0)
    Zcs = np.divide(Hcs, Hcs_, out=np.zeros_like(Hcs), where=Hcs_ > 0)
    
    H = np.nanmean(H, axis=0)
    H_ = np.nanmean(H_, axis=0)
    
    H = np.concatenate((H, H, H), axis=1)
    H_ = np.concatenate((H_, H_, H_), axis=1)
    smoothed_signal = hanningconv3d(H, sigma)
    smoothed_counts = hanningconv3d(H_, sigma)
    Z = np.divide(
        smoothed_signal,
        smoothed_counts,
        out=np.zeros_like(smoothed_signal),
        where=smoothed_counts > 0,
    )
    Z = Z[:, :int(Z.shape[1]/3), :]
    
    xedges = E[0]
    yedges = E[1]
    zedges = E[2]
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    zcenters = (zedges[:-1] + zedges[1:]) / 2

    mask = np.where(~np.isnan(Z))
    interp = NearestNDInterpolator(np.transpose((xcenters[mask[0]], ycenters[mask[1]], zcenters[mask[2]])), Z[mask])
    indices = np.indices(Z.shape)
    indices = np.stack([xcenters[indices[0]], ycenters[indices[1]], zcenters[indices[2]]])
    filled_data0 = interp(*indices)
    filled_data2, plot = approx_Matrix2(filled_data0, None, plotting=plotting)
    plots = [plot[0], np.append(plot[1], plot[1][0]), plot[2]]

    filled_data = np.concatenate((filled_data0, filled_data0, filled_data0), axis=1)

    d_val = zcenters[np.argmax(plot[2])]
    pp = plot[1]
    rr = plot[0]
    HMP = xcenters[np.argmin(abs(rr - (np.max(rr)/2)))]
    cv = circular_variance(np.linspace(0, 360, 20), pp.reshape(-1, 1))
    complexity = 1 - cv[1][0]

    if plotting:
        fig = plt.figure(figsize=(16, 6))
        gs = gridspec.GridSpec(2, 2, figure=fig)
        plt.rcParams.update({
            "font.size": 8,
            "svg.fonttype": "none"
        })

        inner_gs0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0])
        ax1_0 = fig.add_subplot(inner_gs0[0], projection='3d')
        a1_1 = fig.add_subplot(inner_gs0[1])
        threshold = np.nanpercentile(Zcs.flatten(), 95)
        m = a1_1.imshow(Zcs, vmin=0, vmax=threshold, cmap='bone_r')
        a1_1.set_xticks(np.arange(20))
        a1_1.set_xticklabels(10*(0.5 * (Ecs[0][1:] + Ecs[0][:-1])).astype(int) / 10, rotation=45)
        a1_1.set_yticks(np.arange(20))
        a1_1.set_yticklabels(10*(0.5 * (Ecs[0][1:] + Ecs[0][:-1])).astype(int) / 10)
        a1_1.set_xlabel('Sine wavelet (a.u)')
        a1_1.set_ylabel('Cosine wavelet (a.u)')
        a1_1.set_title('Firing rate histogram')
        fig.colorbar(m, ax=a1_1)

        colors = []
        for i in np.arange(8000, 9000, 1):
            c_val = np.mean(spk, axis=0)[i]
            color = plt.cm.bone_r(255 * c_val / 100)
            colors.append(color)
            ax1_0.scatter(rho[i - 1:i + 1], phi[i - 1:i + 1], dphi[i - 1:i + 1], s=10, color=color)
            ax1_0.set_xlabel('Amplitude (a.u)')
            ax1_0.set_ylabel('Phase (rad)')
            ax1_0.set_zlabel('Drift (rad/s)')
            ax1_0.set_title('Firing rate trajectory')
        fig.colorbar(m, ax=ax1_0)

        inner_gs1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 0])
        ax3_0 = fig.add_subplot(inner_gs1[0])
        ax3_1 = fig.add_subplot(inner_gs1[1])
        ax3_2 = fig.add_subplot(inner_gs1[2])
        ax3_0.plot(xcenters, np.nanmean(filled_data0, axis=(2, 1)))
        ax3_0.set_title('rho')
        ax3_1.plot(ycenters, np.nanmean(filled_data0, axis=(2, 0)))
        ax3_1.set_title('phi')
        ax3_1.set_ylim(bottom=0)
        ax3_2.plot(zcenters, np.nanmean(filled_data0, axis=(0, 1)))
        ax3_2.set_title('dphi')

        inner_gs2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, 1])
        ax4_0 = fig.add_subplot(inner_gs2[0])
        ax4_1 = fig.add_subplot(inner_gs2[1])
        ax4_2 = fig.add_subplot(inner_gs2[2])
        m1 = ax4_0.imshow(np.nanmean(filled_data0, axis=2).T, cmap='bone_r')
        ax4_0.set_xticks(range(20))
        ax4_0.set_xticklabels((xcenters * 100).astype(int) / 100, rotation=90)
        ax4_0.set_yticks(range(20))
        ax4_0.set_yticklabels((ycenters * 100).astype(int) / 100)
        ax4_0.set_xlabel('rho')
        ax4_0.set_ylabel('phi')
        fig.colorbar(m1, ax=ax4_0)

        m2 = ax4_1.imshow(np.nanmean(filled_data0, axis=1).T, cmap='bone_r')
        ax4_1.set_xticks(range(20))
        ax4_1.set_xticklabels((xcenters * 100).astype(int) / 100, rotation=90)
        ax4_1.set_yticks(range(20))
        ax4_1.set_yticklabels((zcenters * 100).astype(int) / 100)
        ax4_1.set_xlabel('rho')
        ax4_1.set_ylabel('dphi')
        fig.colorbar(m2, ax=ax4_1)

        m3 = ax4_2.imshow(np.nanmean(filled_data0, axis=0).T, cmap='bone_r')
        ax4_2.set_xticks(range(20))
        ax4_2.set_xticklabels((ycenters * 100).astype(int) / 100, rotation=90)
        ax4_2.set_yticks(range(20))
        ax4_2.set_yticklabels((zcenters * 100).astype(int) / 100)
        ax4_2.set_xlabel('phi')
        ax4_2.set_ylabel('dphi')
        fig.colorbar(m3, ax=ax4_2)

        inner_gs3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 1])
        ax5_0 = fig.add_subplot(inner_gs3[0])
        ax5_1 = fig.add_subplot(inner_gs3[1])
        ax5_2 = fig.add_subplot(inner_gs3[2])
        m4 = ax5_0.imshow(np.nanmean(Z, axis=2).T, cmap='coolwarm')
        ax5_0.set_xticks(range(20))
        ax5_0.set_xticklabels((xcenters * 100).astype(int) / 100, rotation=90)
        ax5_0.set_yticks(range(20))
        ax5_0.set_yticklabels((ycenters * 100).astype(int) / 100)
        ax5_0.set_xlabel('rho')
        ax5_0.set_ylabel('phi')
        fig.colorbar(m4, ax=ax5_0)

        m5 = ax5_1.imshow(np.nanmean(Z, axis=1).T, cmap='coolwarm')
        ax5_1.set_xticks(range(20))
        ax5_1.set_xticklabels((xcenters * 100).astype(int) / 100, rotation=90)
        ax5_1.set_yticks(range(20))
        ax5_1.set_yticklabels((zcenters * 100).astype(int) / 100)
        ax5_1.set_xlabel('rho')
        ax5_1.set_ylabel('dphi')
        fig.colorbar(m5, ax=ax5_1)

        m6 = ax5_2.imshow(np.nanmean(Z, axis=0).T, cmap='coolwarm')
        ax5_2.set_xticks(range(20))
        ax5_2.set_xticklabels((ycenters * 100).astype(int) / 100, rotation=90)
        ax5_2.set_yticks(range(20))
        ax5_2.set_yticklabels((zcenters * 100).astype(int) / 100)
        ax5_2.set_xlabel('phi')
        ax5_2.set_ylabel('dphi')
        fig.colorbar(m6, ax=ax5_2)

        plt.tight_layout()
        plt.show()

    a = abs(max(rho_h.min(), rho_h.max()))
    c = abs(max(phi_h.min(), phi_h.max()))
    if a == 0:
        a = 0.3
    b = abs(max(dphi_h.min(), dphi_h.max()))
    if b == 0:
        b = 1
    E = [np.linspace(0, a, ncut + 1), np.linspace(0, c, ncut + 1), np.linspace(-b, b, ncut + 1)]
    
    # ---------------------------------------------------------
    # OPTIMIZATION 2: Multithreaded histogram loop 2
    # ---------------------------------------------------------
    H_inhib = np.empty((n_spk, ncut, ncut, ncut))
    H_inhib_ = np.empty((n_spk, ncut, ncut, ncut))
    data_h = np.concatenate([rho_h, phi_h, dphi_h], axis=1)

    def compute_hist_2(i):
        """Function for compute hist 2.

        Args:
            i: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        Y = spk[i].reshape(-1, 1)
        _h_inh, _ = np.histogramdd(data_h, bins=E, density=False, weights=Y[:, 0])
        _h_inh_, _ = np.histogramdd(data_h, bins=E)
        return i, _h_inh, _h_inh_

    with cpu_threadpool_scope(threads), ThreadPoolExecutor(max_workers=threads) as executor:
        for i, _h_inh, _h_inh_ in executor.map(compute_hist_2, range(n_spk)):
            H_inhib[i] = _h_inh
            H_inhib_[i] = _h_inh_

    H = np.mean(H_inhib, axis=0)
    H_ = np.mean(H_inhib_, axis=0)
    
    smoothed_signal = hanningconv3d(H, 3)
    smoothed_counts = hanningconv3d(H_, 3)
    Z = np.divide(
        smoothed_signal,
        smoothed_counts,
        out=np.zeros_like(smoothed_signal),
        where=smoothed_counts > 0,
    )
    xedges = E[0]
    yedges = E[1]
    zedges = E[2]
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    zcenters = (zedges[:-1] + zedges[1:]) / 2

    mask = np.where(~np.isnan(Z))
    interp_h = NearestNDInterpolator(np.transpose((xcenters[mask[0]], ycenters[mask[1]], zcenters[mask[2]])), Z[mask])
    indices = np.indices(Z.shape)
    indices = np.stack([xcenters[indices[0]], ycenters[indices[1]], zcenters[indices[2]]])
    filled_data = interp_h(*indices)
    filled_data, plot_h = approx_Matrix2(filled_data, plotting=plotting)

    d_h = zcenters[np.argmax(plot[2])] 

    pp = plot_h[1]
    rr = plot_h[0]
    HMP_f = xcenters[np.argmin(abs(rr - (np.max(rr)/2)))]
    cv = circular_variance(np.linspace(0, 360, 20), pp.reshape(-1, 1))
    complexity_f = 1 - cv[1][0]


    return interp, interp_h, d_val, d_h, complexity, complexity_f, HMP, HMP_f, [plots, plot_h]


def GetNeuronVisresponse(idx, w_i, w_r, w_i_inhib, w_r_inhib, dphi, dphi_inhib,
                          spks, n_min, double_wavelet_model, dt1=9000,
                          train_idx=[0, 2, 4], test_idx=[1, 3],
                          lastmin=False, func=relu, sigma=7, plotting=False,
                          frames_per_minute=None) :
    """Function for GetNeuronVisresponse.

    Args:
        idx: Input value for this operation.
        w_i: Input value for this operation.
        w_r: Input value for this operation.
        w_i_inhib: Input value for this operation.
        w_r_inhib: Input value for this operation.
        dphi: Input value for this operation.
        dphi_inhib: Input value for this operation.
        spks: Input value for this operation.
        n_min: Input value for this operation.
        double_wavelet_model: Input value for this operation.
        dt1: Input value for this operation.
        train_idx: Input value for this operation.
        test_idx: Input value for this operation.
        lastmin: Input value for this operation.
        func: Input value for this operation.
        sigma: Input value for this operation.
        plotting: Input value for this operation.
        frames_per_minute: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    if frames_per_minute is None:
        raise ValueError(
            "GetNeuronVisresponse requires frames_per_minute from the configured "
            "stimulus frame rate."
        )
    frames_per_minute = int(frames_per_minute)
    train_idx = [int(i) for i in train_idx]
    test_idx = [int(i) for i in test_idx]
    if not train_idx or not test_idx:
        raise ValueError("GetNeuronVisresponse requires at least one train and one test trial.")
    n_trials = spks.shape[0]
    invalid = [i for i in train_idx + test_idx if i < 0 or i >= n_trials]
    if invalid:
        raise ValueError(f"Trial index/indices out of range for {n_trials} trials: {invalid}")
    spk = spks[:, :n_min * frames_per_minute, idx]
    signal_len = min(len(w_i), len(w_r), len(w_i_inhib), len(w_r_inhib), len(dphi), len(dphi_inhib), spks.shape[1])
    if lastmin:
        dt1 = min(n_min * frames_per_minute, signal_len)
    else:
        dt1 = min(int(dt1), signal_len)
    y_train = spks[train_idx, :dt1, idx]
    y_test = spks[test_idx, :dt1, idx]
    rho, phi = getpolar(w_i, w_r)  # [:dt1]
    f, f_h, d, d_h, c, c_h, hmp, hmp_h, plot= getPhiRho(y_train[:, :dt1], w_i[:dt1], w_r[:dt1], dphi[:dt1], w_i_inhib[:dt1], w_r_inhib[:dt1], dphi_inhib[:dt1], plotting=plotting, sigma=sigma)




    rho_h, phi_h = getpolar( w_i_inhib, w_r_inhib)#[:dt1]
    # --- Training Data Preparation ---
    pred = f(rho, phi, dphi)
    pred_h = f_h(rho_h, phi_h, dphi_inhib.reshape(-1, 1))

    # Vectorized tiling saves memory and CPU time compared to list comprehensions
    base_X1 = np.concatenate((pred[:dt1], pred_h[:dt1]), axis=1)
    X1 = np.tile(base_X1, (len(train_idx), 1))
    X1 = np.nan_to_num(X1)

    # Ravel flattens the array using a memory view (zero RAM cost if contiguous)
    y_train = spks[train_idx, :dt1, idx].ravel()
    y_test = spks[test_idx, :dt1, idx].ravel()
    
    fittedParameters, pcov, res = fitnonlin(X1, y_train, func)

    fittedParameters, pcov, res = fitnonlin(X1, y_train, func)

    res1 = res  # + (w_pc*pcs_test[:, 0])
    res2 = np.mean(res1.reshape(len(train_idx), dt1), axis=0)
    ev = explained_variance_score(np.mean(y_train.reshape(len(train_idx), dt1), axis=0), res2, multioutput='uniform_average')
    feve = FEVE(y_train.reshape(len(train_idx), dt1), res2)
    cc_train=np.corrcoef(np.mean(y_train.reshape(len(train_idx), dt1), axis=0), res2)[0][1]

    # --- Test Data Preparation ---
    # Reuse the same tiling logic for the test indices
    X1 = np.tile(base_X1, (len(test_idx), 1))
    X1 = np.nan_to_num(X1)
    
    unrectified = np.mean(X1[:, 0].reshape(len(test_idx), dt1), axis=0)
    unrectified2 = np.mean(X1[:, 1].reshape(len(test_idx), dt1), axis=0)
    res = func(X1, *fittedParameters)

    res1 = res  # + (w_pc*pcs_test[:, 0])
    w = 0
    if double_wavelet_model:
        res2 = np.mean(res1.reshape(len(test_idx), dt1), axis=0)
    else:
        res21=unrectified
        cc1 = np.corrcoef(np.mean(y_test.reshape(len(test_idx), dt1), axis=0), res21)[0,1]
        res22 = unrectified2
        cc2 = np.corrcoef(np.mean(y_test.reshape(len(test_idx), dt1), axis=0), res22)[0,1]
        if cc1>=cc2:
            res2=unrectified
        else:
            w = 1
            res2=unrectified2
    ev = explained_variance_score(np.mean(y_test.reshape(len(test_idx), dt1), axis=0), res2, multioutput='uniform_average')
    feve = FEVE(y_test.reshape(len(test_idx), dt1), res2)
    cc = np.corrcoef(np.mean(y_test.reshape(len(test_idx), dt1), axis=0), res2)

    cclastmin = np.nan
    if lastmin:
        holdout_start = dt1
        holdout_end = min(dt1 + frames_per_minute, signal_len)
        holdout_len = holdout_end - holdout_start
        if holdout_len <= 1:
            print("Skipping last-minute holdout: not enough frames remain after the training window.")
        else:
            base_holdout = np.concatenate(
                (pred[holdout_start:holdout_end], pred_h[holdout_start:holdout_end]),
                axis=1,
            )
            X1 = np.tile(base_holdout, (len(test_idx), 1))
            X1 = np.nan_to_num(X1)
            y_holdout = spks[test_idx, holdout_start:holdout_end, idx].ravel()
            res = func(X1, *fittedParameters)
            reslastmin = np.mean(res.reshape(len(test_idx), holdout_len), axis=0)
            y_holdout_mean = np.mean(y_holdout.reshape(len(test_idx), holdout_len), axis=0)
            cclastmin = np.corrcoef(y_holdout_mean, reslastmin)[0, 1]
    return res2, [feve, ev, cc, cc_train, cclastmin], fittedParameters, [d, c, hmp, d_h, c_h, hmp_h], plot, unrectified, w, f


def PredictNeuronsTest(wt_test, spks, idx, ncut, dt1=9000, func=relu):
    """Function for PredictNeuronsTest.

    Args:
        wt_test: Input value for this operation.
        spks: Input value for this operation.
        idx: Input value for this operation.
        ncut: Input value for this operation.
        dt1: Input value for this operation.
        func: Input value for this operation.
    """
    test_idx = [0, 2]
    train_idx=[1, 3]
    spk = spks[:, :, idx]
    H = []
    H_ = []
    for i in [0, 2]:
        Y = spk[i].reshape(-1, 1)
        a = abs(max(wt_test.min(), wt_test.max()))
        histo, edges = np.histogram(wt_test, bins=np.linspace(wt_test.min(), wt_test.max(), ncut + 1), density=False, weights=Y[:, 0])
        histo_, edges_ = np.histogram(wt_test, bins=np.linspace(wt_test.min(), wt_test.max(), ncut + 1))
        H.append(histo)
        H_.append(histo_)

    H = np.mean(np.array(H), axis=0)
    H_ = np.mean(np.array(H_), axis=0)
    Z = H / H_

    xedges = np.linspace(wt_test.min(), wt_test.max(), ncut + 1)
    xcenters = (xedges[:-1] + xedges[1:]) / 2

    plt.figure()
    plt.plot(xcenters,Z)
    plt.plot(xcenters,H)
    plt.plot(xcenters,H_)
    from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator, interp1d
    mask = np.where(~np.isnan(Z))


    interp = interp1d(np.transpose(xcenters[mask[0]]), Z[mask], kind='nearest', fill_value=0, bounds_error=False)

    pred = interp(wt_test[:dt1])
    pred_h = np.zeros(pred.shape).reshape(-1, 1)
    pred = pred.reshape(-1, 1)
    X1 = np.concatenate([(np.concatenate(
        (pred, pred_h), axis=1))
        for rep in train_idx])

    X1 = np.nan_to_num(X1)

    y_train = np.concatenate([spks[r, :dt1, idx] for r in train_idx])
    y_test = np.concatenate([spks[r, :dt1, idx] for r in test_idx])
    fittedParameters, pcov, res = fitnonlin(X1, y_train, func)

    print('prediction on training set : ')
    res1 = res  # + (w_pc*pcs_test[:, 0])
    res2 = np.mean(res1.reshape(len(train_idx), dt1), axis=0)
    ev = explained_variance_score(np.mean(y_train.reshape(len(train_idx), dt1), axis=0), res2,
                                  multioutput='uniform_average')
    feve = FEVE(y_train.reshape(len(train_idx), dt1), res2)
    print(feve)
    print(ev)
    print(np.corrcoef(np.mean(y_train.reshape(len(train_idx), dt1), axis=0), res2))

    plt.figure()
    plt.plot(res)
    plt.plot(np.mean(spk, axis=0))


def PlotTuningCurve(rfs, idx, visual_coverage, sigmas, screen_ratio, frequencies, show=True):
    """Function for PlotTuningCurve.

    Args:
        rfs: Input value for this operation.
        idx: Input value for this operation.
        visual_coverage: Input value for this operation.
        sigmas: Input value for this operation.
        screen_ratio: Input value for this operation.
        frequencies: Input value for this operation.
        show: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    xM, xm, yM, ym = visual_coverage
    
    # 1. Extract all 5 max indices clearly (to make the code easier to read)
    maxes = np.array(rfs[1])
    x = maxes[0, idx]
    y = maxes[1, idx]
    o = maxes[2, idx]
    s = maxes[3, idx]
    f = maxes[4, idx]  # NEW: The frequency index

    # 2. Slice the 6D array! We freeze the 6th dimension at 'f' for the spatial/ori plots
    cc_f_1_xy = rfs[0][idx, :, :, o, s, f]
    cc_f_1_o = rfs[0][idx, x, y, :, :, f]
    
    # 3. NEW: Extract the 1D frequency tuning curve (varying the 6th dimension)
    f_tuning = rfs[0][idx, x, y, o, s, :]

    # --- Everything below here is your original SVD logic ---
    u, s__, v = svds(cc_f_1_xy, 2)
    ori_tun = np.append(cc_f_1_o[:, s], cc_f_1_o[0, s])
    i = 1
    if v[1][np.argmax(abs(v[1]))] < 0:
        i = -1
        
    if show:
        # Changed 1, 5 to 1, 6 and made the figure slightly wider
        fig, ax = plt.subplots(1, 6, figsize=(18, 1.5))
        m = ax[0].imshow(cc_f_1_xy.T, cmap='coolwarm', aspect='equal')
        fig.colorbar(m)
        ax[0].set_xticks(
            np.linspace(0, cc_f_1_xy.shape[0] - 1, 3),
            np.round(np.linspace(xM, xm, 3), 2),
        )
        ax[0].set_yticks(
            np.linspace(0, cc_f_1_xy.shape[1] - 1, 3),
            np.round(np.linspace(yM, ym, 3), 2),
        )
        ax[0].set_title('2D correlation')
        
        ax[1].plot(i * v[1][::-1], c='k')
        ax[1].set_xticks([0, cc_f_1_xy.shape[1]], [ym, yM])
        ax[1].set_title('Elevation (deg)')
        
        ax[2].plot(i * u[:, 1], c='k')
        ax[2].set_xticks([0, cc_f_1_xy.shape[0]], [xM, xm])
        ax[2].set_title('Azimuth (deg)')
        
        ax[3].plot(ori_tun, 'o-', c='k')
        n_orientations = rfs[0].shape[3]
        ax[3].set_xticks(
            [0, max(1, n_orientations // 2), max(2, n_orientations - 1)],
            [0, 90, 180],
        )
        ax[3].set_title('Orientation (deg)')
        
        ax[4].plot(cc_f_1_o[o, :], 'o-', c='k')
        ax[4].set_xticks(np.arange(len(sigmas)), sigmas)
        ax[4].set_title('Size (deg)')

        # NEW: Plot the frequency tuning curve
        ax[5].plot(f_tuning, 'o-', c='k')
        ax[5].set_xticks(np.arange(len(frequencies)), [round(freq, 2) for freq in frequencies])
        ax[5].set_title('Spatial Freq')

    # Return your original list, but append f_tuning at the end
    return [cc_f_1_xy.T, i * v[1][::-1], i * u[:, 1], ori_tun, cc_f_1_o[o, :], f_tuning]



from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels



