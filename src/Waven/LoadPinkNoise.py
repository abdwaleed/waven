"""Neural data loading, trial alignment, and coarse wavelet preparation.

Loads suite2p mesoscope outputs, aligns spike traces to stimulus frames via
Cortex Lab timeline sync signals, and builds downsampled wavelet tensors used
by receptive-field and nonlinear model stages.
"""
import os
import gc
import matplotlib
import numpy as np
from skimage import transform
import numexpr as ne
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, TensorDataset
import torch

if os.environ.get("waven_NO_PLOTS") == "1":
    matplotlib.use("Agg", force=True)
else:
    matplotlib.use("TkAgg", force=True)

import matplotlib.pyplot as plt
import skimage.transform
from .suite2p.utils import cortex_lab_utils as clu
from .suite2p.utils import timelinepy as tlu
from .suite2p.utils import utils as utils
from .performance import coarse_wavelet_chunk_size, cpu_worker_count
from .Analysis_Utils import *


def load_wavelets(pathdir, nx, ny, wavelets_r, wavelets_i, direction=False):
    """Combine real and imaginary wavelet phases into a normalized magnitude map."""
    w_r = wavelets_r.reshape((-1, ny, nx, 3, 9))
    w_i = wavelets_i.reshape((-1, ny, nx, 3, 9))

    if torch.cuda.is_available():
        try:
            with torch.no_grad():
                w_r_gpu = torch.from_numpy(w_r).cuda()
                w_i_gpu = torch.from_numpy(w_i).cuda()
                pn_wavelets_gpu = torch.abs(w_r_gpu) + torch.abs(w_i_gpu)

                sigma = 1
                pn_wavelets_gpu = pn_wavelets_gpu.reshape((-1, ny, nx, 27))
                sum_wavelets = torch.sum(pn_wavelets_gpu, dim=1, keepdim=True)
                pn_wavelets_gpu = pn_wavelets_gpu / (sigma + sum_wavelets)

                pn_wavelets = pn_wavelets_gpu.cpu().numpy().reshape((-1, ny, nx, 3, 9))

                del w_r_gpu, w_i_gpu, pn_wavelets_gpu, sum_wavelets
                torch.cuda.empty_cache()
        except RuntimeError:
            print("GPU out of memory during load_wavelets; falling back to CPU.")
            torch.cuda.empty_cache()
            pn_wavelets = np.abs(w_r) + np.abs(w_i)
            pn_wavelets = pn_wavelets.reshape((-1, ny, nx, 27))
            sigma = 1
            pn_wavelets = pn_wavelets / (sigma + np.sum(pn_wavelets, axis=1, keepdims=True))
            pn_wavelets = np.reshape(pn_wavelets, (-1, ny, nx, 3, 9))
    else:
        pn_wavelets = np.abs(w_r) + np.abs(w_i)
        pn_wavelets = pn_wavelets.reshape((-1, ny, nx, 27))
        sigma = 1
        pn_wavelets = pn_wavelets / (sigma + np.sum(pn_wavelets, axis=1, keepdims=True))
        pn_wavelets = np.reshape(pn_wavelets, (-1, ny, nx, 3, 9))

    print(pn_wavelets.shape)
    del w_r, w_i
    gc.collect()

    if direction:
        sh = np.array(wavelets_i.shape)
        sh[0] = sh[0] + 1
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = wavelets_i / wavelets_r
            phase = np.insert(
                np.nan_to_num(np.arctan(ratio)),
                [0],
                np.zeros((ny, nx, 3, 9)),
                0,
            ).reshape(sh)
        phase_diff = np.diff(phase, axis=0)
        return pn_wavelets, phase_diff
    return pn_wavelets


def load_stimulus(pathdir, wavelets_r, wavelets_i, nx=161, ny=60):
    wavelets_r = np.load(pathdir + '/cwt_pn_real_1_9000.npy', mmap_mode='c')
    wavelets_i = np.load(pathdir + '/cwt_pn_imag_1_9000.npy', mmap_mode='c')
    
    scale = 2
    direct = True
    if not direct:
        wavelets = load_wavelets(pathdir, nx, ny, direction=direct)
        wavelets = wavelets[:, :, :, scale, :]
        wavelets = transform.resize(wavelets, (wavelets.shape[0], 8, 20, 9))
        wavelets = wavelets.reshape((wavelets.shape[0], wavelets.shape[1] * wavelets.shape[2] * wavelets.shape[3]))
    elif direct:
        w_r = transform.resize(wavelets_r.reshape((-1, ny, nx, 3, 9)), (wavelets_r.shape[0], 8, 20, 3, 9))
        w_i = transform.resize(wavelets_i.reshape((-1, ny, nx, 3, 9)), (wavelets_i.shape[0], 8, 20, 3, 9))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            phase = np.nan_to_num(np.arctan(w_i / w_r))
        phase_diff = np.diff(phase, axis=0)

        # OPTIMIZATION: GPU-accelerated absolute value trick with Fallback
        if torch.cuda.is_available():
            try:
                with torch.no_grad():
                    w_r_gpu = torch.from_numpy(w_r).cuda()
                    w_i_gpu = torch.from_numpy(w_i).cuda()
                    pn_wavelets = (torch.abs(w_r_gpu) + torch.abs(w_i_gpu)).cpu().numpy()
                    del w_r_gpu, w_i_gpu
                    torch.cuda.empty_cache()
            except RuntimeError:
                print("WARNING: GPU Out of Memory. Safely falling back to CPU...")
                torch.cuda.empty_cache()
                pn_wavelets = np.abs(w_r) + np.abs(w_i)
        else:
            pn_wavelets = np.abs(w_r) + np.abs(w_i)
            
        del w_r, w_i
        gc.collect()

        pn_wavelets_fwd = pn_wavelets * np.insert(np.clip(phase_diff, 0, None), [0], np.zeros((8, 20, 3, 9)),
                                                  0).reshape(9000, 8, 20, 3, 9)
        pn_wavelets_bkwd = pn_wavelets * np.insert(np.clip(-phase_diff, 0, None), [0], np.zeros((8, 20, 3, 9)),
                                                   0).reshape(9000, 8, 20, 3, 9)
        wavelets = np.stack([pn_wavelets, pn_wavelets_fwd, pn_wavelets_bkwd], axis=5)
        del pn_wavelets_bkwd, pn_wavelets_fwd, wavelets_r, wavelets_i
        gc.collect()

    wavelets = wavelets[:, :, :, scale, :, :]
    w = wavelets.reshape((9000, 8, 20, 9, 3))
    plt.figure()
    plt.plot(w[:, 5, 14, 4, 0])
    plt.plot(w[:, 5, 14, 4, 1])
    plt.plot(w[:, 5, 14, 4, 2])

    wavelets = wavelets.reshape(
        (wavelets.shape[0], wavelets.shape[1] * wavelets.shape[2] * wavelets.shape[3] * wavelets.shape[4]))
    return wavelets


def load_stimulus_simple_cell(
    path="/media/sophie/Expansion1/UCL/datatest/",
    nx=27,
    ny=11,
    no=8,
    ns=6,
    nf=1,
    downsampling=False,
):
    """Load coarse wavelet phases from ``dwt_videodata_{0,1}.npy`` via memmap."""
    wavelets_r = np.load(os.path.join(path, "dwt_videodata_0.npy"), mmap_mode="r")
    wavelets_i = np.load(os.path.join(path, "dwt_videodata_1.npy"), mmap_mode="r")
    print(wavelets_r.shape)

    if downsampling:
        target_shape = (nx, ny, no, ns, nf)

        def resize_chunk(chunk):
            from skimage import transform

            return transform.resize(
                chunk,
                (chunk.shape[0],) + target_shape,
                anti_aliasing=True,
            )

        safe_jobs = cpu_worker_count(cap=4)
        chunks_r = np.array_split(wavelets_r, safe_jobs, axis=0)
        wavelets_r_resized = Parallel(
            n_jobs=safe_jobs,
            backend="loky",
            timeout=3600,
        )(delayed(resize_chunk)(chunk) for chunk in chunks_r)
        wavelets_r = np.concatenate(wavelets_r_resized, axis=0)
        del wavelets_r_resized
        wavelets_r = np.swapaxes(wavelets_r, 2, 1)
        gc.collect()

        chunks_i = np.array_split(wavelets_i, safe_jobs, axis=0)
        wavelets_i_resized = Parallel(
            n_jobs=safe_jobs,
            backend="loky",
            timeout=3600,
        )(delayed(resize_chunk)(chunk) for chunk in chunks_i)
        wavelets_i = np.concatenate(wavelets_i_resized, axis=0)
        del wavelets_i_resized
        wavelets_i = np.swapaxes(wavelets_i, 2, 1)
        gc.collect()

    return wavelets_r, wavelets_i


def load_stimulus_simple_cell2_i(path='/media/sophie/Expansion1/UCL/datatest/', tt=[0, 9000], downsampling=False):
    wavelets_i = np.load(path + 'dwt_videodata2_i.npy', mmap_mode='r')[tt[0]:tt[1]]
    if downsampling:
        wavelets_i = transform.resize(wavelets_i[:, :, :, :, 2, :], (9000, 27, 11, 8, 4), anti_aliasing=True)
    return wavelets_i


def load_stimulus_simple_cell2_r(path='/media/sophie/Expansion1/UCL/datatest/', tt=[0, 9000], downsampling=False):
    wavelets_r = np.load(path + 'dwt_videodata2_r.npy', mmap_mode='r')[tt[0]:tt[1]]
    if downsampling:
        wavelets_r = transform.resize(wavelets_r[:, :, :, :, 2, :], (tt[1]-tt[0], 27, 11, 8, 4), anti_aliasing=True)
    return wavelets_r


def load_stimulus_simple_cell2(path='/media/sophie/Expansion1/UCL/datatest/', tt=[0, 9000], downsampling=False):
    w_i = load_stimulus_simple_cell2_i(path, tt, downsampling)
    w_r = load_stimulus_simple_cell2_r(path, tt, downsampling)
    return w_r, w_i


def loadExperiment(dirs, exp_info, pathdir, block_end, n_planes, n_repeat=6, n_frames=9000):
    exp_path = exp_info[0] + '/' + exp_info[1]
    tlfile = clu.find_expt_file(exp_info, 'root', dirs=dirs) 
    tlfile = clu.find_expt_file(exp_info, 'timeline', dirs)
    tl = tlu.load_timeline(tlfile)

    input_ind = 'neuralFrames' == tlu.get_input_names(tl)
    tp = tl['rawDAQData'][:, input_ind].flatten()
    ind = np.diff(tp, prepend=tp[0]) > 0
    frame_times = tl['rawDAQTimestamps'][ind]
    frame_times = frame_times[:frame_times.shape[0] - (frame_times.shape[0] % n_planes)]
    frame_times = frame_times.reshape((-1, n_planes))

    neuron_pos = np.concatenate([np.asarray([sta['med'] for sta in np.load(
        dirs[0] + exp_path + '/suite2p/plane%d/stat.npy' % plane,
        allow_pickle=True)[np.load(
        dirs[0] + exp_path + '/suite2p/plane%d/iscell.npy' % plane, mmap_mode='r')[:, 0].astype(bool)]]) for plane in range(n_planes)])

    n_cell = neuron_pos.shape[0]
    Nb_frames = n_frames * n_repeat

    input_ind = 'photoDiode' == tlu.get_input_names(tl)
    syncEcho_thresh = 1.8
    esynv = tl['rawDAQData'][:, input_ind].flatten() > syncEcho_thresh
    syncEcho_flip = np.asarray(np.logical_or(
        np.logical_and(np.logical_not(esynv[:-1]), esynv[1:]),
        np.logical_and(np.logical_not(esynv[1:]), esynv[:-1])
    )).nonzero()[0]
    syncEcho_flip_times = tl['rawDAQTimestamps'][syncEcho_flip]
    print('syncEcho_flip_times: ', syncEcho_flip_times.shape)

    plt.figure()
    input_ind = 'photoDiode' == tlu.get_input_names(tl)
    plt.plot(tl['rawDAQData'][:, input_ind].flatten())
    plt.twiny()
    plt.scatter(frame_times[:, 0], np.ones((frame_times.shape[0])), c='r')

    R = []
    for plane in range(n_planes):
        print('plane :', plane)
        
        is_cell_mask = np.load(dirs[0] + exp_path + '/suite2p/plane%d/iscell.npy' % plane, mmap_mode='r')[:, 0].astype(bool)
        F = np.load(dirs[0] + exp_path + '/suite2p/plane%d/F.npy' % plane, mmap_mode='c')[:, block_end:][is_cell_mask]
        Fneu = np.load(dirs[0] + exp_path + '/suite2p/plane%d/Fneu.npy' % plane, mmap_mode='c')[:, block_end:][is_cell_mask]

        # OPTIMIZATION: NumExpr strictly in-place memory map math without temporary arrays
        spks = ne.evaluate("F - (0.7 * Fneu)")
        
        window = [1.15]
        spks_rt_noz = spks[:, :frame_times.shape[0]]
        spks_rt = utils.zscore(spks_rt_noz, ax=1, epsilon=1e-5)
        
        spks_rt -= np.min(spks_rt, axis=1, keepdims=True)

        try:
            resps_all = utils.interp_event_responses(frame_times[:, plane], spks_rt, events=syncEcho_flip_times,
                                                     window=window, mean_over_window=False, print_interval=None)
        except:
            resps_all = utils.interp_event_responses(frame_times[:spks_rt.shape[1], plane], spks_rt, events=syncEcho_flip_times,
                                                     window=window, mean_over_window=False, print_interval=None)
        print(np.array(resps_all).shape)
        if plane == 0:
            R = np.array(resps_all)
        else:
            R = np.concatenate((R, np.array(resps_all)), axis=1)

    plt.figure()
    plt.scatter(syncEcho_flip_times, np.ones(syncEcho_flip_times.shape[0]))
    plt.scatter(syncEcho_flip_times, resps_all[:, 0], c='g')

    resps_all = np.nan_to_num(R)
    return resps_all, neuron_pos, syncEcho_flip_times


def align_rotary_encoder(exp_info, dirs, spks, Nb_frames, nb_plane=1, plane=-1, w=0.0, threshold=1.25, methods='frame2ttl'):
    tl, frame_times, input_ind, syncEcho_thresh = _extract_timeline_sync(exp_info, dirs, threshold, methods)
    
    rotary_encoder_ind = 'rotary_encoder' == tlu.get_input_names(tl)
    esynv = tl['rawDAQData'][:, input_ind].flatten() > syncEcho_thresh
    syncEcho_flip = np.asarray(np.logical_or(
        np.logical_and(np.logical_not(esynv[:-1]), esynv[1:]),
        np.logical_and(np.logical_not(esynv[1:]), esynv[:-1])
    )).nonzero()[0]
    
    syncEcho_flip_times = tl['rawDAQTimestamps'][syncEcho_flip]
    print('syncEcho_flip_times: ', syncEcho_flip_times.shape)
    rotary_encoder_vals = np.clip(np.diff(tl['rawDAQData'][:, rotary_encoder_ind].flatten()), -10, 10)[syncEcho_flip]
    return rotary_encoder_vals


def align_datas(exp_info, dirs, spks, Nb_frames, nb_plane=1, plane=-1, w=0.0, threshold=1.25, methods='frame2ttl', exptype='zebra', plotting=False):
    """Segment neural activity into stimulus trials and resample onto frame grid.

    Uses photodiode or TTL edges from ``Timeline.mat`` to delimit trials, then
    interpolates z-scored spike traces onto ``Nb_frames`` bins per trial.
    """
    tl, frame_times, input_ind, syncEcho_thresh = _extract_timeline_sync(exp_info, dirs, threshold, methods)
    
    print(methods, syncEcho_thresh)
    esynv = tl['rawDAQData'][:, input_ind].flatten() > syncEcho_thresh

    if exptype != 'zebra':
        print('only up flips are considered')
        syncEcho_flip = np.asarray(
            np.logical_and(np.logical_not(esynv[1:]), esynv[:-1])
        ).nonzero()[0]
    else:
        syncEcho_flip = np.asarray(np.logical_or(
            np.logical_and(np.logical_not(esynv[:-1]), esynv[1:]),
            np.logical_and(np.logical_not(esynv[1:]), esynv[:-1])
        )).nonzero()[0]
        
    syncEcho_flip_times = tl['rawDAQTimestamps'][syncEcho_flip]
    print('syncEcho_flip_times: ', syncEcho_flip_times.shape)

    if plotting:
        plt.figure()
        plt.plot(tl['rawDAQData'][:, input_ind].flatten())
        plt.scatter(syncEcho_flip, np.ones(syncEcho_flip_times.shape[0]), c='k')

    if nb_plane != 1:
        print('multiple planes')
        frame_times = frame_times[(frame_times.shape[0] % nb_plane):].reshape(-1, nb_plane)
        print(frame_times.shape)
        if plane == -1:
            frame_times = np.mean(frame_times, axis=1)
        else:
            frame_times = frame_times[:, plane]
        print(frame_times.shape)

    starttrial = frame_times[frame_times >= syncEcho_flip_times[0]]
    trials = []
    time_trials = []
    tt = True
    t = 1
    while tt:
        try:
            if t == 1:
                print('trial', t)
                trial1 = np.logical_and(frame_times >= syncEcho_flip_times[0], frame_times < syncEcho_flip_times[Nb_frames * t])
                time_trial1 = frame_times[trial1]
                t = t + 1
                print(trial1.shape)
            else:
                print('trial', t)
                trial1 = np.logical_and(frame_times >= syncEcho_flip_times[Nb_frames * (t - 1)],
                                        frame_times < syncEcho_flip_times[Nb_frames * t])
                time_trial1 = frame_times[trial1]
                t = t + 1
            trials.append(trial1)
            time_trials.append(time_trial1)
        except:
            print('incomplete trial')
            tt = False
            trial1 = np.zeros(trials[0].shape)
            temp = np.logical_and(frame_times >= syncEcho_flip_times[Nb_frames * (t - 1)],
                                  frame_times < syncEcho_flip_times[np.minimum(Nb_frames * t, syncEcho_flip_times.shape[0] - 1)])
            trial1[:temp.shape[0]] = temp
            trial1 = trial1.astype(bool)
            time_trial1 = frame_times[trial1]

            if time_trial1.shape != (0,):
                trials.append(trial1)
                time_trials.append(time_trial1)

    window = [w]
    resps_all = []
    resps_all_raw = []
    if plotting:
        plt.figure()
    for i, trial in enumerate(trials):
        print(i, trial.shape, spks.shape, np.max(np.asarray(trial != 0).nonzero()[0]))

        if exptype == 'zebra' or exptype == 'sparse':
            try:
                spks_rt = utils.zscore(spks[:, np.asarray(trial != 0).nonzero()[0]], ax=1, epsilon=1e-5)
                spks_rt -= np.min(spks_rt, axis=1, keepdims=True)
                
                if plotting:
                    plt.plot(spks_rt[200, :])
                print(np.sum(trial), len(time_trials[i]), spks_rt.shape)
                temp = np.zeros((Nb_frames, spks.shape[0], 1))
                print('exptype : ', exptype)
                temp1 = utils.interp_event_responses(time_trials[i], spks_rt,
                                                     events=syncEcho_flip_times[Nb_frames * i:Nb_frames * (i + 1)],
                                                     window=window, mean_over_window=False, print_interval=None)
            except:
                print('warning: spks too short ?')
                print(spks.shape, np.max(np.asarray(trial != 0).nonzero()[0]))
                spks_t = np.zeros((spks.shape[0], 1 + np.max(np.asarray(trial != 0).nonzero()[0])))
                spks_t[:, :spks.shape[1]] = spks
                spks_rt = utils.zscore(spks_t[:, np.asarray(trial != 0).nonzero()[0]], ax=1, epsilon=1e-5)

                spks_rt -= np.min(spks_rt, axis=1, keepdims=True)
                
                if plotting:
                    plt.plot(spks_rt[200, :])
                print(np.sum(trial), len(time_trials[i]), spks_rt.shape)
                temp = np.zeros((Nb_frames, spks.shape[0], 1))
                print('exptype : ', exptype)
                temp1 = utils.interp_event_responses(time_trials[i], spks_rt,
                                                     events=syncEcho_flip_times[Nb_frames * i:Nb_frames * (i + 1)],
                                                     window=window, mean_over_window=False, print_interval=None)
        elif exptype == 'gratings':
            window_ts = np.arange(0, 2, 0.033)
            spks_rt = utils.scale_std(spks[:, np.asarray(trial != 0).nonzero()[0]])

            resps = utils.interp_event_responses(time_trials[i], spks_rt, events=syncEcho_flip_times[Nb_frames * i:Nb_frames * (i + 1)],
                                                 window=window_ts, mean_over_window=False, print_interval=None)
            print(resps.shape)
            temp1 = np.moveaxis(resps, 2, 1).reshape(-1, resps.shape[1], 1)
            temp = np.zeros((int(temp1.shape[0]), spks.shape[0], 1))

        temp[:temp1.shape[0]] = temp1
        resps_all.append([temp])
        resps_all_raw.append(spks_rt)

    return resps_all, resps_all_raw


def _base_load_mesoscope(data_type, exp_info, dirs, path, block_end, Nb_plane=3, Nb_frames=9000, first=False, last=True, threshold=1.25, plane=-1, method='frame2ttl', exptype='zebra', w=0.0, plotting=False):
    """Shared loader for fluorescence and deconvolved spike mesoscope data.

    Reads suite2p plane outputs with memory mapping, extracts ROI positions from
    ``stat.npy``, and delegates trial alignment to :func:`align_datas`.
    """
    
    def load_plane_data(p, start_idx, end_idx):
        mask = np.load(path + '/plane%d/iscell.npy' % p, mmap_mode='r')[:, 0].astype(bool)
        if data_type == 'fluo':
            F = np.load(path + '/plane%d/F.npy' % p, mmap_mode='c')[mask]
            Fneu = np.load(path + '/plane%d/Fneu.npy' % p, mmap_mode='c')[mask]
            return ne.evaluate("F - (0.7 * Fneu)")[:, start_idx:end_idx]
        else: # 'spk'
            return np.load(path + '/plane%d/spks.npy' % p, mmap_mode='c')[mask][:, start_idx:end_idx]

    if first:
        print('first session')
        slice_start, slice_end = None, block_end
    elif last:
        print('last session')
        slice_start, slice_end = block_end, None
    else:
        print('mid')
        slice_start, slice_end = block_end[0], block_end[1]

    if Nb_plane != 1:
        print('multiple planes')
        if plane != -1:
            print('loading planes nb ', plane)
            spks = load_plane_data(plane, slice_start, slice_end)
        else:
            print('loading all planes')
            M = [load_plane_data(p, None, None) for p in range(Nb_plane)]
            min_len = M[-1].shape[1] if M else 0
            spks = np.concatenate([m[:, :min_len] for m in M])[:, slice_start:slice_end]
    else:
        print('single plane')
        spks = np.concatenate([load_plane_data(p, slice_start, slice_end) for p in range(Nb_plane)])

    if Nb_plane != 1:
        if plane != -1:
            if data_type == 'fluo':
                print('loading planes nb ', plane)
            neuron_pos = np.array([(1, plane * 512) + np.asarray([sta['med'] for sta in np.load(
                path + '/plane%d/stat.npy' % plane, allow_pickle=True)[
                np.load(path + '/plane%d/iscell.npy' % plane, mmap_mode='r')[:, 0].astype(bool)]])])[0]
        else:
            if data_type == 'fluo':
                print('loading all planes')
            neuron_pos = np.concatenate([(1, p * 512) + np.asarray([sta['med'] for sta in np.load(
                path + '/plane%d/stat.npy' % p, allow_pickle=True)[
                np.load(path + '/plane%d/iscell.npy' % p, mmap_mode='r')[:, 0].astype(bool)]]) for p in range(1, Nb_plane)])
    else:
        if data_type == 'fluo':
            print('single plane')
        neuron_pos = np.concatenate([(1, p * 512) + np.asarray([sta['med'] for sta in np.load(
            path + '/plane%d/stat.npy' % p, allow_pickle=True)[
            np.load(path + '/plane%d/iscell.npy' % p, mmap_mode='r')[:, 0].astype(bool)]]) for p in range(Nb_plane)])

    print('shape spks : ', spks.shape)
    print('neuron_pos spks : ', neuron_pos.shape)

    resps_all, resps_all2 = align_datas(exp_info, dirs, spks, Nb_frames, nb_plane=Nb_plane, threshold=threshold,
                                        plane=plane, methods=method, exptype=exptype, w=w, plotting=plotting)
    print('data aligned')
    resps_all = np.array(resps_all)
    resps_all = np.nan_to_num(resps_all)
    resps_all = resps_all[:, 0, :, :, 0]
    return resps_all, resps_all2, neuron_pos


def _extract_timeline_sync(exp_info, dirs, threshold, methods):
    """Consolidated logic to parse timeline sync thresholds and inputs."""
    tlfile = clu.find_expt_file(exp_info, 'timeline', dirs)
    tl = tlu.load_timeline(tlfile)

    try:
        input_ind = 'neuralFrames' == tlu.get_input_names(tl)
        tp = tl['rawDAQData'][:, input_ind].flatten()
        ind = np.diff(tp, prepend=tp[0]) > 0
        frame_times = tl['rawDAQTimestamps'][ind]
        input_ind = 'photoDiode' == tlu.get_input_names(tl)
        syncEcho_thresh = 1.5
    except:
        if methods in ['photosensor', 'frame2ttl']:
            syncEcho_thresh = threshold
        else:
            print('unknown timeline variable')

        print(methods, syncEcho_thresh)
        input_ind = 'neural_frames' == tlu.get_input_names(tl)
        tp = tl['rawDAQData'][:, input_ind].flatten()
        ind = np.diff(tp, prepend=tp[0]) > 0
        frame_times = tl['rawDAQTimestamps'][ind]
        input_ind = methods == tlu.get_input_names(tl)

    return tl, frame_times, input_ind, syncEcho_thresh


def loadFluoMesoscope(exp_info, dirs, path, block_end, Nb_plane=3, Nb_frames=9000, first=False, last=True,
                      threshold=1.25, plane=-1, method='frame2ttl', exptype='zebra'):
    return _base_load_mesoscope('fluo', exp_info, dirs, path, block_end, Nb_plane, Nb_frames, 
                                first, last, threshold, plane, method, exptype, w=0.0, plotting=False)


def loadSPKMesoscope(exp_info, dirs, path, block_end, Nb_plane=3, Nb_frames=9000, first=False, last=True, 
                     threshold=1.25, plane=-1, method='frame2ttl', exptype='zebra', w=0, plotting=False):
    """Load deconvolved suite2p spikes and align them to stimulus trials."""
    return _base_load_mesoscope('spk', exp_info, dirs, path, block_end, Nb_plane, Nb_frames, 
                                first, last, threshold, plane, method, exptype, w=w, plotting=plotting)


def correctNeuronPos(neuron_pos, resolution=1.3671):
    """Convert suite2p pixel coordinates to microns on a flattened cortical map.

    Multi-plane mesoscope layouts stack planes vertically in pixel space.  This
    function unwraps that tiling so neighbouring neurons share a continuous 2-D
    coordinate system, then scales by ``resolution`` (µm per pixel).

    Parameters
    ----------
    neuron_pos : ndarray, shape (n_neurons, 2)
        ROI centroids in suite2p coordinates (row, column).
    resolution : float
        Microscope sampling resolution in µm per pixel.

    Returns
    -------
    ndarray
        Positions in microns on the unwrapped map.
    """
    neuron_pos = np.asarray(neuron_pos, dtype=np.float64).copy()
    ly = np.ceil(np.max(neuron_pos[:, 0]) / 3)
    lx = np.ceil(np.max(neuron_pos[:, 1]))

    mid_plane = np.logical_and(neuron_pos[:, 0] > ly, neuron_pos[:, 0] <= 2 * ly)
    neuron_pos[mid_plane] = neuron_pos[mid_plane] + np.array([-ly, lx])

    top_plane = neuron_pos[:, 0] > 2 * ly
    neuron_pos[top_plane] = neuron_pos[top_plane] + np.array([-2 * ly, 2 * lx])

    return resolution * neuron_pos


def coarseWavelet(
    path,
    downsampling,
    nx0=135,
    ny0=54,
    nx=27,
    ny=11,
    no=8,
    ns=6,
    nf=1,
    chunk_size=None,
):
    """Load or build spatially downsampled wavelets for receptive-field analysis.

    Reads ``dwt_downsampled_videodata.npy`` when present; otherwise downsamples
    ``dwt_videodata_{0,1}.npy`` in time chunks and caches the result to disk.

    Returns
    -------
    w_r, w_i, w_c : ndarray
        Real, imaginary, and power (r² + i²) wavelet tensors at coarse resolution.
    """
    if chunk_size is None:
        chunk_size = coarse_wavelet_chunk_size()

    cache_path = os.path.join(path, "dwt_downsampled_videodata.npy")
    print("loading wavelets...")
    if os.path.exists(cache_path):
        print("already downsampled")
        wavelets_downsampled = np.load(cache_path, mmap_mode="r")
        w_r_downsampled = wavelets_downsampled[0]
        w_i_downsampled = wavelets_downsampled[1]
        w_c_downsampled = wavelets_downsampled[2]
        return w_r_downsampled, w_i_downsampled, w_c_downsampled

    print("downsampling")
    wavelets_r, wavelets_i = load_stimulus_simple_cell(
        path, nx, ny, no, ns, nf, downsampling
    )
    n_frames = wavelets_r.shape[0]
    n_chunks = max(1, int(n_frames / chunk_size))

    real_parts = []
    imag_parts = []
    power_parts = []

    for chunk_index in range(n_chunks):
        start = chunk_index * chunk_size
        end = min((chunk_index + 1) * chunk_size, n_frames)
        w_r = wavelets_r[start:end]
        w_i = wavelets_i[start:end]
        wavelets_complex = np.square(w_r) + np.square(w_i)

        target_r = (w_r.shape[0], nx, ny, no, ns, nf)
        w_r_downsampled = skimage.transform.resize(
            w_r.reshape((-1, nx0, ny0, no, ns)),
            target_r,
            anti_aliasing=True,
        )
        w_i_downsampled = skimage.transform.resize(
            w_i.reshape((-1, nx0, ny0, no, ns)),
            target_r,
            anti_aliasing=True,
        )
        w_c_downsampled = skimage.transform.resize(
            wavelets_complex.reshape((-1, nx0, ny0, no, ns, nf)),
            target_r,
            anti_aliasing=True,
        )
        del w_r, w_i, wavelets_complex

        real_parts.append(w_r_downsampled)
        imag_parts.append(w_i_downsampled)
        power_parts.append(w_c_downsampled)
        gc.collect()

    del wavelets_r, wavelets_i
    w_r_downsampled = np.concatenate(real_parts, axis=0)
    w_i_downsampled = np.concatenate(imag_parts, axis=0)
    w_c_downsampled = np.concatenate(power_parts, axis=0)
    del real_parts, imag_parts, power_parts

    np.save(cache_path, [w_r_downsampled, w_i_downsampled, w_c_downsampled])
    return w_r_downsampled, w_i_downsampled, w_c_downsampled
