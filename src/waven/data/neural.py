"""Neural recording loaders and stimulus-alignment helpers.

Functions here read suite2p fluorescence/spike outputs, parse timeline sync
signals, align neural activity to stimulus trials, and transform ROI positions.
Stimulus wavelet arrays are handled by :mod:`waven.stimulus`.
"""
import os

import matplotlib

if os.environ.get("waven_NO_PLOTS") == "1":
    matplotlib.use("Agg", force=True)
else:
    matplotlib.use("TkAgg", force=True)

import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np

from ..Analysis_Utils import *
from ..suite2p.utils import cortex_lab_utils as clu
from ..suite2p.utils import timelinepy as tlu
from ..suite2p.utils import utils

def loadExperiment(dirs, exp_info, pathdir, block_end, n_planes=1, n_repeat=6, n_frames=18000):
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


# Updated defaults: Nb_plane=1, Nb_frames=18000
def _base_load_mesoscope(data_type, exp_info, dirs, path, block_end, Nb_plane=1, Nb_frames=18000, first=False, last=True, threshold=1.25, plane=-1, method='frame2ttl', exptype='zebra', w=0.0, plotting=False):
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


# Updated defaults: Nb_plane=1, Nb_frames=18000
def loadFluoMesoscope(exp_info, dirs, path, block_end, Nb_plane=1, Nb_frames=18000, first=False, last=True,
                      threshold=1.25, plane=-1, method='frame2ttl', exptype='zebra'):
    return _base_load_mesoscope('fluo', exp_info, dirs, path, block_end, Nb_plane, Nb_frames, 
                                first, last, threshold, plane, method, exptype, w=0.0, plotting=False)


# Updated defaults: Nb_plane=1, Nb_frames=18000
def loadSPKMesoscope(exp_info, dirs, path, block_end, Nb_plane=1, Nb_frames=18000, first=False, last=True, 
                     threshold=1.25, plane=-1, method='frame2ttl', exptype='zebra', w=0, plotting=False):
    return _base_load_mesoscope('spk', exp_info, dirs, path, block_end, Nb_plane, Nb_frames, 
                                first, last, threshold, plane, method, exptype, w=w, plotting=plotting)


# Added n_planes parameter to bypass the 3-plane math if you only have 1 plane
def correctNeuronPos(neuron_pos, resolution, n_planes):
    neuron_pos = np.asarray(neuron_pos, dtype=np.float64).copy()
    
    # If single plane layout, bypass the legacy unwrapping
    if n_planes == 1:
        return resolution * neuron_pos

    # Calculate dimensions of a single plane
    ly = np.ceil(np.max(neuron_pos[:, 0]) / n_planes)
    lx = np.ceil(np.max(neuron_pos[:, 1]))

    # Generalize the unwrapping for any number of planes
    for p in range(1, n_planes):
        if p == n_planes - 1:
            # The last plane catches everything to the bottom edge
            in_plane = neuron_pos[:, 0] > p * ly
        else:
            in_plane = np.logical_and(neuron_pos[:, 0] > p * ly, neuron_pos[:, 0] <= (p + 1) * ly)
        
        # Shift Y up by p * ly, Shift X right by p * lx
        neuron_pos[in_plane] = neuron_pos[in_plane] + np.array([-p * ly, p * lx])

    return resolution * neuron_pos
