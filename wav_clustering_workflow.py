import os
import glob
import pandas as pd
from shutil import copy
import pickle

from pyAudioAnalysis import ShortTermFeatures as aF
from pyAudioAnalysis import audioBasicIO as aIO

from scipy.cluster.hierarchy import ward, leaves_list, fcluster
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt


def get_short_term_features(wav_loc, win=0.10, step=0.10):
    fs, s = aIO.read_audio_file(wav_loc)
    s = aIO.stereo_to_mono(s)
    duration = len(s) / float(fs)
    print(f'{wav_loc} duration = {duration} seconds')
    try:
        f, fn = aF.feature_extraction(s, fs, int(fs * win), int(fs * step))
        print(f'{f.shape[1]} frames, {f.shape[0]} features')
        return f, fn
    except ValueError:
        return None, None


def flatten_n_frames(f, n):
    m = f[:, :n]
    return m.flatten('F')


def get_features_frame(wav_locs, first_n_frames, include_parent_dir=False):
    feature_dict = {}
    for w in wav_locs:
        name = os.path.basename(w)
        if include_parent_dir:
            name = os.path.basename(os.path.dirname(w)) + '/' + name
        f, fn = get_short_term_features(w)
        if f is not None:
            feature_dict[name] = flatten_n_frames(f, first_n_frames)
        else:
            print(f'{w} too short, skipping')
    df = pd.DataFrame.from_dict(feature_dict, orient='index').transpose()
    return df


def pickle_object(obj, outdir, out_name):
    with open(os.path.join(outdir, out_name), 'wb') as f:
        pickle.dump(obj, f)


def save_cluster_wavs(parent_dir, samples, outdir):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'original_filenames.txt'), 'w') as f:
        for s in samples:
            f.write(os.path.join(parent_dir, s) + '\n')
    for i, s in enumerate(samples):
        out_name = f"{str(i).zfill(5)}_{s.replace('/', '_')}"
        copy(os.path.join(parent_dir, s), os.path.join(outdir, out_name))


def cluster_and_save_k(globbed_wav_list, n_frames, parent_dir, outdir, k):
    if len(globbed_wav_list) < k:
        raise ValueError("Not enough samples to form k clusters")

    df = get_features_frame(globbed_wav_list, n_frames, include_parent_dir=False)
    df.fillna(0, inplace=True)
    X = df.T.values

    # hierarchical clustering
    Z = ward(pdist(X))
    # flat cluster assignment
    labels = fcluster(Z, t=k, criterion='maxclust')

    # save linkage and data
    os.makedirs(outdir, exist_ok=True)
    pickle_object(Z, outdir, 'ward_linkage.pkl')
    pickle_object(df, outdir, 'df_matrix.pkl')
    df.to_csv(os.path.join(outdir, 'all_features.csv'), index=False)

    # create k folders and save
    for cluster_id in range(1, k+1):
        cluster_samples = [s for s, lbl in zip(df.columns, labels) if lbl == cluster_id]
        cluster_dir = os.path.join(outdir, f'cluster_{cluster_id:02d}')
        save_cluster_wavs(parent_dir, cluster_samples, cluster_dir)


