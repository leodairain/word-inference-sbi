import numpy as np
import pandas as pd



import torch 
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

import librosa
from scipy.signal import spectrogram
from scipy.interpolate import interp1d


from librosa.sequence import dtw


import warnings
import os

warnings.filterwarnings('ignore')


def distance_matrix(emb: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared Euclidean distances between embeddings.

    Parameters
    ----------
    emb : torch.Tensor
        Tensor of shape (N, D) containing N embeddings of dimension D.

    Returns
    -------
    torch.Tensor
        Distance matrix of shape (N, N) with squared distances.
    """
    # Compute all pairwise differences between embeddings
    diff = emb.unsqueeze(0) - emb.unsqueeze(1)  # (N, N, D)
    # Square and sum across dimensions to get squared distances
    return (diff ** 2).sum(dim=2)  # (N, N)


def add_noise_with_snr(clean_signal: np.ndarray, noise_signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add noise to a clean signal at a specified signal-to-noise ratio (SNR).

    Parameters
    ----------
    clean_signal : np.ndarray
        Input clean signal (1D array).
    noise_signal : np.ndarray
        Noise signal (1D array).
    snr_db : float
        Desired SNR in decibels.

    Returns
    -------
    np.ndarray
        Noisy signal.
    """
    # Extend noise to at least match clean signal length
    if len(noise_signal) < len(clean_signal):
        repeats = int(np.ceil(len(clean_signal) / len(noise_signal)))
        noise_signal = np.tile(noise_signal, repeats)
    noise_signal = noise_signal[:len(clean_signal)]

    # Compute power of clean and noise
    clean_power = np.mean(clean_signal ** 2)
    noise_power = np.mean(noise_signal ** 2)

    # Target noise power for desired SNR
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    # Scale noise accordingly
    scale = np.sqrt(target_noise_power / noise_power)
    scaled_noise = noise_signal * scale
    # Add scaled noise to clean signal
    noisy_signal = clean_signal + scaled_noise

    return noisy_signal


def periodic_interrupt(signal: np.ndarray, fs: int, f: float, duty_cycle: float = 0.5) -> np.ndarray:
    """
    Insert periodic silences into an audio signal.

    Parameters
    ----------
    signal : np.ndarray
        Input audio signal (1D array).
    fs : int
        Sampling frequency in Hz.
    f : float
        Interruption frequency in Hz.
    duty_cycle : float, optional
        Fraction of each period containing signal (0 < duty_cycle <= 1).

    Returns
    -------
    np.ndarray
        Signal with periodic interruptions.
    """
    if not (0 < duty_cycle <= 1):
        raise ValueError("duty_cycle must be between 0 and 1.")

    # Number of samples in a full interruption cycle
    period_samples = int(fs / f)
    sound_samples = int(period_samples * duty_cycle)
    silence_samples = period_samples - sound_samples
    
    # Initialize output with zeros (silence)
    output = np.zeros_like(signal)
    n = len(signal)
    
    # Apply periodic copying of signal
    idx = 0
    while idx < n:
        end_sound = min(idx + sound_samples, n)
        output[idx:end_sound] = signal[idx:end_sound]
        idx += period_samples
    
    return output


def reduced_interpolated_spectrogram(signal: np.ndarray, fs: int, bands: int = 6, target_time_steps: int = 8) -> np.ndarray:
    """
    Compute a reduced spectrogram with frequency band averaging and temporal interpolation.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D or 2D).
    fs : int
        Sampling frequency.
    bands : int, optional
        Number of frequency bands.
    target_time_steps : int, optional
        Number of temporal interpolation steps.

    Returns
    -------
    np.ndarray
        Spectrogram of shape (bands, target_time_steps).
    """
    # Use first channel if stereo
    mono_signal = signal[:,0] if len(signal.shape)==2 else signal
    freqs, times, Sxx = spectrogram(mono_signal, fs, nfft=2**15)

    # Select frequency range up to 3000 Hz
    min_freq = freqs.min()
    max_freq = 3000
    band_edges = np.linspace(min_freq, max_freq, bands + 1)

    # Initialize reduced spectrogram
    S_reduced = np.zeros((bands, Sxx.shape[1]))
        
    # Average power over frequency bands
    for i in range(bands):
        f_start, f_end = band_edges[i], band_edges[i + 1]
        band_mask = (freqs >= f_start) & (freqs < f_end)
        if not np.any(band_mask):
            continue
        S_reduced[i, :] = np.mean(Sxx[band_mask, :], axis=0)

    # Interpolate across time to target_time_steps
    new_times = np.linspace(times[0], times[-1], target_time_steps)
    S_reduced_interp = np.zeros((bands, target_time_steps))
    for i in range(bands):
        f = interp1d(times, S_reduced[i, :], kind='linear', fill_value='extrapolate')
        S_reduced_interp[i, :] = f(new_times)

    return S_reduced_interp


def unflatten_spectrogram(S: np.ndarray, bands: int = 6, times: int = 8) -> np.ndarray:
    """
    Reshape a flattened spectrogram into 2D form.

    Parameters
    ----------
    S : np.ndarray
        Flattened spectrogram of length bands*times.
    bands : int, optional
        Number of frequency bands.
    times : int, optional
        Number of time steps.

    Returns
    -------
    np.ndarray
        Spectrogram of shape (bands, times).
    """
    # Initialize 2D spectrogram
    S_u = np.zeros((bands, times))
    # Fill each band row with the corresponding slice
    for i in range(bands):
        S_u[i,:] = S[times*i:times*(i+1)]
    return S_u


def align_syll_words(sylls: pd.DataFrame, words: pd.DataFrame) -> tuple[list[str], list[list[str]]]:
    """
    Align syllables with words based on time offsets.

    Parameters
    ----------
    sylls : pd.DataFrame
        DataFrame with columns ['onset', 'offset', 'label'] for syllables.
    words : pd.DataFrame
        DataFrame with columns ['onset', 'offset', 'label'] for words.

    Returns
    -------
    tuple
        (list of words, list of aligned syllables).
    """
    word_list = []
    syll_list = []
    iword = 0
    isyll = 0
    current_word = []
    current_sylls = []
    
    # Traverse both syllable and word lists in parallel
    while (iword<len(words)) and (isyll<len(sylls)):
        if int(words['offset'][iword])==int(sylls['offset'][isyll]):
            # Match found: align word and syllable
            current_word.append(words['label'][iword])
            current_sylls.append(sylls['label'][isyll][1:])
            word_list.append(current_word)
            syll_list.append(current_sylls)
            iword+=1
            isyll+=1
            current_word=[]
            current_sylls=[]
        elif int(words['offset'][iword])<int(sylls['offset'][isyll]):
            # Word ends earlier: move to next word
            current_word.append(words['label'][iword])
            iword+=1
        elif int(words['offset'][iword])>int(sylls['offset'][isyll]):
            # Syllable ends earlier: move to next syllable
            current_sylls.append(sylls['label'][isyll][1:])
            isyll+=1
        
    # Join word tokens into complete words
    for i in range(len(word_list)):
        word_list[i] = ' '.join(word_list[i])
        
    return word_list, syll_list


def get_data(main_dir: str, speakers: list[int], n_sentence: int, bands: int, target_time_steps: int) -> tuple:
    """
    Load audio, syllables, and words, compute spectrograms, and align syllables with words.

    Parameters
    ----------
    main_dir : str
        Root directory of data.
    speakers : list[int]
        List of speaker indices.
    n_sentence : int
        Number of sentences per speaker.
    bands : int
        Number of spectrogram frequency bands.
    target_time_steps : int
        Number of spectrogram time steps.

    Returns
    -------
    tuple
        (syllable_labels, spectrograms, aligned_syllables, aligned_words, gaps, nans).
    """
    dir_list = os.listdir(main_dir)[1:-1]
    all_syllable_labels = []
    all_spectrograms = []
    all_aligned_words, all_aligned_sylls = [], []
    sentences_syll = []
    sentences_words = []
    gaps = []
    nans = []

    # Iterate through selected speakers and sentences
    for dir in [dir_list[speaker] for speaker in speakers]:
        for subdir in os.listdir(main_dir + dir)[:n_sentence]:
            if subdir[0]=='S':
                # Load audio
                signal, fs =  librosa.load(main_dir + dir + '/' + subdir + '/' + subdir + '.wav', sr=None, mono=True)
                signal = signal.astype(np.float32)
                # Load syllable timings
                sylls = pd.read_csv(main_dir + dir + '/' + subdir + '/' + subdir +'.sylbtime',sep='\t', header=None, names=['onset', 'offset', 'label'])
                
                # Process each syllable
                for i in range(len(sylls)):
                    if sylls['label'][i][1:]=='#': 
                        gaps.append(main_dir + dir + '/' + subdir + '/' + subdir)
                        print("Gap found in sentence, skipping") 
                        continue
                    onset, offset = int(sylls['onset'][i]), int(sylls['offset'][i])
                    syll_signal = signal[onset:offset]
                    # Compute reduced spectrogram
                    S = reduced_interpolated_spectrogram(syll_signal, fs, bands=bands, target_time_steps=target_time_steps)
                    if np.isnan(S).any():
                        print(f"NaN found in spectrogram for {dir+'/'+subdir}, skipping")
                        nans.append(main_dir + dir + '/' + subdir + '/' + subdir)
                        continue
                    # Save label and normalized spectrogram
                    all_syllable_labels.append(sylls['label'][i][1:])
                    all_spectrograms.append(S / np.max(S.flatten()))
                sentences_syll.append(list(sylls['label']))
                # Load words and align with syllables
                words = pd.read_csv(main_dir + dir + '/' + subdir + '/' + subdir +'.WRD',sep=' ', header=None, names=['onset', 'offset', 'label'])
                word_list, syll_list = align_syll_words(sylls, words)
                all_aligned_words.extend(word_list)
                all_aligned_sylls.extend(syll_list)
                sentences_words.append(word_list)

    return all_syllable_labels, all_spectrograms, all_aligned_sylls, all_aligned_words, gaps, nans


def compute_dtw(all_spectrograms: list[np.ndarray]) -> torch.Tensor:
    """
    Compute pairwise DTW distances between spectrograms.

    Parameters
    ----------
    all_spectrograms : list of np.ndarray
        List of spectrograms.

    Returns
    -------
    torch.Tensor
        Normalized DTW distance matrix.
    """
    n = len(all_spectrograms)
    dtw_distances = np.zeros((n, n))
    # Compute DTW between each pair of spectrograms
    for i in tqdm(range(n)):
        si = all_spectrograms[i]
        for j in range(n):
            sj = all_spectrograms[j]
            d = dtw(si, sj)[0]
            dtw_distances[i,j] = d[-1][-1]

    # Normalize distances to [0,1] and convert to tensor
    dtw_tensor = torch.tensor(dtw_distances)
    dtw_tensor = (dtw_tensor - dtw_tensor.min()) / (dtw_tensor.max() - dtw_tensor.min())
    return dtw_tensor


def compute_embeddings(dtw_tensor: torch.Tensor, num_syll_total: int, embedding_dim: int = 8, num_epochs: int = 10000) -> torch.Tensor:
    """
    Learn embeddings by minimizing distance mismatch with DTW distances.

    Parameters
    ----------
    dtw_tensor : torch.Tensor
        Normalized DTW distance matrix.
    num_syll_total : int
        Total number of syllables.
    embedding_dim : int, optional
        Dimensionality of embeddings.
    num_epochs : int, optional
        Number of optimization epochs.

    Returns
    -------
    torch.Tensor
        Learned embeddings of shape (num_syll_total, embedding_dim).
    """
    # Initialize embeddings randomly
    init = torch.randn(num_syll_total, embedding_dim)*3.0
    embedding = nn.Parameter(init, requires_grad=True)
    optimizer = optim.Adam([embedding], lr=0.1)

    # Gradient descent optimization
    for _ in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        pairwise_dists = distance_matrix(embedding)
        loss = ((pairwise_dists - dtw_tensor) ** 2).mean()
        loss.backward()
        optimizer.step()

    return embedding


def get_path_list(main_dir: str, speakers: list[int], n_sentence: int, gaps: list[str] = [], nans: list[str] = []) -> list[str]:
    """
    Generate list of valid sentence paths, excluding those with gaps or NaNs.

    Parameters
    ----------
    main_dir : str
        Root directory of data.
    speakers : list[int]
        List of speaker indices.
    n_sentence : int
        Number of sentences per speaker.
    gaps : list of str, optional
        Paths with detected gaps.
    nans : list of str, optional
        Paths with detected NaNs.

    Returns
    -------
    list of str
        Valid sentence paths.
    """
    dir_list = os.listdir(main_dir)[1:-1]
    path_list = []
    # Iterate through selected speakers and sentences
    for dir in tqdm([dir_list[speaker] for speaker in speakers]):
        for subdir in tqdm(os.listdir(main_dir + dir)[:n_sentence]):
            if subdir[0]=='S':
                path = main_dir + dir + '/' + subdir + '/' + subdir
                # Exclude invalid paths
                if (path in gaps) or (path in nans):
                    continue
                path_list.append(path)
    return path_list



