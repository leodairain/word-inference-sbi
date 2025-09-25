import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.cm import get_cmap

import gc

import data_prep

import torch 


from tqdm import tqdm

import librosa

from scipy.stats import entropy

import sbi
from sbi import analysis as analysis
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils.user_input_checks import (

    process_prior,
    process_simulator,
)

from statsmodels.stats.contingency_tables import mcnemar

import warnings


warnings.filterwarnings('ignore')

cmap = get_cmap("viridis")


def softmax_temperature(logits, temperature=1.0):
    """
    Compute softmax distribution with temperature scaling.
    
    Args:
        logits (array-like): Input values.
        temperature (float): Scaling factor. >1 flattens, <1 sharpens.
    
    Returns:
        np.ndarray: Probability distribution after softmax.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    logits = np.asarray(logits)
    scaled_logits = logits / temperature

    # Numerical stability trick: subtract maximum
    if scaled_logits.ndim == 1:
        exps = np.exp(scaled_logits - np.max(scaled_logits))
        return exps / np.sum(exps)
    else:  # Handle 2D input
        exps = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)



def build_simulator(embedding, all_spectrograms, sigma=0.05):
    """
    Build a simulator function that generates spectrogram-like features 
    given input parameters.
    
    Args:
        embedding (torch.Tensor): Embedding space for comparison.
        all_spectrograms (list): List of spectrogram arrays.
        sigma (float): Noise scaling factor.
    
    Returns:
        function: Simulator function mapping theta -> noisy spectrograms.
    """
    def simulator(theta):
        bands, target_time_steps = all_spectrograms[0].shape
        batch_size = theta.shape[0]
        x = torch.zeros((batch_size, bands * target_time_steps))
        for batch in range(batch_size):
            # Find nearest embedding
            dists = torch.norm(embedding.detach() - theta[batch], dim=1).detach()
            i = int(torch.argmin(dists))            
            s = torch.tensor(all_spectrograms[i].flatten(), dtype=torch.float32)
            # Add Gaussian noise
            x[batch] = s + torch.randn_like(s) * sigma * torch.mean(s)

        return x

    return simulator



def compute_density_estimator(simulator, embedding, num_simulations=10000, estimator='mdn'):
    """
    Train a density estimator using SBI with given simulator.
    
    Args:
        simulator (function): Simulator mapping theta -> x.
        embedding (torch.Tensor): Embedding data for bounds.
        num_simulations (int): Number of simulations to run.
        estimator (str): Estimator type (e.g., 'mdn').
    
    Returns:
        tuple: (inference object, trained density estimator)
    """
    lo = torch.tensor(np.min(embedding.detach().numpy(), axis=0)) * 1.1
    hi = torch.tensor(np.max(embedding.detach().numpy(), axis=0)) * 1.1

    prior = sbi.utils.BoxUniform(low=lo, high=hi)

    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    simulator = process_simulator(simulator, prior, prior_returns_numpy)

    # Inference engine
    inference = NPE(prior=prior, density_estimator=estimator)

    # Run simulations
    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=num_simulations)
    inference = inference.append_simulations(theta.clone(), x.clone())

    # Train estimator
    torch.autograd.set_detect_anomaly(True)  
    density_estimator = inference.train(max_num_epochs=150)

    return inference, density_estimator



def compute_prob_words(prob_words, p_follow, discrete_posterior, T2, alpha=.5):
    """
    Update word probabilities using posterior syllable probabilities.
    
    Args:
        prob_words (np.ndarray): Current word probabilities.
        p_follow (np.ndarray): Transition probabilities.
        discrete_posterior (np.ndarray): Posterior over syllables.
        T2 (float): Softmax temperature for normalization.
        alpha (float): Weighting factor between prior and posterior.
    
    Returns:
        np.ndarray: Updated word probabilities.
    """
    p_follow_with_posterior = np.dot(p_follow, discrete_posterior)
    p_follow_with_posterior = np.log(softmax_temperature(p_follow_with_posterior, 0.01))

    prob_words = (1 - alpha) * np.log(prob_words) + alpha * p_follow_with_posterior
    prob_words += np.max(abs(prob_words))  # shift for stability
    prob_words = softmax_temperature(prob_words, T2)

    return prob_words



def extract_from_path(path, bands, target_time_steps, **kwargs):
    """
    Extract syllables and spectrograms from audio and annotation files.
    
    Args:
        path (str): File path prefix (expects .wav, .sylbtime, .WRD).
        bands (int): Number of frequency bands for spectrogram.
        target_time_steps (int): Temporal resolution of spectrogram.
        kwargs: Optional noise parameters.
    
    Returns:
        tuple: (aligned syllables, list of spectrogram tensors)
    """
    signal, fs = librosa.load(path + '.wav', sr=None, mono=True)
    signal = signal.astype(np.float32)
    
    sylls = pd.read_csv(path + '.sylbtime', sep='\t', header=None, names=['onset', 'offset', 'label'])
    words = pd.read_csv(path + '.WRD', sep=' ', header=None, names=['onset', 'offset', 'label'])

    _, aligned_sylls = data_prep.align_syll_words(sylls, words)

    spectrograms = []

    if 'noise_params' in kwargs:
        q, snr = kwargs['noise_params'] 
        noise = np.random.normal(0, 1, signal.shape)
        noisy_signal = data_prep.add_noise_with_snr(signal, noise, snr)
        p = (np.random.rand(len(sylls)) < q)

        for i in range(len(sylls)):
            onset, offset = int(sylls['onset'][i]), int(sylls['offset'][i])
            syll_signal = noisy_signal[onset:offset] if p[i] else signal[onset:offset]       
            S = data_prep.reduced_interpolated_spectrogram(syll_signal, fs, bands=bands, target_time_steps=target_time_steps)
            spectrograms.append(torch.tensor(S.flatten() / np.max(S.flatten()), dtype=torch.float32))

    else:
        for i in range(len(sylls)):
            onset, offset = int(sylls['onset'][i]), int(sylls['offset'][i])
            syll_signal = signal[onset:offset]   
            S = data_prep.reduced_interpolated_spectrogram(syll_signal, fs, bands=bands, target_time_steps=target_time_steps)
            spectrograms.append(torch.tensor(S.flatten() / np.max(S.flatten()), dtype=torch.float32))

    return aligned_sylls, spectrograms



def sequential_inference(aligned_sylls, spectrograms,
                         embedding, posterior, 
                         unique_words_sequences, unique_syllable_labels,
                         classes, count_classes,
                         T1, T2, beta):
    """
    Perform sequential inference over a sentence.
    
    Args:
        aligned_sylls (list): Aligned syllables per word sequence.
        spectrograms (list): Spectrograms for each syllable.
        embedding (torch.Tensor): Embedding representation.
        posterior (object): Trained posterior density estimator.
        unique_words_sequences (list): List of possible word sequences.
        unique_syllable_labels (list): Unique syllable labels.
        classes (list): Syllable-to-class mapping.
        count_classes (int): Total number of syllable classes.
        T1 (float): Temperature for syllable posterior.
        T2 (float): Temperature for word posterior.
        beta (float): Weighting for posterior-prior interpolation.
    
    Returns:
        tuple: (inferred word sequence, inferred syllables)
    """
    num_syll = len(unique_syllable_labels)
    num_syll_tot = len(classes)
    num_words = len(unique_words_sequences)    

    words, sylls = [], []
    i_sentence = 0

    for w, seq in enumerate(aligned_sylls):
        prob_words = np.ones(num_words) / num_words
        prior_syll_classes = np.ones(num_syll_tot) / num_syll_tot

        for i, syll in enumerate(seq):
            # (a) Inference step
            xi = spectrograms[i_sentence]
            if beta > 0.25:
                eta = 1 if i == 0 else ((len(seq) - i) * (beta + 0.25) + i * (beta - 0.25)) / len(seq)
            else:
                eta = 1

            discrete_posterior_all = softmax_temperature(
                eta * posterior.log_prob(embedding, xi).detach() + (1 - eta) * np.log(prior_syll_classes), T1
            )
            
            discrete_posterior = np.array(
                [np.sum(discrete_posterior_all[np.where(np.array(classes) == i)[0]]) for i in range(num_syll)]
            )

            sylls.append(unique_syllable_labels[np.argmax(discrete_posterior)])
            entropy_posterior = entropy(discrete_posterior) / np.log(num_syll)

            # (b) Word probability update
            numerator_follow = np.array(
                [[1 if (len(word_seq) > i and word_seq[i] == s) else 0.1
                  for s in unique_syllable_labels]
                 for word_seq in unique_words_sequences]
            ) 
            p_follow = numerator_follow / np.sum(numerator_follow + 1e-10, axis=-1, keepdims=True) 
            weight_past_estimates = 1 / (1 + 0.5 * entropy_posterior)

            prob_words = compute_prob_words(prob_words, p_follow, discrete_posterior, T2, weight_past_estimates)
            
            # (c) Next prior computation
            next_syll = np.sum(
                [(prob_words[l]) * np.array(
                    [1 if (len(unique_words_sequences[l]) > i+1 and unique_words_sequences[l][i+1] == s) else 0.05
                     for s in unique_syllable_labels])
                 for l in range(num_words)],
                axis=0
            )            
            prior_syll_classes = np.array([next_syll[c]/count_classes[c] for c in classes])
            
            i_sentence += 1
            gc.collect()

        words.append(unique_words_sequences[np.argmax(prob_words)])

    return words, sylls


def mcnemar_test(model_A, model_B, exact=True):
    """
    Run McNemar's test on two paired models.
    
    Args:
        model_A (iterable): Predictions from model A.
        model_B (iterable): Predictions from model B.
        exact (bool): Whether to run exact test.
    
    Returns:
        object: McNemar test result.
    """
    table = np.zeros((2, 2), dtype=int)
    for a, b in zip(model_A, model_B):
        table[a, b] += 1
    return mcnemar(table, exact=exact)



def compute_stats(no_prior, with_prior):
    """
    Compute accuracy means, errors, and significance between conditions.
    
    Args:
        no_prior (list): Performance without prior.
        with_prior (list): Performance with prior.
    
    Returns:
        tuple: (means, errors, p-value)
    """
    df = pd.DataFrame(np.array([no_prior, with_prior]).T, columns=["No Prior", "Word Prior"])
    df = df.map(int)
    pvalue = mcnemar_test(df["No Prior"], df["Word Prior"]).pvalue.round(4)
    means = df.mean()
    errors = df.sem()  
    return means, errors, pvalue


def symbol_pvalue(pvalue):
    """
    Map p-value to significance symbol.
    
    Args:
        pvalue (float): Test p-value.
    
    Returns:
        str: Significance symbol.
    """
    if pvalue > .05:
        return 'ns'
    elif pvalue > 0.01:
        return '*' 
    elif pvalue > 0.001:
        return '**'
    else:
        return '***'
    


def compare_performance(perf_words, perf_words_reduced, perf_sylls, perf_sylls_reduced):
    """
    Compare word and syllable performance with and without prior.
    
    Args:
        perf_words (list): Word performance with prior.
        perf_words_reduced (list): Word performance without prior.
        perf_sylls (list): Syllable performance with prior.
        perf_sylls_reduced (list): Syllable performance without prior.
    
    Returns:
        None: Displays bar plot with error bars and significance markers.
    """
    means_w, errors_w, pvalue_w = compute_stats(perf_words_reduced, perf_words)
    means_s, errors_s, pvalue_s = compute_stats(perf_sylls_reduced, perf_sylls)

    colors = {"w":cmap(0.25), "s":cmap(0.75)}
    real_metrics = {"w":"Words","s":"Syllables"}
    hatches = {'//': "BU", '': "BU + TD"}  

    metric_handles = {}
    hatch_handles = {hatch : Patch(facecolor="white",
                                   edgecolor="black",
                                   hatch=hatch,
                                   label=hatches[hatch]) for hatch in hatches}

    width = 2
    y, h, col = 1.05, 0.05, 'k'
    a = np.array([-width, width])/2

    plt.figure(figsize=(5,5))

    # Words
    bar_w = plt.bar(-1*width*1.5+a, means_w, yerr=errors_w, capsize=5,
                    color=colors["w"], hatch=list(hatches.keys()),
                    width=width, edgecolor="black")
    plt.bar_label(bar_w, fmt='{:.2f}', padding=7.5, fontsize=12)
    metric_handles[real_metrics["w"]] = bar_w[1]
    x1, x2 = -1*width*1.5 + a
    symbol = symbol_pvalue(pvalue_w)
    plt.plot([x1+0.05, x1+0.05, x2-0.05, x2-0.05],
             [y+0.1, y+h+0.1, y+h+0.1, y+0.1], lw=1.5, c=col)
    plt.text((x1+x2)*.5, y+h+0.1, symbol, ha='center', va='bottom', color=col)

    # Syllables
    bar_s = plt.bar(1*width*1.5+a, means_s, yerr=errors_s, capsize=5,
                    color=colors["s"], hatch=list(hatches.keys()),
                    width=width, edgecolor="black")
    plt.bar_label(bar_s, fmt='{:.2f}', padding=7.5, fontsize=12)
    metric_handles[real_metrics["s"]] = bar_s[1]
    x1, x2 = 1*width*1.5 + a
    symbol = symbol_pvalue(pvalue_s)
    plt.plot([x1+0.05, x1+0.05, x2-0.05, x2-0.05],
             [y+0.1, y+h+0.1, y+h+0.1, y+0.1], lw=1.5, c=col)
    plt.text((x1+x2)*.5, y+h+0.1, symbol, ha='center', va='bottom', color=col)

    # Legends
    metric_legend = plt.figlegend(metric_handles.values(), real_metrics.values(),
                                  loc=8, bbox_to_anchor=(0.53, 0.08),
                                  ncol=len(metric_handles), frameon=True, fontsize=12)
    plt.gca().add_artist(metric_legend)

    plt.figlegend(hatch_handles.values(),
                  [hatches[h] for h in hatch_handles.keys()],
                  loc=8, bbox_to_anchor=(0.53, 0.0),
                  ncol=len(hatch_handles), frameon=True, fontsize=12)

    plt.ylim(0, 1.3)
    plt.xticks([])
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Model peformance on raw sentences", fontsize=15)
    plt.tight_layout(rect=[0,0.15,1,1])
    plt.show()
