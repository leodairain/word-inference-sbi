import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import gc


import torch 

import librosa
from scipy.signal import spectrogram
from scipy.interpolate import interp1d
from scipy.stats import entropy

from librosa.sequence import dtw

import sbi
from sbi import analysis as analysis
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)


import ipywidgets as widgets
from IPython.display import clear_output, display
import warnings
import os

warnings.filterwarnings('ignore')



def reduced_interpolated_spectrogram(signal, fs, bands=6, target_time_steps=8):
    """
    Returned a spectrogram with the specified number of frequency bands, with the specified number of interpolation points.
    """
    mono_signal = signal[:,0] if len(signal.shape)==2 else signal
    freqs, times, Sxx = spectrogram(mono_signal, fs, nfft=2**15)

    # Sxx = np.log(Sxx + 1e-3)
    F, T = Sxx.shape
    min_freq = freqs.min()
    max_freq = 3000
        
    # Define band edges (uniform in linear Hz scale)
    band_edges = np.linspace(min_freq, max_freq, bands + 1)
        
    S_reduced = np.zeros((bands, T))
        
    for i in range(bands):
        # Find indices of frequencies in current band
        f_start, f_end = band_edges[i], band_edges[i + 1]
        band_mask = (freqs >= f_start) & (freqs < f_end)
            
        if not np.any(band_mask):
            continue  # avoid divide-by-zero if no freqs fall in this band
            
        # Average over frequency bins within the band
        S_reduced[i, :] = np.mean(Sxx[band_mask, :], axis=0)

    # Temporal interpolation
    new_times = np.linspace(times[0], times[-1], target_time_steps)
    S_reduced_interp = np.zeros((bands, target_time_steps))

    for i in range(bands):
        f = interp1d(times, S_reduced[i, :], kind='linear', fill_value='extrapolate')
        S_reduced_interp[i, :] = f(new_times)

    return S_reduced_interp


def unflatten_spectrogram(S, bands=6, times=8):
    """Transforms a 1D spectrogram of length (n*m) into a 2D spectrogram of size (n, m)"""
    S_u = np.zeros((bands, times))
    for i in range(bands):
        S_u[i,:] = S[times*i:times*(i+1)]

    return S_u


def softmax_temperature(logits, temperature=1.0):
    """Softmax function with a temperature to control the sharpness of the output distribution
    temperature > 1 : flatter
    temperature = 1 : basic softmax
    temperature < 1 : sharper
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    logits = np.asarray(logits)
    scaled_logits = logits / temperature

    # Subtract max for numerical stability
    if scaled_logits.ndim == 1:
        exps = np.exp(scaled_logits - np.max(scaled_logits))
        return exps / np.sum(exps)
    else:  # assume 2D input
        exps = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)


def build_prior(v,embedding, sigma=0.05):
    
    # Wrap in mixture
    mix = torch.distributions.MixtureSameFamily(
        mixture_distribution=torch.distributions.Categorical(probs=v),
        component_distribution=torch.distributions.Independent(torch.distributions.Normal(embedding, sigma), 1)  # if using Diagonal Gaussians
    )

    return mix


def sequential_inference_3d(path, embedding, posterior, prior, transition_matrix, syll2spectrogram, syll2idx, sigma=0.05, show=False):
    num_syll = transition_matrix.shape[0]
    unique_syllable_labels = list(syll2idx.keys())
    signal, fs =  librosa.load(path + '.wav', sr=None, mono=True)
    signal = signal.astype(np.float32)

    sylls = pd.read_csv(path + '.sylbtime',
                        sep='\t', header=None, names=['onset', 'offset', 'label'])

    syllables = ['#']
    spectrograms = []
    bands, target_time_steps = syll2spectrogram[unique_syllable_labels[0]].shape

    for i in range(len(sylls)):
        onset, offset = int(sylls['onset'][i]), int(sylls['offset'][i])
        syll_signal = signal[onset:offset]

        syll_label = sylls['label'][i][1:]

        syllables.append(syll_label)
                    
        S = reduced_interpolated_spectrogram(syll_signal, fs, bands=bands, target_time_steps=target_time_steps)
        spectrograms.append(torch.tensor(S.flatten() / np.max(S.flatten()), dtype=torch.float32))

    res_with = np.zeros(len(syllables)-1)
    res_without = np.zeros(len(syllables)-1)

    syll2idx_all = syll2idx
    syll2idx_all['#'] = 0
    for i in range(len(syllables)-1):
        syll_gt = syllables[i+1]
        syll_i = syll2idx[syll_gt]-1
        xi = torch.tensor(syll2spectrogram[syll_gt].flatten(), dtype=torch.float32)
        xi = spectrograms[i]


        samples = posterior.sample((1000,), x=xi) 

        theta_0 = embedding[syll_i].detach()[None,:]     
        

        

        dists = [torch.norm(embedding.detach() - samples[j], dim=1).detach() for j in range(samples.shape[0])]
        predictions = np.argmin(dists, axis=1)
        print(predictions.shape)
        u, c = np.unique(predictions, return_counts=True)

        res_without[i] = int(u[np.argmax(c)] == syll_i)

        v = torch.tensor(transition_matrix[syll2idx[syllables[i]]][1:], dtype=torch.float32)
        new_prior = build_prior(v, num_syll, embedding, sigma=0.05)
        new_prior_samples = new_prior.sample((1000,))


        log_weights = new_prior.log_prob(samples)  - prior.log_prob(samples)

        # For numerical stability, subtract the max log weight before exponentiating
        log_weights -= torch.max(log_weights)
        weights = torch.exp(log_weights)

        normalized_weights = weights / torch.sum(weights)

        resampled_indices = torch.multinomial(normalized_weights, num_samples=1000, replacement=True)
        resampled_theta = samples[resampled_indices]

        dists_wp = [torch.norm(embedding.detach() - resampled_theta[j], dim=1).detach() for j in range(resampled_theta.shape[0])]
        predictions_wp = np.argmin(dists_wp, axis=1)
        u_wp, c_wp = np.unique(predictions_wp, return_counts=True)

        res_with[i] = int(u_wp[np.argmax(c_wp)] == syll_i)

        if show:
            fig = interactive_scatter_3d(samples, name="Posterior without prior")  

            fig.add_trace(go.Scatter3d(
                x=theta_0[:,0].numpy(), 
                y=theta_0[:,1].numpy(), 
                z=theta_0[:,2].numpy(),
                mode='markers+text',
                marker=dict(
                    size=5,
                    color='red',  
                    opacity=1.0
                ),
                name='Ground Truth',
                text="Ground Truth"
            ))

            

            new_prior_samples = new_prior.sample((1000,))

            fig.add_trace(go.Scatter3d(
                x=new_prior_samples[:,0].numpy(), 
                y=new_prior_samples[:,1].numpy(), 
                z=new_prior_samples[:,2].numpy(),
                mode='markers',
                marker=dict(
                    size=3,
                    color='green',  
                    opacity=0.1
                ),
                name="New Prior"
                
            ))

            fig.add_trace(go.Scatter3d(
                x=resampled_theta[:,0].numpy(), 
                y=resampled_theta[:,1].numpy(), 
                z=resampled_theta[:,2].numpy(),
                mode='markers',
                marker=dict(
                    size=3,
                    color='orange',  
                    opacity=0.1
                ),
                name="Posterior with prior"
                
            ))



            fig.update_layout(
                scene=dict(
                    xaxis=dict(range=[lo[0].item(), hi[0].item()]),
                    yaxis=dict(range=[lo[1].item(), hi[1].item()]),
                    zaxis=dict(range=[lo[2].item(), hi[2].item()])
                )
            )

            fig.show()

    return res_without, res_with

def align_syll_words(sylls:pd.DataFrame, words:pd.DataFrame):
    word_list = []
    syll_list = []
    iword = 0
    isyll = 0
    current_word = []
    current_sylls = []
    
    while (iword<len(words)) and (isyll<len(sylls)):
        
        if int(words['offset'][iword])==int(sylls['offset'][isyll]):

            current_word.append(words['label'][iword])
            current_sylls.append(sylls['label'][isyll][1:])
            word_list.append(current_word)
            syll_list.append(current_sylls)
            iword+=1
            isyll+=1
            current_word=[]
            current_sylls=[]
        elif int(words['offset'][iword])<int(sylls['offset'][isyll]):
            current_word.append(words['label'][iword])
            iword+=1
        elif int(words['offset'][iword])>int(sylls['offset'][isyll]):
            current_sylls.append(sylls['label'][isyll][1:])
            isyll+=1
        
    for i in range(len(word_list)):
        word_list[i] = ' '.join(word_list[i])
        

        
    return word_list, syll_list


def compute_discrete_posterior(samples, embedding, old_prior, new_prior, T1, beta=0.5):
    log_weights = beta*new_prior.log_prob(samples)  - (1-beta)*old_prior.log_prob(samples)
            
    log_weights -= torch.max(log_weights)
    weights = torch.exp(log_weights)

    normalized_weights = weights / torch.sum(weights)


    resampled_indices = torch.multinomial(normalized_weights, num_samples=1000, replacement=True)
    resampled_theta = samples[resampled_indices]
    

    dists = [torch.norm(embedding.detach() - resampled_theta[j], dim=1).detach() for j in range(resampled_theta.shape[0])]
    predictions = np.argmin(dists, axis=1)
    u, c = np.unique(predictions, return_counts=True)

    discrete_posterior = (np.array([c[int(np.where(u==k)[0])] if k in u else 1 for k in range(embedding.shape[0])]))
    discrete_posterior = softmax_temperature(discrete_posterior, T1)

    return discrete_posterior

def compute_prob_words(prob_words, p_follow, discrete_posterior, T2, alpha=.5):
    p_follow_with_posterior = np.dot(p_follow, discrete_posterior)
    p_follow_with_posterior = np.log(softmax_temperature(p_follow_with_posterior, 1))

    prob_words = (1-alpha)*np.log(prob_words / np.sum(prob_words)) + (alpha)*p_follow_with_posterior
    prob_words += (np.max(abs(prob_words)))
    prob_words = softmax_temperature(prob_words, T2)

    return prob_words

def KL_divergence_discrete(p, q):
    return np.sum(
        [p[i] * (np.log10(p[i]) - np.log10(q[i])) for i in range(len(p))]
    )
    

def sequential_inference_3d_words_sequences(path, 
                                  embedding, 
                                  posterior, 
                                  unique_words_sequences, all_aligned_sylls, 
                                  syll2spectrogram, syll2idx, count_syll,
                                  sigma, 
                                  T1, T2, T3, 
                                  alpha, beta):
    
   
    syllable_labels = list(syll2idx.keys())
    num_syll = len(syllable_labels)

    num_words = len(unique_words_sequences)


    signal, fs =  librosa.load(path + '.wav', sr=None, mono=True)
    signal = signal.astype(np.float32)

    sylls = pd.read_csv(path + '.sylbtime',
                        sep='\t', header=None, names=['onset', 'offset', 'label'])

    words = pd.read_csv(path + '.WRD',
                        sep=' ', header=None, names=['onset', 'offset', 'label'])



    aligned_words, aligned_sylls = align_syll_words(sylls, words)


    spectrograms = []
    bands, target_time_steps = syll2spectrogram[syllable_labels[0]].shape
    for i in range(len(sylls)):
        onset, offset = int(sylls['onset'][i]), int(sylls['offset'][i])
        syll_signal = signal[onset:offset]

                    
        S = reduced_interpolated_spectrogram(syll_signal, fs, bands=bands, target_time_steps=target_time_steps)
        spectrograms.append(torch.tensor(S.flatten() / np.max(S.flatten()), dtype=torch.float32))

    
    res = np.zeros(len(aligned_words))
    res_wp_syll = np.zeros(len(aligned_words))
    res_wp_words = np.zeros(len(aligned_words))

    resyll = []
    resyll_by_count = {i+1:[] for i in range(count_syll[max(count_syll, key=lambda x : count_syll[x])])}

    i_sentence = 0
    for w, seq in enumerate(aligned_sylls):
        if w==0:
            prob_words = np.ones(num_words) / num_words
            prob_words_wp_syll = np.ones(num_words) / num_words
            prob_words_wp_words = np.ones(num_words) / num_words

            prior_syll_v = torch.tensor(
                [j==syll2idx[seq[0]] for j in range(num_syll)], 
                dtype=torch.float32)
            # prior_syll = build_prior(prior_syll_v, embedding, sigma)

            prior_words_v = torch.tensor(np.ones(num_syll)/num_syll, dtype=torch.float32)
            # prior_words = build_prior(prior_words_v, embedding, sigma)

            

        for i, syll in enumerate(seq):
            pw_entropy_np = entropy(prob_words) / np.log(num_words)
            pw_entropy_wp_words = entropy(prob_words_wp_words) / np.log(num_words)
            #########################################################################################
            ########### (a) Use the inference network to infer the posterior distribution ###########
            #########################################################################################
            
            xi = spectrograms[i_sentence]
            i_sentence+=1


            

            # Compute Posterior with flat prior
            discrete_posterior = softmax_temperature(
                posterior.log_prob(embedding, xi).detach(), T1
                )
            
            

            entropy_np = entropy(discrete_posterior) / np.log(num_syll)

            
            resyll.append(int(np.argmax(discrete_posterior)==syll2idx[syll]))
            resyll_by_count[count_syll[syll]].append(int(np.argmax(discrete_posterior)==syll2idx[syll]))

            # Compute Posterior with word modulated prior
            # discrete_posterior_wp_syll = softmax_temperature(
            #     (1-beta)*posterior.log_prob(embedding, xi).detach() + beta*prior_syll.log_prob(embedding).detach(), T1
            #     )
            discrete_posterior_wp_syll = prior_syll_v

    

            # discrete_posterior_wp_words = softmax_temperature(
            #     (1-beta)*posterior.log_prob(embedding, xi).detach() + beta*prior_words.log_prob(embedding).detach(), T1
            #     )
            
            discrete_posterior_wp_words = softmax_temperature(
                (1-beta)*posterior.log_prob(embedding, xi).detach() + beta*prior_words_v, T1
                )
            
            
            entropy_wp_words = entropy(discrete_posterior_wp_words) / np.log(num_syll)
            

            # plt.figure()
            # plt.plot(discrete_posterior, label='No Prior')
            # plt.plot(discrete_posterior_wp_words, label='Word Prior')
            # plt.axvline(syll2idx[syll], c='r', linestyle='--')
            # plt.legend()
            # plt.show()

            ##################################################################
            ########### (b) Compute the updated word probabilities ###########
            ##################################################################


            numerator_follow = np.array(
                [
                    [
                        1 if ((len(word_seq)>i) and (word_seq[i]==s)) else 0.05
                        for s in syllable_labels
                    ] 
                    for word_seq in unique_words_sequences
                ]
            ) 

            
            p_follow = numerator_follow / np.sum(numerator_follow + 1e-10, axis=-1, keepdims=True) 
            # p_follow = np.round(p_follow)

            # plt.figure()
            # plt.pcolormesh(p_follow)
            # plt.show()

            # Flat prior
            prob_words = compute_prob_words(prob_words, p_follow, discrete_posterior, T2, 1 - (1 /(1+ alpha * entropy_np)))
            
            # Syllable Prior
            prob_words_wp_syll = compute_prob_words(prob_words_wp_syll, p_follow, discrete_posterior_wp_syll, T2, 0.5)

            # Word Prior
            prob_words_wp_words = compute_prob_words(prob_words_wp_words, p_follow, discrete_posterior_wp_words, T2, (1 /(1+ alpha* entropy_wp_words)))

            # plt.figure()
            # plt.plot(prob_words, label='No Prior')
            # plt.plot(prob_words_wp_syll, label='GT Prior')
            # plt.plot(prob_words_wp_words, label='Word Prior')
            # plt.axvline(x=unique_words_sequences.index(seq), c='r', linestyle='--')
            # plt.title(f"Word {w}/{len(aligned_sylls)}, Syllable {i+1}/{len(seq)}")
            # plt.legend()
            # plt.tight_layout()
            # plt.show()
            

            
            ##########################################################################
            ########### (c) Compute the global syllable transition network ###########
            ##########################################################################
            # global_syllable_network = np.sum(
            #     np.array([prob_words_wp_words[l] * transition_matrices_words[all_aligned_words[l]] for l in range(num_words)]),
            #     axis=0
            #     )
            
                      
            global_syllable_network = np.sum(
                [prob_words_wp_words[l] * np.array(
                
                [
                    [
                        1
                        if ((len(unique_words_sequences[l])>i+1) and (unique_words_sequences[l][i]==s1) and (unique_words_sequences[l][i+1]==s2)) 
                        else 0
                        for s2 in syllable_labels
                    ] 
                    for s1 in syllable_labels
                ])
                for l in range(num_words)
                ],
            axis=0) 

            
            

            
  
            ##################################################
            ########### (d) Compute the next prior ###########
            ##################################################
            
            # Syllable Prior
            
            if i<len(aligned_sylls[w])-1:
                prior_syll_v = torch.tensor(
                    [j==syll2idx[aligned_sylls[w][i+1]] for j in range(num_syll)], 
                    dtype=torch.float32
                )

            elif w<len(aligned_words)-1:
                prior_syll_v = torch.tensor(
                    [j==syll2idx[aligned_sylls[w+1][0]] for j in range(num_syll)], 
                    dtype=torch.float32
                )

            # prior_syll = build_prior(prior_syll_v, embedding, sigma)

            # Word Prior
            # prior_words_v = torch.tensor(
            #     softmax_temperature(discrete_posterior_wp_words @ global_syllable_network.T, T3),
            #     dtype=torch.float32
            #     )
            

            
            prior_words_v = torch.tensor(
                softmax_temperature(np.sum(global_syllable_network, axis=0), T3),
                dtype=torch.float32
                )
            
            
            # print(np.unique(prior_words_v))
            # plt.figure()
            # plt.subplot(2,1,1)
            # plt.pcolormesh(global_syllable_network)
            # plt.subplot(2,1,2)
            # plt.plot(prior_words_v)
            # if len(all_aligned_sylls[w])>i+1:
            #     plt.axvline(syll2idx[all_aligned_sylls[w][i+1]], c='r')
            # plt.show()


            
            # prior_words = build_prior(prior_words_v, embedding, sigma)
            # clear_output(wait=True)

            # plt.figure()
            # plt.subplot(3,1,1)
            # plt.plot(prior_syll_v)
            # plt.title(f"Syllable Prior {w} - {i}")
            # plt.axvline(torch.argmax(prior_syll_v), c='r', linestyle='--')

            # plt.subplot(3,1,2)
            # plt.plot(prior_words_v)
            # plt.title(f"Word Prior {w} - {i}")
            # plt.axvline(torch.argmax(prior_syll_v), c='r', linestyle='--')
            # plt.tight_layout()
            # plt.show()

            # plt.subplot(3,1,3)
            # plt.plot(discrete_posterior_wp_words)
            # plt.title(f"Posterior {w} - {i}")
            # plt.axvline(syll2idx[syll], c='orange', linestyle='--')
            # plt.tight_layout()
            # plt.show()

            gc.collect()



        if np.argmax(prob_words_wp_syll)!=unique_words_sequences.index(seq):
            print(unique_words_sequences[np.argmax(prob_words_wp_syll)], seq)

        res[w] = int(np.argmax(prob_words)==unique_words_sequences.index(seq))
        res_wp_syll[w] = int(np.argmax(prob_words_wp_syll)==unique_words_sequences.index(seq))
        res_wp_words[w] = int(np.argmax(prob_words_wp_words)==unique_words_sequences.index(seq))


        
        prior_words_v = torch.tensor(np.ones(num_syll)/num_syll, dtype=torch.float32)

        # prior_words = build_prior(prior_words_v, embedding, sigma)





    return res, res_wp_syll, res_wp_words, resyll, resyll_by_count


def build_simulator(embedding, syll2spectrogram, sigma=0.05):
        def simulator(theta):
            u = np.unique(list(syll2spectrogram.keys()))

            bands, target_time_steps = syll2spectrogram[u[0]].shape
            batch_size = theta.shape[0]
            x = torch.zeros((batch_size, bands * target_time_steps))
            for batch in range(batch_size):
                dists = torch.norm(embedding.detach() - theta[batch], dim=1).detach()
                i = int(torch.argmin(dists))            
                s = torch.tensor(syll2spectrogram[u[i]].flatten(), dtype=torch.float32)
                x[batch] = s + torch.randn_like(s)*sigma*torch.mean(s)

            return x

        return simulator 


