# Sequential word recognition using Simulation-Based Inference

The model implemented here performs word recognition from an acoustic signal using a syllable module powered by Simulation-Based Inference and a recursive Bayesian update scheme for word-level processing. The model works on syllabified sentences from the TIMIT speech corpus. 

This repository contains utiliy files and a notebook demonstrating how we can use this model, evaluate its performance and compare it to a reduced model. 

- the `DR8` folder contains the syllabified speech corpus (TIMIT).
- the `data_prep.py` file contains functions for extracting and formatting lexical and acoustic data from the speech corpus, and for computing syllable embeddings.
- the `inference_utils.py` file contains functions for building the generative model, training the inference network, running the model, and comparing its performance to the reduced model.
- the `demo.ipynb` notebook goes step-by-step through all the process, from data extraction to model evaluation. 
