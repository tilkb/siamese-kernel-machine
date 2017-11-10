# siamese-kernel-machine
This is the official PyTorch implementation of the `Make SVM Great Again with Siamese kernel for few-shot learning` [paper](https://openreview.net/pdf?id=B1EVwkqTW).
This implementation contains only Omniglot experiment.

## Install Prerequisites
    pip install -r requirements.txt

## Run
 
    python train.py

Parameters:
* --batch-size <batch size=256> 
* --test-batch-size <test batch size=256> 
* --epochs <epoch number=200> 
* --C <C=0.2> 
* --test-number <number of test=10>
* --seed <random seed=42>

## Keras version of the experiment
Keras 1.2.2 required, this is the original implementation of the paper.

    python train-keras.py
