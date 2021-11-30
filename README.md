# ITU-ML5G-PS-007-Feedback-Magnitude-Pruning
**The A(MC) Team:** [Jakob Krzyston](https://github.com/JakobKrzyston/), Rajib Bhattacharjea, and Andrew Stark

## About
This repo is for the **ITU-ML5G-PS-007: Lightning-Fast Modulation Classification with Hardware-Efficient Neural Networks** Challenge.
The goal of the competition was to created the fastest* and most efficient* deep learning architecture that could achieve a minimum of 56% classificaiton accuracy on the  DeepSig RadioML 2018 dataset.

In this competition, we highlighted pruning methods for network compression. We use Iterative Magnitude Pruning (IMP) with an accuracy threshold and develop a variant called Feedback Magnitude Pruning (FMP). FMP, analgous to a decaying learning rate scheduler, reduces the pruning parameter when the network is unable to achieve a specified criterion (i.e. accuracy threshold). We demonstrate iterative pruning methods are very effective for network compression, and how adding feedback to IMP enables much greater compression and an improved normalized inference. Our final compression ratios and normalized inference costs are in the table below.

|| Baseline | Iterative Magnitude Pruning  | Feedback Magnitude Pruning |
|-|-------------| ------------- | ------------- | 
|Compression Ratio| 1  | 9.3  | 813  | 
|Inference Cost| 1  | 0.0424  | 0.0419 |


*Speed and efficiency metrics are defined at: https://challenge.aiforgood.itu.int/match/matchitem/34

## Report & Slides ##
Please refer to our report (Feedback_Magnitude_Pruning_11_29_21.pdf) as well as our slides (FMP_slides.pdf) in the repo, for more details!

## Feedback Magnitude Pruning (FMP) ##
In this work we propose an enhanced version of Iterative Magnitude Pruning (IMP) that leverages feedback to adjust the pruning parameters. The algorithm for IMP with an accuracy criterion is as follows:

![](https://github.com/ITU-AI-ML-in-5G-Challenge/ITU-ML5G-PS-007-The-AMC-Team/blob/main/IMP_algorithm.JPG)

Feedback Magnitude Pruning requires the specificaiton of parameters:
- *p_methods*: List of pruning methods to be aplied serially 
- *p*: Initial pruning parameter
- *n*: Determines the rate of decay for *p*

The algorithm for FMP with an accuracy criterion is as follows:

![](https://github.com/ITU-AI-ML-in-5G-Challenge/ITU-ML5G-PS-007-The-AMC-Team/blob/main/FMP_algorithm.JPG)

In this work we used:
- *p_methods*: Unstructured pruning 
- *p*: 0.20 (20% per iteration)
- *n*: 2

**Please note there is a threshold to *p* in order to limit runtime**

## Run the code ##
To run FMP, with our best (to date) results, please run the following (**be sure to input the path to where the data is stored**):
```
python3 feedback_magnitude_pruning.py
```
The parameters (bits, conv filters, dense nodes, sparsity and sparsity decay parameters, are in the top of the .py file.


### Notes
- The code used in this repo stems from the provided repo: https://github.com/Xilinx/brevitas-radioml-challenge-21
