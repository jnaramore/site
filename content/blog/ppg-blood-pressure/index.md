---
title: PPG to Blood Pressure
summary: PPG signals are collected from optical sensors on smart watches and pulse oximeters to measure heart rate and blood oxygen levels. New research has indicated the shape of the PPG waveform could contain information on blood pressure, which could allow for "cuff-less" blood pressure measurements. This project lays out the pulse wave analysis method, that is how to process raw PPG signals into blood pressure values using machine learning.
date: 2025-10-19

# Featured image
# Place an image named `featured.jpg/png` in this page's folder and customize its options here.
image:
  caption: 'Example PPG with derivatives and fiducial points'

authors:
  - admin

tags:
  - Biosignals
  - PPG
  - Machine Learning
---

{{< toc mobile_only=true is_open=true >}}

## Intro

PPG signals are collected from optical sensors on smart watches and pulse oximeters to measure heart rate and blood oxygen levels. New research has indicated the shape of the PPG waveform contains information on blood pressure, which could allow for "cuff-less" blood pressure measurements. This project lays out the pulse wave analysis method, that is how to process raw PPG signals into blood pressure values using machine learning.

Code for this analysis can be found in the PPG_to_BP github [repository](https://github.com/jnaramore/PPG_to_BP).

## Methods

The goal here is to reproduce similar results to the paper by [Gonzalez et al. (2023)](https://pubmed.ncbi.nlm.nih.gov/36944668/), which proposes a baseline for estimating blood pressure from PPGs using a variety of ML algorithms. There are many ways to approach this, but for now, I will extract features from the PPG waveforms using manually derived "biomarkers", or features extracted from the signal and its 1st, 2nd and 3rd derivatives. Then those features will be used as predictors in a regression-type machine learning model, with systolic or diastolic blood pressure as the response. Another approach that will be explored in a follow-up post is using a convolutional neural network to extract features, or an embeddeding representing the pulse shapes.

The signal processing pipeline includes 

1. Apply a Chebyshev bandpass filter to remove noise, calculate and smooth the derivatives
2. Segment pulses using a beat detection algorithm
3. Find "fiducial" points in the signal and its derivatives
4. Process the fiducial points into "biomarkers"
5. Fit a machine learning algorithm using the biomarkers as predictors, and the systolic/diastolic blood pressure values as the response

For steps 1, 3 and 4, the [pyPPG library](https://pyppg.readthedocs.io/en/latest/) developed by Peter Charlton is used as a starter. pyPPG was originally built to analyze longer signal segments and take advantage of correlation between sequential pulses for things like beat detection and signal quality index, but it does not work well with shorter segments like those in the PPGBP dataset described below. An assumption and goal of the analysis presented here is that a *single* pulse waveform can be analyzed to predict blood pressure. A custom library of functions was built to interface with pyPPG to work with short segments, and can be found as `PPG_to_BP_lib.py` in the github [repository](https://physionet.org/content/pcst/1.0.0/). Also, the QPPG beat detection algorithm published as Matlab code in the [PhysioNet Cardiovascular Signal Toolbox](https://physionet.org/content/pcst/1.0.0/) is modified for shorter pulses in this custom library.

A total of 15 fiducial points are found on the PPG and its derivatives, which are used to calculate 91 biomarkers. Figure 1 illustrates fiducial points in a smoothed PPG signal, and some example biomarker calculations. For example, systolic peak time is the time difference between the systolic peak (sp) and pulse onset.

{{< figure src="Biomarkers_pyppg.png" caption="**Figure 1.** Example fiducial points and biomarker calculations in the PPG pulse (source: Goda et al. [2023](https://iopscience.iop.org/article/10.1088/1361-6579/ad33a2))." >}}

At this point a variety of machine learning algorithms can be trained using the 91 biomarkers as predictors, and blood pressure as the response. As far as more traditional ML algorithms, I've found nonlinear algorithms like RandomForest, XGBoost, and Multi-layer perceptron to perform best and give similar results, so the results of XGBoost will be presented in this post for simplicity.

## Dataset

High-quality PPG signals were collected from 219 healthy and hypertensive patients in a 15-minute-long standard protocol using a specific fingertip PPG sensor (SEP9AF-2). In the standard protocol, the subjects rested for 10 minutes. Then blood pressure measurements were performed with a cuff, and 3x 2.1-second PPG segments were
recorded. Each subject has a single blood pressure measurement with 2.1 x 3 = 6.3 seconds of PPG signals (source: [Liang et al 2018](https://www.nature.com/articles/sdata201820)).

## Results

Cross-validation was used to measure performance with and without data leakage, that is allowing pulses from the same subject or segment to be in both the training and test sets. In this case, "leakage" is not referring to the same exact data being included in the train and test sets, but rather nearby pulses in the same segment of the same subject. Data leakage allows an ML algorithm to overfit to the nearby pulse shapes, which would cause prediction error when generalizing the algorithm to new patients. As Gonzalez (2023) and others have pointed out, data leakage is commonly overlooked in BP, and may amount to misleading results. Using `sklearn`, ensuring cross-validation by group is as simple as using `GroupKFold` rather than `KFold` when creating folds. The number of folds is was set to the number of subjects, making this leave-one-subject-out cross-validation. This is appropriate to estimate the performance of the algorithm for a new subject. Paramameter tuning was done using sklearn's RandomizedSearchCV for a few of XGBoost's parameters including n_estimators, max_depth, learning_rate, and min_child_weight. 

Figure 2 shows the XGBoost model prediction results. Both CV methods result in better performance than the mean SBP MAE of the dataset (16.81 mmHg), with the leak-CV method performing markedly better. These results are also very close to those presented in the benchmark paper by Gonzalez.

{{< figure src="XGBoost_SBP_Predictions_Leak_vs_NoLeak.png" caption="**Figure 2.** Systolic blood pressure prediction with leak and no-leak leave-one-subject-out cross-validation schemes." >}}

A more qualitive thing we can observe in the no-leak-CV result is that the algorithm performs poorly at predicting low and high blood pressures, it is essentially predicting most of them within a normal range at about 110 to 140 mmHg. Even though the algorithm is an improvement over the mean SBP, it provides very little or even misleading information for the cases that matter, i.e. patients with abnormally high blood pressure. The PPGBP dataset is high quality in that it comes from a controlled experiment, but the number of points at extreme high and low blood pressures is low.

The only current product I am aware of that uses PPG to estimate blood pressure is [hilo](https://hilo.com/), which has only recently received market clearance, and requires monthly calibration. The pulse wave analysis presented within this post may work better when calibrated to each person or site, because of the unnaccountable differences between individuals with only PPG measurements. Or, a device that simulataneously measures ECG or PPG at a second site may be accurate using the pulse transit time (PTT) technique.

Although the prediction results aren't great, they indicate there could be some association between the PPG shape and blood pressure.  Similar results were found with other datasets like [SENSORS](https://www.mdpi.com/1424-8220/21/6/2167). Future work might include a weighting or sampling of the more extreme blood pressure values.


## References

González, Sergio, Wan-Ting Hsieh, and Trista Pei-Chun Chen. "A benchmark for machine-learning based non-invasive blood pressure estimation using photoplethysmogram." Scientific Data 10, no. 1 (2023): 149.

Goda, Márton Áron, Peter H. Charlton and Joachim A. Behar. “pyPPG: a Python toolbox for comprehensive photoplethysmography signal analysis.” Physiological Measurement 45 (2023)

Vest, Adriana Nicholson, Giulia Da Poian, Qiao Li, Chengyu Liu, Shamim Nemati, Amit J. Shah and Gari D. Clifford. “An open source benchmarked toolbox for cardiovascular waveform and interval analysis.” Physiological Measurement 39 (2018)

Liang, Yongbo, Zhencheng Chen, Guiyong Liu and Mohamed Elgendi. “A new, short-recorded photoplethysmogram dataset for blood pressure monitoring in China.” Scientific Data 5 (2018)

Aguirre, Nicolas, Edith Grall-Maës, Leandro J. Cymberknop, and Ricardo L. Armentano. "Blood pressure morphology assessment from photoplethysmogram and demographic information using deep learning with attention mechanism." Sensors 21, no. 6 (2021): 2167.
