# **Peak detector for property intruder alerts (TensorFlow)**
## Overview
**Project context** 

  In this project I used real-world sensor data to show the machine learning (ML) concepts I learned in [Google's Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course). The data was originally collected for a room occupancy alarm by biomedical engineering students that are friends of mine. Although they ultimately did not train an ML model, they shared their data with me so that I could finish the project by implementing a logistic regression model with TensorFlow (Keras).

**Data summary**

  The data is a relative humidity (RH) signal recorded in a room as a person entered and left at specific times, simulating an "intrusion." The recording lasted 2 hours and 19 minutes, with an RH measurement taken every 2 seconds. This resulted in 4,169 samples and 12 manually recorded intrusions.

<img src="/Graphs/raw_RH_signal.png">

**Analysis tools**: Jupyter Notebooks, TensorFlow (Keras), NumPy, Pandas, Scikit-learn, Plotly, Matplotlib

## Data Processing
First, the five-point moving average of the signal was calculated and subtracted from the next average. This leveled the baseline, highlighted sudden increases, and enabled real-time analysis.

<img src="/Graphs/delta_RH_moving_avg.png">

Next, "positive" signal windows containing peaks were obtained using the ground-truth intrusion time marks as the central sample.

<img src="/Graphs/12_positive_windows.png">

Then, "negative" windows (without peaks) were obtained, avoiding values from the positive windows and ensuring they were at least 90 samples away from a real peak. These 90 samples (3 min) acted as a detector blocking mechanism to prevent false positives after a detection.

<img src="/Graphs/negative_windows.png">

Several features were then extracted from each of these windows as inputs for the ML model:
- Mean
- Standard deviation
- Peak-to-peak difference
- Argmax (position in the window of the highest signal value)
- First two frequency bins, to detect differences in lower frequencies (since the signal doesn't fluctuate much, meaning that positive windows only have one bump)

<img src="/Graphs/features.png">
<img src="/Graphs/features_corr_mx.png">

Each feature was normalized and grouped together to form an input matrix, with each row containing the six features from each window.

## Model Setup & Validation

The trained model was a logistic regression model that took the six features as inputs and returned a probability that a window was positive.

Since the dataset was very imbalanced (only 12 out of 2,906 windows were positive), it would be best to use all of the positive peaks to train the model. However, before doing so, we must ensure that the model can detect unseen peaks and will not merely memorize the 12 known peaks.

To accomplish this, a Leave-One-Group-Out (LOGO) validation strategy was implemented. First, one peak is randomly selected. Then, a model is trained using the remaining peaks (ensuring that there is no leakage of the selected peak). Finally, the model is evaluated to determine its ability to detect the selected peak.

## Final Model Training & Results

After confirming that the LOGO models could detect an unseen peak, the models' probability outputs were used to determine the optimal probability threshold for peak detection (0.77389). Then, the final model was trained using all 12 peaks.

As the images below show, the model detected all 12 peak onsets in the signal without missing any or making extra detections. Most of the model's detections (red dashed lines) occurred a few samples before the ground truth positions (black dashed lines).

<img src="/Graphs/delta_RH_moving_avg_model.png">

Ground-truth peak positions: [344, 514, 843, 1166, 1504, 1691, 2030, 2362, 2690, 3035, 3380, 3988]

Model-detected peak positions: [340, 510, 842, 1163, 1500, 1686, 2026, 2359, 2686, 3031, 3375, 3983]

<img src="/Graphs/raw_RH_signal_model.png">

## Conclusion & Considerations

A small six-feature model is enough to detect RH peak onsets. Since it's prepared to handle real-time data, it could also be integrated directly into a sensor. Before actual deployment, it should be exposed to more training and test data, including that from different rooms and areas.
