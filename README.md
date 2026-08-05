# Simulations Task EEG Workload Dataset (STEW)– subject’s identification 

> dataset's source: 2018 STEW: Simultaneous Task EEG Workload Dataset. Available at: https://dx.doi.org/10.21227/44r8-ya50
> paper: W. L. Lim, O. Sourina and L. P. Wang, "STEW: Simultaneous Task EEG Workload Data Set," in IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 26, no. 11, pp. 2106-2114, Nov. 2018, doi: 10.1109/TNSRE.2018.2872924. https://ieeexplore.ieee.org/document/8478165 

**STEW-SI** is a lightweight artificial neural network based subject’s identification framework implemented in Python with categorical feature support.

**About STEW dataset:**
Dataset can be found at: https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset).

**Dataset Description:** 
This dataset consists of raw EEG data from 48 subjects who participated in a multitasking workload experiment utilizing the SIMKAP multitasking test. The subjects’ brain activity at rest was also recorded before the test and is included as well. The Emotiv EPOC device, with sampling frequency of 128Hz and 14 channels was used to obtain the data, with 2.5 minutes of EEG recording for each case. Subjects were also asked to rate their perceived mental workload after each stage on a rating scale of 1 to 9 and the ratings are provided in a separate file.

**Instructions:** 
The data for each subject follows the naming convention: subno_task.txt. For example, sub01_lo.txt would be raw EEG data for subject 1 at rest, while sub23_hi.txt would be raw EEG data for subject 23 during the multitasking test. The rows of each datafile corresponds to the samples in the recording and the columns corresponds to the 14 channels of the EEG device: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4, respectively.

**Methodology:**
From the pre-processed EEG signal 17 statistical, entropy, and energy features were extracted. For performing the identification task Artificial neural network (ANN) was used. 

## H1: paired rest vs high ASI-EEG across subjects

The key H1 plot compares each subject’s ASI-EEG value in the rest condition against the same subject’s ASI-EEG value in the high-workload condition. Each subject is represented by a paired line connecting the two conditions, while rest is shown in blue and high workload in orange. This visualization emphasizes within-subject change rather than between-subject differences.

The interpretation is direct: if the condition increases cortical arousal-related EEG structure under workload, the paired lines should generally slope upward, and the group median should shift higher in the high condition. In this dataset, the median ASI-EEG increases from rest to high workload (approximately 0.513 to 0.531), with a positive median within-subject difference and a statistically significant Wilcoxon paired test (p = 0.0018). This supports H1: the workload condition shows a measurable increase in the composite ASI-EEG signal relative to rest.

The corresponding proof figures are saved in the Model results directory:
- `Model/results/h1_paired_asi_eeg.png`
- `Model/results/h1_delta_distribution.png`

## Results summary

For the primary H1 test on 48 subjects, the median ASI-EEG was 0.5129 at rest and 0.5311 during the high-workload condition. The paired median change was +0.0218 (95% bootstrap CI: [0.0049, 0.0533]), and the Wilcoxon signed-rank test was significant at p = 0.00182 (two-sided). This supports H1: ASI-EEG is higher during high workload than at rest in this dataset.

### Support

There are many ways to support a project - starring⭐️ the GitHub repos is just one.

