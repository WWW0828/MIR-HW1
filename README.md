# MIR-HW1
11020 Music Information Retrieval, Su Li (Academia Sinica)
Homework 1 Global key and local key detection of audio and symbolic music

## Prerequisite:
- The following libraries are suggested for this assignment:
    1. [librosa](https://librosa.org/doc/latest/index.html), a Python library for music and audio signal processing:
    2. [pretty-midi](http://craffel.github.io/pretty-midi/), a Python library for MIDI signal processing:
    3. [mir_eval](https://craffel.github.io/mir_eval/), a Python library for MIR evaluation:

## Dataset
[Download Here](https://drive.google.com/drive/folders/1eS_UUX2MrEbEeTVmiDZwIrSW5VBamNrX)
1. GTZAN Dataset
    - 有10種不同曲風，各100首(30 sec 片段，沒頭沒尾)
    - classical 沒給 annotation，其他9種都有，所以管那九種就好
    - A Major: 0, #A Major: 1 ...

2. Schubert Wintereise Dataset (SWD)
    - 德國 Audio Lab 蒐集所有譜，轉成 MIDI，全部都有做 annotation 
    - (有9個版本-可以使用其中4個)

3. The GiantStep Dataset
    - training data


## Task 1: Global key detection based on template matching
- Key detection
    - Key1: 音階的主音
        - 直接假設成是最常出現的音
        - find max average chroma
    - Key2: 音階的全音、半音安排
- Can simply design a Binary Template
    - cannot identify C Major and A minor
- Can also use Krumhansl-Schmuckler template
    - 做內積
- Harmonic template
    - slide 裡面講到每個音的harmonic 要考慮進去
    - 自己建構 (比較進階)
- Data-driven template
    - 可以用giant dataset來挖

### Q1 (40%)
Perform global key finding on the 9 genres in the GTZAN dataset using the feature settings of 1) STFT-based chromagram, 2) CQT chromagram and 3) CENS chromagram and the matching scheme of 1) binary-valued template matching, 2) K-S template matching, and 3) harmonic template matching (you may try 𝛼 = 0.9). Again, since there is no annotation in the classical genre, you don’t need to run that genre. Report the raw accuracy and weighted accuracy per genre and per method. Which genre achieves better performance and why? Which method appear to be more competitive and why? Discuss your results.
- Hint: the chroma features can be obtained from the following functions:
    -  librosa.feature.chroma_stft
    - librosa.feature.chroma_cqt
    - librosa.feature.chroma_cens

- 進 librosa 去用那三個template，做在 GTZAN 來算準確率
    - 調性對: O
    - 調性錯: X
    - random guess: 1/24
- 觀察9種曲風哪個做的比較好


| Relation to correct key | Points |
| :----------------------- | :------ |
|   正確                   |   1    |
|   屬調                   |   0.5  |
|   大小調(C大找成A小)       |   0.3  |
|   平行調(C大找成C小)       |   0.2  |
|   其他                   |   0    |

### Q2 (30%)
Repeat the process in Q1 on the MIDI data and all the available audio versions (i.e., HU33, SC06, FI66, FI80) of the Schubert Winterreise Dataset. Report the average raw accuracy and weighted accuracy for each version. Is there any difference among the versions? Are MIDI data easier for key finding? Discuss your results.
- Hint: for symbolic data, you may use pretty_midi.Instrument.get_chroma to get the chroma vector.

### Q3 (bonus)
Construct the templates for the 24 major/minor keys using the GiantStep dataset. There are many possible ways to construct the templates. There can also be multiple templates for each key. For example, the template of D major can be constructed by taking the average over all chroma vectors annotated as D major in the dataset. We can also take the 𝑘-means algorithm over these chroma vectors to obtain 𝑘 templates for D major. For the keys not in the dataset, you may consider constructing them by circular shifting from the existing keys. Perform global key finding on the GTZAN dataset using the data-driven template. Does this method benefit some genres? Discuss your results.

## Task 2: Local key detection
- Hint: 要算有沒有over segmentation (轉調100次)

### Q4 (20%)
Based on Task 1, design a local key detector that outputs the key of the music every 0.1 second. That means, there is a key detection output for every time step, and in this task, we set the time step be 0.1 second. Perform your method one the MIDI data and all the available audio versions of the Schubert Winterreise Dataset. For simplicity, let’s evaluate the results against the annotator 1. Report the raw accuracy and the weighted accuracy.
- Hint: to get the local tonality feature, you may consider the mean-pooled chroma of a segment (maybe 30 seconds or so), not of the whole music piece. For example, the feature representing the local key at the 60th second can be obtained by summing up the chorma vectors from the 45th to the 75th second.You may try the optimal segment size empirically.
### Q5 (10%)
The local key detection problem can be regarded as a segmentation problem. There has been evaluation metrics for the segmentation performance in the chord recognition problem, but such metrics have not been applied in local key detection. Please apply the over-segmentation, undersegmentation and average segmentation measures (please refer to the directional Hamming divergence and see page 33 in Lecture 3 slides) on the local key detection of the Schubert Winterreise Dataset.
- Hint: these metrics have been implemented somewhere in mir_eval.chord.

### Q6 (bonus)
if possible, please design an algorithm that (hopefully can) outperforms the template matching algorithms introduced here. You may use more advanced method (e.g., deep learning) and novel data representations that you may want to create.

## Notice
- dataset 很多, 要跑很久
- Please submit your .zip file containing the report (PDF) and your codes, with the file name “HW1_[your ID]” to the course website.
- deadline: 4/26
