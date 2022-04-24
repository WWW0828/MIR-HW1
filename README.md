# MIR-HW1
- 11020 Music Information Retrieval, Su Li (Academia Sinica)
- Homework 1 Global key and local key detection of audio and symbolic music

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
- Step1
    - 算出 chromagram
- Step2
    - 取每個 frame 的平均
- Step3
    - 找出最大值當作主音
- Step4
    - 對照 template 找出最有可能的大小調
- Step5
    - 計算精準度 (raw/weighted)

### Q1 (40%)
Perform global key finding on the 9 genres in the GTZAN dataset using the feature settings of 1) STFT-based chromagram, 2) CQT chromagram and 3) CENS chromagram and the matching scheme of 1) binary-valued template matching, 2) K-S template matching, and 3) harmonic template matching (you may try 𝛼 = 0.9). Again, since there is no annotation in the classical genre, you don’t need to run that genre. Report the raw accuracy and weighted accuracy per genre and per method. Which genre achieves better performance and why? Which method appear to be more competitive and why? Discuss your results.
- Hint: the chroma features can be obtained from the following functions:
    - librosa.feature.chroma_stft
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

#### Result
- Binary Template

| genres  | stft score | cqt score | cens score |
|---------|------------|-----------|------------|
| blues   | 0.112245   | 0.174490  | 0.170408   |
| country | 0.683838   | 0.768687  | 0.753535   |
| disco   | 0.451020   | 0.457143  | 0.481633   |
| hiphop  | 0.111111   | 0.185185  | 0.204938   |
| jazz    | 0.455696   | 0.430380  | 0.388608   |
| metal   | 0.155914   | 0.170968  | 0.201075   |
| pop     | 0.538298   | 0.439362  | 0.448936   |
| reggae  | 0.547423   | 0.490722  | 0.479381   |
| rock    | 0.558163   | 0.581633  | 0.558163   |

- KS Template

| genres  | stft score | cqt score | cens score |
|---------|------------|-----------|------------|
| blues   | 0.326531   | 0.334694  | 0.286735   |
| country | 0.668687   | 0.795960  | 0.747474   |
| disco   | 0.534694   | 0.614286  | 0.584694   |
| hiphop  | 0.316049   | 0.320988  | 0.302469   |
| jazz    | 0.408861   | 0.475949  | 0.445570   |
| metal   | 0.507527   | 0.598925  | 0.579570   |
| pop     | 0.680851   | 0.638298  | 0.607447   |
| reggae  | 0.626804   | 0.594845  | 0.537113   |
| rock    | 0.592857   | 0.685714  | 0.673469   |

- Harmonic Template

| genres  | stft score | cqt score | cens score |
|---------|------------|-----------|------------|
| blues   | 0.184694   | 0.252041  | 0.197959   |
| country | 0.811111   | 0.573737  | 0.636364   |
| disco   | 0.512245   | 0.464286  | 0.427551   |
| hiphop  | 0.129630   | 0.220988  | 0.208642   |
| jazz    | 0.432911   | 0.441772  | 0.435443   |
| metal   | 0.236559   | 0.348387  | 0.320430   |
| pop     | 0.578723   | 0.309574  | 0.377660   |
| reggae  | 0.527835   | 0.490722  | 0.448454   |
| rock    | 0.593878   | 0.511224  | 0.526531   |

### Q2 (30%)
Repeat the process in Q1 on the MIDI data and all the available audio versions (i.e., HU33, SC06, FI66, FI80) of the Schubert Winterreise Dataset. Report the average raw accuracy and weighted accuracy for each version. Is there any difference among the versions? Are MIDI data easier for key finding? Discuss your results.
- Hint: for symbolic data, you may use pretty_midi.Instrument.get_chroma to get the chroma vector.
- Problem
    - `librosa.load` doesn't support mp3 files
- Solve
    - download ffmpeg package [here](https://drive.google.com/drive/folders/18EjrdBXhrhj_8K3yiaBWIl5x8WyQLQX1?usp=sharing)
    - put the `FFmpeg` folder in `C:\`
    - add `C:\FFmpeg\bin` to path (system variables)
        - win + s
        - search `system variables`
        - system: path
        - click New
        - paste `C:\FFmpeg\bin`
        - click OK

#### Result
- Binary Template
    - midi
        - score: 0.641667
    - audio

|Version|stft score|cqt score|cens score|
|-------|----------|---------|----------|
| FI66 | 0.512500 | 0.812500 | 0.783333 |
| FI80 | 0.445833 | 0.645833 | 0.625000 |
| HU33 | 0.625000 | 0.791667 | 0.741667 |
| SC06 | 0.554167 | 0.754167 | 0.716667 |

- KS Template
    - midi
        - score: 0.791667
    - audio

|Version|stft score|cqt score|cens score|
|-------|----------|---------|----------|
| FI66 | 0.545833 | 0.737500 | 0.675000 |
| FI80 | 0.687500 | 0.800000 | 0.825000 |
| HU33 | 0.612500 | 0.687500 | 0.666667 |
| SC06 | 0.666667 | 0.654167 | 0.645833 |

- Harmonic Template
    - midi
        - score: 0.545833
    - audio

|Version|stft score|cqt score|cens score|
|-------|----------|---------|----------|
| FI66 | 0.687500 | 0.862500 | 0.800000 |
| FI80 | 0.433333 | 0.625000 | 0.687500 |
| HU33 | 0.708333 | 0.666667 | 0.708333 |
| SC06 | 0.595833 | 0.695833 | 0.725000 |

### Q3 (bonus)
Construct the templates for the 24 major/minor keys using the GiantStep dataset. There are many possible ways to construct the templates. There can also be multiple templates for each key. For example, the template of D major can be constructed by taking the average over all chroma vectors annotated as D major in the dataset. We can also take the 𝑘-means algorithm over these chroma vectors to obtain 𝑘 templates for D major. For the keys not in the dataset, you may consider constructing them by circular shifting from the existing keys. Perform global key finding on the GTZAN dataset using the data-driven template. Does this method benefit some genres? Discuss your results.

## Task 2: Local key detection
- Hint: 要算有沒有over segmentation (轉調100次)

### Q4 (20%)
Based on Task 1, design a local key detector that outputs the key of the music every 0.1 second. That means, there is a key detection output for every time step, and in this task, we set the time step be 0.1 second. Perform your method one the MIDI data and all the available audio versions of the Schubert Winterreise Dataset. For simplicity, let’s evaluate the results against the annotator 1. Report the raw accuracy and the weighted accuracy.
- Hint: to get the local tonality feature, you may consider the mean-pooled chroma of a segment (maybe 30 seconds or so), not of the whole music piece. For example, the feature representing the local key at the 60th second can be obtained by summing up the chorma vectors from the 45th to the 75th second.You may try the optimal segment size empirically.

#### Result
- Binary Template
    - midi
        - score: 
    - audio

|Version|stft score|cqt score|cens score|
|-------|----------|---------|----------|
| FI66 | 0.154255 | 0.189565 | 0.185779 |
| FI80 | 0.138596 | 0.181742 | 0.183270 |
| HU33 | 0.153278 | 0.181481 | 0.179828 |
| SC06 | 0.163004 | 0.176554 | 0.174731 |

- KS Template
    - midi
        - score: 
    - audio

|Version|stft score|cqt score|cens score|
|-------|----------|---------|----------|
| FI66 |  |  |  |
| FI80 |  |  |  |
| HU33 |  |  |  |
| SC06 |  |  |  |

- Harmonic Template
    - midi
        - score: 
    - audio

|Version|stft score|cqt score|cens score|
|-------|----------|---------|----------|
| FI66 |  |  |  |
| FI80 |  |  |  |
| HU33 |  |  |  |
| SC06 |  |  |  |

### Q5 (10%)
The local key detection problem can be regarded as a segmentation problem. There has been evaluation metrics for the segmentation performance in the chord recognition problem, but such metrics have not been applied in local key detection. Please apply the over-segmentation, under-segmentation and average segmentation measures (please refer to the directional Hamming divergence and see page 33 in Lecture 3 slides) on the local key detection of the Schubert Winterreise Dataset.
- Hint: these metrics have been implemented somewhere in mir_eval.chord.

### Q6 (bonus)
if possible, please design an algorithm that (hopefully can) outperforms the template matching algorithms introduced here. You may use more advanced method (e.g., deep learning) and novel data representations that you may want to create.

## Notice
- dataset 很多, 要跑很久
- Please submit your .zip file containing the report (PDF) and your codes, with the file name “HW1_[your ID]” to the course website.
- deadline: 4/26
