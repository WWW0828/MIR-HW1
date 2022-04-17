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

### Q3 (bonus)

## Task 2: Local key detection
要算有沒有over segmentation (轉調100次)

### Q4 (20%)
### Q5 (10%)
### Q6 (bonus)

## Notice
- dataset 很多
- 要跑很久
