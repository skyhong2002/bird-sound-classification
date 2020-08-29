# tweet-net
## Pre-process
### Overview
1. MFCC
2. split to 90 * 10 (~ 10 seconds.) time step. Drop length < 450.
3. save them  as np file
```
<dataset>
    class_A
        1.npy
        2.npy
    class_B
        1.npy
```
### Process
1. preparing audio files.
```
data/
    class_A/
        1.mp3
        2.mp3
    class_B/
        1.mp3
        2.mp3
    ....
```
2. run preprocess.py with the following command:
```bash
    N_THREAD=32 python3 preprocess.py
```
It would take ~ 4 hr on Xeon Gold (72T).
3. split train/val/test
```bash
python3 dataset.py
```

## Training
```bash
python3 train.py
```



## Troubleshooting

### Preprocess.py "audioread no backend"
 - install ffmpeg
     - using conda:
        ```bash
        conda install ffmpeg
        ```

