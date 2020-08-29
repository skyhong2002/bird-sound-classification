from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

from multiprocessing import Pool
from glob import glob

DS_DIR = "/home/skyhong9109/data/green_bird_disk1/綠繡眼聲音/dataset"
OUTPUT_DIR = "./dataset/raw"
'''
DS_DIR
  - class_1
    - 1.wav
    - 2.wav
  - class_2
  ...
'''

def job(parm):
    audiopath, class_name  = parm
    audiotype = 'wav'  # 如果wav、mp4其他格式参看pydub.AudioSegment的API
    chunks_path = f"{OUTPUT_DIR}/{class_name}"
    print(chunks_path)
    sound = AudioSegment.from_file(audiopath, format=audiotype)
    # 分割
    print('开始分割')
    chunks = split_on_silence(sound, min_silence_len=2000,keep_silence=200,
                              silence_thresh=-35)  # min_silence_len: 拆分语句时，静默满0.3秒则拆分。silence_thresh：小于-70dBFS以下的为静默。
    # 创建保存目录
    filepath = os.path.split(audiopath)[0]
    try:
        if not os.path.exists(chunks_path): os.mkdir(chunks_path)
    except Exception as e:
        print(e)

    # 保存所有分段
    print('开始保存')
    for i in range(len(chunks)):
        new = chunks[i]
        save_name = chunks_path + '/%04d.%s' % (i, audiotype)
        new.export(save_name, format=audiotype)
        print('%04d' % i, len(new))
    print('保存完毕')


l = []
CLASSES = ["before_net", "build_nest","hug_egg", "out_nest", "spawn", "yu_zhu_qi"]

for c in CLASSES:
    for audiopath in glob(f"{DS_DIR}/{c}/*.wav"):
        l.append((audiopath, c))

with Pool(72) as p:
    p.map(job, l)
