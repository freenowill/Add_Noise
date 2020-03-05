import numpy as np
import librosa
import os
from tqdm import tqdm
import random
def add_noise(x, d, SNR):
    assert(len(d)>len(x))
    d=d[:len(x)]
    P_signal=np.sum(abs(x)**2)
    P_d=np.sum(abs(d)**2)
    P_noise=P_signal/10**(SNR/10)
    noise=np.sqrt(P_noise/P_d)*d
    noise_signal=x+noise
    return noise_signal

#加高斯白噪声
def wgn(x, snr):
    P_signal = np.sum(abs(x)**2)/len(x)
    P_noise = P_signal/10**(snr/10.0)
    return np.random.randn(len(x)) * np.sqrt(P_noise)

def preprocess(x,d,SNR=5):
    data=add_noise(x,d,SNR)
    data=np.float32(data)
    return data

if __name__ == '__main__':
    from utils.util import load_audio,save_audio
    path="D:/ftp/hey_snips_kws_4.0/hey_snips/train/train_positive_vad/"
    random_noise="D:/ftp/hey_snips_kws_4.0/random/"
    out_path="D:/ftp/hey_snips_kws_4.0/hey_snips/train/train_positive_vad(random_noise_SNR_5db)/"
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    ids=os.listdir(path)
    noises=os.listdir(random_noise)

    for id in tqdm(ids):
        x = load_audio(path+id)

        noise = noises[random.randint(0, len(noises) - 1)]
        d = load_audio(random_noise+noise)
        while len(d)<len(x):
            noise = noises[random.randint(0, len(noises) - 1)]
            d = load_audio(random_noise + noise)
        data=preprocess(x,d,SNR=5)
        save_audio(out_path+"-snr5db-"+noise+"-"+id,data)
