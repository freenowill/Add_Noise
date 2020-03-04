import numpy as np
import librosa
import os
from tqdm import tqdm
def add_noise(x, d, SNR):
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

def preprocess(path,id,x,d=None,SNR=5):
    if d==None:
        d=wgn(x,SNR)
    data=add_noise(x,d,SNR)
    data=np.float32(data)
    return data

if __name__ == '__main__':
    from utils.util import load_audio,save_audio
    path="D:/ftp/hey_snips_kws_4.0/hey_snips/dev/dev_positive_vad/"
    out_path="D:/ftp/hey_snips_kws_4.0/hey_snips/dev/dev_positive_vad_SNR(5db)/"
    ids=os.listdir(path)
    for id in tqdm(ids):
        x = load_audio(path+id)
        data=preprocess(path,id,x,SNR=5)
        save_audio(out_path+id,data)
