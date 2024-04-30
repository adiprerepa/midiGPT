from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os

from midigpt import GPT, TetradPlayer, WatermarkLogitsProcessor, WatermarkDetector
from midigpt.datasets import BachChoralesEncoder
import fire

def load_chorales(filepaths):
    return [pd.read_csv(filepath).values.tolist() for filepath in filepaths]


def main(generated_tokens = 80, tempo = 220, temperatures = "0.5"):
    if not os.path.exists("results/"):
        os.mkdir("results/")
    assert generated_tokens % 4 == 0
    jsb_chorales_dir = Path("projects/bach-chorales/jsb_chorales/")
    test_files = sorted(jsb_chorales_dir.glob("test/chorale*.csv"))
    test_chorales = load_chorales(test_files)

    model = GPT.from_checkpoint("projects/bach-chorales/best_model.ckpt", map_location=torch.device("cpu"))
    encoder = BachChoralesEncoder()

    watermark_processor = WatermarkLogitsProcessor()
    watermark_detector = WatermarkDetector(device = 'cpu', tokenizer= encoder)

    seed_notes = encoder.encode(torch.as_tensor(test_chorales[10][:8]).flatten())
    temperatures = list(map(float, temperatures.split()))
    
    for temperature in temperatures:
        #generate unwatermarked
        chords = encoder.decode(model.generate(seed_notes, generated_tokens, do_sample=True, temperature=temperature))
        chords = chords[:, -generated_tokens:]
        TetradPlayer().to_wav(chords.flatten().reshape(-1, 4).cpu(), f"results/generated-bach-max_tokens={generated_tokens}-tempeature={temperature}-tempo={tempo}.wav", tempo=tempo)
        
        #detect unwatermarked
        genchords = TetradPlayer().from_wav(f"results/generated-bach-max_tokens={generated_tokens}-tempeature={temperature}-tempo={tempo}.wav", tempo=tempo)
        unwatermarked_detection_result = watermark_detector.detect(text= genchords)
        with open(f"results/detection-max_tokens={generated_tokens}-tempeature={temperature}-tempo={tempo}.txt", "w") as f:
            f.write(f"unwatermarked detection result: {unwatermarked_detection_result}\n")
        
        #watermarked
        chords = encoder.decode(model.generate(seed_notes, generated_tokens, do_sample=True, temperature=temperature, watermark_proccessor= watermark_processor))
        chords = chords[:, -generated_tokens:]
        TetradPlayer().to_wav(chords.flatten().reshape(-1, 4).cpu(), f"results/generated-bach-max_tokens={generated_tokens}-tempeature={temperature}-tempo={tempo}-watermarked.wav", tempo=tempo)
        
        genchords = TetradPlayer().from_wav(f"results/generated-bach-max_tokens={generated_tokens}-tempeature={temperature}-tempo={tempo}-watermarked.wav", tempo=tempo)
        watermarked_detection_result = watermark_detector.detect(text= genchords)
        with open(f"results/detection-max_tokens={generated_tokens}-tempeature={temperature}-tempo={tempo}.txt", "a") as f:
            f.write(f"watermarked detection result: {watermarked_detection_result}\n")

if __name__ == '__main__':
  fire.Fire(main)
    

