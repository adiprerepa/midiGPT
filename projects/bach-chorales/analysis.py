from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from midigpt import GPT, TetradPlayer, WatermarkLogitsProcessor, WatermarkDetector
from midigpt.datasets import BachChoralesEncoder


def load_chorales(filepaths):
    return [pd.read_csv(filepath).values.tolist() for filepath in filepaths]



jsb_chorales_dir = Path("projects/bach-chorales/jsb_chorales/")
test_files = sorted(jsb_chorales_dir.glob("test/chorale*.csv"))
test_chorales = load_chorales(test_files)

model = GPT.from_checkpoint("projects/bach-chorales/best_model.ckpt", map_location=torch.device("cpu"))
encoder = BachChoralesEncoder()

watermark_processor = WatermarkLogitsProcessor()
watermark_detector = WatermarkDetector(device = 'cpu', tokenizer= encoder)

seed_notes = encoder.encode(torch.as_tensor(test_chorales[10][:8]).flatten())

for temperature in [0.5]:
    #generate unwatermarked
    chords = encoder.decode(model.generate(seed_notes, 20, do_sample=True, temperature=temperature))
    TetradPlayer().to_wav(chords.flatten().reshape(-1, 4).cpu(), f"generated-bach-{temperature}.wav", tempo=220)
    
    #detect unwatermarked
    genchords = TetradPlayer().from_wav(f"generated-bach-{temperature}.wav", tempo=220)
    unwatermarked_detection_result = watermark_detector.detect(text= genchords)
    with open(f"projects/bach-chorales/detection-{temperature}.txt", "w") as f:
        f.write(f"unwatermarked detection result: {unwatermarked_detection_result}\n")
    
    #watermarked
    chords = encoder.decode(model.generate(seed_notes, 20, do_sample=True, temperature=temperature, watermark_proccessor= watermark_processor))
    TetradPlayer().to_wav(chords.flatten().reshape(-1, 4).cpu(), f"generated-bach-{temperature}-watermarked.wav", tempo=220)
    
    gencords = TetradPlayer().from_wav(f"generated-bach-{temperature}-watermarked.wav", tempo=220)
    watermarked_detection_result = watermark_detector.detect(text= genchords)
    with open(f"projects/bach-chorales/detection-{temperature}.txt", "a") as f:
        f.write(f"watermarked detection result: {watermarked_detection_result}\n")
    