from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from midigpt import GPT, TetradPlayer
from midigpt.datasets import BachChoralesEncoder


def load_chorales(filepaths):
    return [pd.read_csv(filepath).values.tolist() for filepath in filepaths]



jsb_chorales_dir = Path("projects/bach-chorales/jsb_chorales/")
test_files = sorted(jsb_chorales_dir.glob("test/chorale*.csv"))
test_chorales = load_chorales(test_files)

model = GPT.from_checkpoint("projects/bach-chorales/best_model.ckpt", map_location=torch.device("cpu"))
encoder = BachChoralesEncoder()

seed_notes = encoder.encode(torch.as_tensor(test_chorales[10][:8]).flatten())
# #     print("After Gen: ", chords.shape)
# TetradPlayer().to_wav(sn, f"generated-bach-0.5.wav", tempo=220)
# genchords = TetradPlayer().from_wav("generated-bach-0.5.wav", tempo=220)

# # print("Fileread: ", torch.as_tensor(genchords))


for temperature in [0.5]:
    chords = encoder.decode(model.generate(seed_notes, 120, do_sample=True, temperature=temperature))
    print("After Gen: ", chords.shape)
    TetradPlayer().to_wav(chords.flatten().reshape(-1, 4).cpu(), f"generated-bach-{temperature}.wav", tempo=220)
    print(chords.flatten().reshape(-1, 4).shape)

genchords = TetradPlayer().from_wav("generated-bach-0.5.wav", tempo=220)