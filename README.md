# Watermarking Music Generation

A fork of https://github.com/johnnygreco/midiGPT

Uses methods from https://arxiv.org/abs/2301.10226 to watermark chorales generation in `projects/bach-chorales`. This is a decoder-only model.

Authors: Aditya Prerepa, Tarun Suresh, Ayush Raman, Aditya Korlahalli

UIUC Phil 380 Sp 24 Final Project

## Details & Running

Flask webapp available in `app/app.py`

Actual Watermarking code is in `src/midigpt/watermarking.py`, and watermark detection code is available in `projects/bach-chorales/analysis.py`.