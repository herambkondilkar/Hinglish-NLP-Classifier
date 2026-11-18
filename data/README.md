# Dataset files

Dataset files are intentionally not committed to this repository. The official COMI-LINGUA source files are large and should be downloaded from the dataset maintainer using `python download_comi_lingua.py`. The converter writes derived token files to `data/processed/comi_lingua/`.

Local annotation and augmentation files (for example `data/word_labels.csv` and `data/real_world_hinglish_eval.csv`) are not included. The augmented training script requires these files if you want to reproduce the locally augmented model; the supplied desktop application model can be used without them.
