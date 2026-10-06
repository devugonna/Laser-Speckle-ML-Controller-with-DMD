# Data

Large files are not stored in Git. Put them here (or in `results/` for trained models) after downloading:

- **Dataset** (10,000 paired speckle / DMD-pattern images) -> `data/dataset/`
  containing `input_#####.png` and `output32_#####.png`.
  If you have already run stages 02-03 yourself, this folder is created for you.
- **Raw captures** (only produced by `02_data_acquisition/rpatt.m`) -> `data/raw/`
- **Subsystem-test subset** (small sample for `05_evaluation/subsystem_test.m`) -> `data/subsystem_test/{train,val,test}/`
- **Pretrained models** (optional, skips training) -> `results/slnn/` and `results/ppo/`

Download links: _add your GitHub Release / Zenodo / Drive link here_
