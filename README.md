# NeuralMCS 

## Datasets

Retrieve the datasets from:
https://drive.google.com/drive/folders/1cVuGx8SFKxgqUUk5FRCltJVPrtoiMjMW?usp=sharing
Save the klepto files to the NeuralMCS/save/dataset/ directory.

## Dependencies

Reference commands:

```pip3 install torch torchvision```

```pip3 install -r requirements.txt```

If you use macOS, check the website for PyTorch Geometric.
You may need to use `MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ pip install torch-scatter torch-cluster torch-spline-conv`.

## Run

Modify model/Our/config.py and run 

```cd model/Our && python3 main.py```
