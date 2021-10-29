# MolGym3

## Installation

```
pip install -r requirements.txt
```

```bash
CUDA="cpu" # or, for instance, "cu102"

pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.8.1+${CUDA}.html
pip install torch-sparse==0.6.12 -f https://pytorch-geometric.com/whl/torch-1.8.1+${CUDA}.html
```