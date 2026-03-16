# MGFormer


This is the pytorch implementation of paper "MGFormer: A Graph-Enhanced Transformer Framework for Long-Range Gap Imputation of AIS Trajectories"

![model-structure](figures/MGFormer.png)


## Requirements

```
torch==1.7.1
numpy==1.19.2
prettytable==2.0.0
matplotlib==3.3.4
scipy==1.6.1
torch_summary==1.4.5
tqdm==4.58.0
pandas==1.1.5
data==0.4
PyYAML==6.0
scikit_learn==1.0.2
torchsummary==1.5.1
```

## Train
- Unzip `dataset/AIS.zip` to `dataset/AIS`.

- Extract waypoints using the Douglas–Peucker algorithm and DBSCAN.

- Run `build_graph.py` to construct the maritime network.

- Train the model using python `train.py`. All hyper-parameters are defined in `param_parser.py`




