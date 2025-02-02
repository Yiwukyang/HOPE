# HOPE
The source code for "Improving Hypergraph Contrastive Learning with Hyperbolic Space and Dual-level Expansion."
Here we offer three datasets, Cora, Citeseer, and PubMed for reproducibility.
# Requirement
    Python == 3.11
    PyTorch == 2.0.1 + cu118
    torch_geometric == 2.0.4
    torch_cluster == 1.6.3
    torch_scatter == 2.1.2
    torch_sparse == 0.6.18
    torch_spline_conv == 1.2.2
    scikit-learn == 1.5.2
    numpy == 1.26.3
# Usage
You need to make the following two parameter adjustments after replacing the dataset：

**Firstly**, in the line **parser.add_argument(‘--c_hypergraph’, type=float, default=1)** under the **train.py**, 
corresponding to the different datasets, you need to change the c_hypergraph, which hyperparameter can be found in the paper or in config.yaml.

**Secondly**, you need to change the value of **Threshold** in the **utils.py**, 
which hyperparameter can be found in the paper or in config.yaml.

Then, you can run the following the command.

```
python train.py
```
