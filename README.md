# Isolation Forest and variants
This repository contains Python implementations of the Isolation Forest algorithm and its variants: FairCutForest and SciForest. These algorithms are used for anomaly detection in high-dimensional datasets. The only requirement is the numpy library.

## Algorithms

- Isolation Forest (or iForest): A tree-based algorithm for anomaly detection that isolates observations by randomly selecting features and splitting values. http://www.lamda.nju.edu.cn/publication/icdm08b.pdf
- SciForest: Replaces uniform random splits by generating a linear combination of a selected 
number of features at each step and then finds the best split point guided by a gain criterion 
that minimizes the average standard deviation between the two split sets. https://link.springer.com/chapter/10.1007/978-3-642-15883-4_18
- Fair Cut Forest: Uses similar metric as SciForest but it takes into account
not only the homogeneity of the two split sets, but also the difference in size, so the average of the standard deviations is weighted. https://arxiv.org/pdf/2110.13402
- Fair Cut Forest with random step: use fully random split instead of metric with set probability 

## Running
```python
from src.FairCutForest import FairCutForest

# n_trees, sample_size, k_planes, extension_level
fcf = FairCutForest(100, 256, 1, 'full')
fcf.fit(data)

pred = fcf.predict(data)
```

## Demo
Check out the `eval.ipynb` jupyter notebook for some demos of evaluation and visualizations of the results.

## License
This code was created as a part of [my diploma thesis](https://dspace.cvut.cz/bitstream/handle/10467/108884/F8-DP-2023-Kramar-Maros-thesis.pdf). You are welcome to use it in accordance with the thesis license.
