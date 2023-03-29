# Normalised Dice Score (nDSC)

The current repo contains implementations of the Normalised Dice score (nDSC) described in our [paper]().
nDSC is a corrected Dice score decorrelated from the class count within an image. Decorrelation is provided by replacing 
the class load of the image in the Dice formulae with an effective class load. The choice of effective class load is left
for the user. We could, however, recommend using average across the dataset class load.

Requirements
---

* python 3.8
* monai==0.9.0
* numpy==1.23.0
* torch==1.12.0

Usage
---

**"ndsc_numpy.py" module** contains implementation of the nDSC for `numpy.ndarray` inputs for one class only in 
`ndsc_metric(y_pred: np.ndarray, y: np.ndarray, effective_load: float = 0.001, check: bool = False) -> float` function. 
The dimensionality of images is arbitrary.

*Example 2D images:*
```python
from ndsc_numpy import ndsc_metric
import cv2
# loading images to numpy arrays
ground_truth = cv2.imread("apple_gt.jpeg")
prediction = cv2.imread("apple_bin_pred.jpeg")
ndsc = ndsc_metric(y_pred=prediction, y=ground_truth, effective_load=0.1)
```

**"numpy_pytorch.py" module** contains PyTorch-based implementation of nDSC implemented in [MONAI library style](https://docs.monai.io/en/stable/metrics.html) fashion.
Particularly, we provide a class `NormalisedDiceMetric`, which uses function
`compute_ndice(y_pred: torch.Tensor, y: torch.Tensor, effective_load: Union[torch.Tensor, float], include_background: bool = True, ignore_empty: bool = True) -> torch.Tensor`.
Implementations can be used for 2D or 3D images. Please, refer to functions' docstring documentation for more information about the usage.


*Example 3D images:*
```python
from ndsc_pytorch import NormalisedDiceMetric, compute_ndice
from monai.transforms import LoadImage, AddChannel, ToTensor, Compose
# loading images to torch.Tensors of shape [B=1, C=1, H, W, D]
transforms = Compose([
    LoadImage(), AddChannel(), AddChannel(), ToTensor()
])
ground_truth = transforms("gt.nii.gz")
prediction = transforms("binarised_pred.nii.gz")
# through class
ndsc_metric = NormalisedDiceMetric(effective_load=0.002)
ndsc = ndsc_metric(y_pred=prediction, y=ground_truth)
# or through function
ndsc = compute_ndice(y_pred=prediction, y=ground_truth, effective_load=0.002)
```

Citation
---

Please, use the following citation if using nDSC in your work:

```text
@misc{raina2023tackling,
      title={Tackling Bias in the Dice Similarity Coefficient: Introducing nDSC for White Matter Lesion Segmentation}, 
      author={Vatsal Raina and Nataliia Molchanova and Mara Graziani and Andrey Malinin and Henning Muller and Meritxell Bach Cuadra and Mark Gales},
      year={2023},
      eprint={2302.05432},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
