# symmetric_object_detection
Detect symmtric object by using [Segment Anything Model](https://github.com/facebookresearch/segment-anything) (SAM)

This repo use the Segment Anything Model

# To run the code
## Installation of SAM
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```
Noting that pytorch and torchvision is required in prior to install SAM.

Setup SAM checkpoint from SAM and prepare images
```
--- images
  |
  --- *.jpg or *.png
--- models
  |
  --- *.pth
```

## Run the scrip
```
python symmetric_seg.py
```

# Results
Extract the symmetric masks from SAM

<!-- #region -->
<p align="center">
<img  src="outputs/dogs_output.png">
</p>
<!-- #endregion -->

<!-- #region -->
<p align="center">
<img  src="outputs/104e693a955df89d8fb6aff46154844a_output.png">
</p>
<!-- #endregion -->

<!-- #region -->
<p align="center">
<img  src="outputs/17deb0517c2136b00ea1e8a9dfd1d20e_output.png">
</p>
<!-- #endregion -->

<!-- #region -->
<p align="center">
<img  src="outputs/1dcd14d4a97445597fa6df5429d5e3f3_output.png">
</p>
<!-- #endregion -->


