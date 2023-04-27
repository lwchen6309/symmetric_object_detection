# symmetric_object_detection
Detect symmtric object by using Segment Anything Model (SAM)

This repo use the Segment Anything Model

To run the code
Installation of SAM
pip install git+https://github.com/facebookresearch/segment-anything.git
Noting that pytorch and torchvision is required in prior to install SAM.

Setup SAM checkpoint from SAM and prepare images
--- images
  |
  --- *.jpg or *.png
--- models
  |
  --- *.pth
Run the scrip
python symmetric_seg.py
Results
