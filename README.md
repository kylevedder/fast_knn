# Fast KNN

Based heavily on [PyTorch3d's KNN implementation](https://github.com/facebookresearch/pytorch3d/blob/4ae25bfce7eb42042a34585acc3df81cf4be7d85/pytorch3d/csrc/knn/knn.cu), this is a fast KNN implementation for PyTorch. 

It has most of the features of PyTorch3d's KNN removed to focus on Nx3 point clouds, and designed to be simple to hack on to speed up your own code.