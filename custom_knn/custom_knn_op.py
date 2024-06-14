import torch
import custom_knn_ext as _C
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from dataclasses import dataclass


@dataclass
class _KNN:
    dists: torch.Tensor
    idx: torch.Tensor


class _knn_points(Function):
    """
    Torch autograd Function wrapper for KNN C++/CUDA implementations.
    """

    @staticmethod
    # type: ignore
    def forward(
        ctx,
        p1,
        p2,
    ):
        """
        K-Nearest neighbors on point clouds.

        Args:
            p1: Tensor of shape (N, P1, D) giving a batch of N point clouds, each
                containing up to P1 points of dimension D.
            p2: Tensor of shape (N, P2, D) giving a batch of N point clouds, each
                containing up to P2 points of dimension D.
            lengths1: LongTensor of shape (N,) of values in the range [0, P1], giving the
                length of each pointcloud in p1. Or None to indicate that every cloud has
                length P1.
            lengths2: LongTensor of shape (N,) of values in the range [0, P2], giving the
                length of each pointcloud in p2. Or None to indicate that every cloud has
                length P2.
            K: Integer giving the number of nearest neighbors to return.
            version: Which KNN implementation to use in the backend. If version=-1,
                the correct implementation is selected based on the shapes of the inputs.
            norm: (int) indicating the norm. Only supports 1 (for L1) and 2 (for L2).
            return_sorted: (bool) whether to return the nearest neighbors sorted in
                ascending order of distance.

        Returns:
            p1_dists: Tensor of shape (N, P1, K) giving the squared distances to
                the nearest neighbors. This is padded with zeros both where a cloud in p2
                has fewer than K points and where a cloud in p1 has fewer than P1 points.

            p1_idx: LongTensor of shape (N, P1, K) giving the indices of the
                K nearest neighbors from points in p1 to points in p2.
                Concretely, if `p1_idx[n, i, k] = j` then `p2[n, j]` is the k-th nearest
                neighbors to `p1[n, i]` in `p2[n]`. This is padded with zeros both where a cloud
                in p2 has fewer than K points and where a cloud in p1 has fewer than P1 points.
        """

        idx, dists = _C.knn_forward(
            p1,
            p2,
        )  # type: ignore

        ctx.save_for_backward(p1, p2, idx)
        ctx.mark_non_differentiable(idx)
        return dists, idx

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists, grad_idx):
        p1, p2, idx = ctx.saved_tensors
        grad_p1, grad_p2 = _C.knn_backward(p1, p2, idx, grad_dists)
        return grad_p1, grad_p2, None, None, None, None, None, None


def knn_points(
    p1: torch.Tensor,
    p2: torch.Tensor,
) -> _KNN:
    """
    K-Nearest neighbors on point clouds.

    Args:
        p1: Tensor of shape (N, P1, D) giving a batch of N point clouds, each
            containing up to P1 points of dimension D.
        p2: Tensor of shape (N, P2, D) giving a batch of N point clouds, each
            containing up to P2 points of dimension D.
        lengths1: LongTensor of shape (N,) of values in the range [0, P1], giving the
            length of each pointcloud in p1. Or None to indicate that every cloud has
            length P1.
        lengths2: LongTensor of shape (N,) of values in the range [0, P2], giving the
            length of each pointcloud in p2. Or None to indicate that every cloud has
            length P2.
        norm: Integer indicating the norm of the distance. Supports only 1 for L1, 2 for L2.
        K: Integer giving the number of nearest neighbors to return.
        version: Which KNN implementation to use in the backend. If version=-1,
            the correct implementation is selected based on the shapes of the inputs.
        return_nn: If set to True returns the K nearest neighbors in p2 for each point in p1.
        return_sorted: (bool) whether to return the nearest neighbors sorted in
            ascending order of distance.

    Returns:
        dists: Tensor of shape (N, P1, K) giving the squared distances to
            the nearest neighbors. This is padded with zeros both where a cloud in p2
            has fewer than K points and where a cloud in p1 has fewer than P1 points.

        idx: LongTensor of shape (N, P1, K) giving the indices of the
            K nearest neighbors from points in p1 to points in p2.
            Concretely, if `p1_idx[n, i, k] = j` then `p2[n, j]` is the k-th nearest
            neighbors to `p1[n, i]` in `p2[n]`. This is padded with zeros both where a cloud
            in p2 has fewer than K points and where a cloud in p1 has fewer than P1
            points.

        nn: Tensor of shape (N, P1, K, D) giving the K nearest neighbors in p2 for
            each point in p1. Concretely, `p2_nn[n, i, k]` gives the k-th nearest neighbor
            for `p1[n, i]`. Returned if `return_nn` is True.
            The nearest neighbors are collected using `knn_gather`

            .. code-block::

                p2_nn = knn_gather(p2, p1_idx, lengths2)

            which is a helper function that allows indexing any tensor of shape (N, P2, U) with
            the indices `p1_idx` returned by `knn_points`. The output is a tensor
            of shape (N, P1, K, U).

    """
    if p1.shape[1] != 3:
        raise ValueError("pts1 must have 3 point dimension.")
    if p2.shape[1] != 3:
        raise ValueError("pts2 must have 3 point dimension.")

    p1 = p1.contiguous()
    p2 = p2.contiguous()

    p1_dists, p1_idx = _knn_points.apply(p1, p2)
    return _KNN(dists=p1_dists, idx=p1_idx)
