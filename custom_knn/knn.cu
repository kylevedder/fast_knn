/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <float.h>
#include <iostream>
#include <tuple>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "mink.cuh"

// A chunk of work is blocksize-many points of P1.
// The number of potential chunks to do is N*(1+(P1-1)/blocksize)
// call (1+(P1-1)/blocksize) chunks_per_cloud
// These chunks are divided among the gridSize-many blocks.
// In block b, we work on chunks b, b+gridSize, b+2*gridSize etc .
// In chunk i, we work on cloud i/chunks_per_cloud on points starting from
// blocksize*(i%chunks_per_cloud).


template <typename scalar_t, int D, int K>
__global__ void KNearestNeighborKernelV3(
    const scalar_t* __restrict__ points1,
    const scalar_t* __restrict__ points2,
    const int64_t* __restrict__ lengths1,
    const int64_t* __restrict__ lengths2,
    scalar_t* __restrict__ dists,
    int64_t* __restrict__ idxs,
    const size_t P1,
    const size_t P2) {
  // Same idea as V2, but use register indexing for thread-local arrays.
  // Enabling sorting for this version leads to huge slowdowns; I suspect
  // that it forces min_dists into local memory rather than registers.
  // As a result this version is always unsorted.
  scalar_t cur_point[D];
  scalar_t min_dists[K];
  int min_idxs[K];
  const int64_t chunks_per_cloud = (1 + (P1 - 1) / blockDim.x);
  const int64_t chunks_to_do = chunks_per_cloud;
  for (int64_t chunk = blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    const int64_t n = chunk / chunks_per_cloud;
    const int64_t start_point = blockDim.x * (chunk % chunks_per_cloud);
    int64_t p1 = start_point + threadIdx.x;
    if (p1 >= lengths1[n])
      continue;
    for (int d = 0; d < D; ++d) {
      cur_point[d] = points1[n * P1 * D + p1 * D + d];
    }
    int64_t length2 = lengths2[n];
    RegisterMinK<scalar_t, int, K> mink(min_dists, min_idxs);
    for (int p2 = 0; p2 < length2; ++p2) {
      scalar_t dist = 0;
      for (int d = 0; d < D; ++d) {
        int offset = n * P2 * D + p2 * D + d;
        scalar_t diff = cur_point[d] - points2[offset];
        scalar_t norm_diff = diff * diff;
        dist += norm_diff;
      }
      mink.add(dist, p2);
    }
    for (int k = 0; k < mink.size(); ++k) {
      idxs[n * P1 * K + p1 * K + k] = min_idxs[k];
      dists[n * P1 * K + p1 * K + k] = min_dists[k];
    }
  }
}

std::tuple<at::Tensor, at::Tensor> KNearestNeighborIdxCuda(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2) {
  // Check inputs are on the same device
  at::TensorArg p1_t{p1, "p1", 1}, p2_t{p2, "p2", 2},
      lengths1_t{lengths1, "lengths1", 3}, lengths2_t{lengths2, "lengths2", 4};
  at::CheckedFrom c = "KNearestNeighborIdxCuda";
  at::checkAllSameGPU(c, {p1_t, p2_t, lengths1_t, lengths2_t});
  at::checkAllSameType(c, {p1_t, p2_t});

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(p1.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();


  const auto P1 = p1.size(0);
  const auto P2 = p2.size(0);
  TORCH_CHECK(p1.size(1) == 3, "Point sets must have 3 dim");
  TORCH_CHECK(p2.size(1) == 3, "Point sets must have 3 dim");
  auto long_dtype = lengths1.options().dtype(at::kLong);
  auto idxs = at::zeros({P1}, long_dtype);
  auto dists = at::zeros({P1}, p1.options());

  if (idxs.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(idxs, dists);
  }

  const size_t threads = 256;
  const size_t blocks = 256;

  AT_DISPATCH_FLOATING_TYPES(p1.scalar_type(), "knn_kernel_cuda", ([&] {
                                KNearestNeighborKernelV3<scalar_t, 3, 1><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                                    p1.contiguous().data_ptr<scalar_t>(),
                                    p2.contiguous().data_ptr<scalar_t>(),
                                    lengths1.contiguous().data_ptr<int64_t>(),
                                    lengths2.contiguous().data_ptr<int64_t>(),
                                    dists.data_ptr<scalar_t>(),
                                    idxs.data_ptr<int64_t>(),
                                    P1,
                                    P2);
                              }));
  
  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(idxs, dists);
}

// ------------------------------------------------------------- //
//                   Backward Operators                          //
// ------------------------------------------------------------- //

// TODO(gkioxari) support all data types once AtomicAdd supports doubles.
// Currently, support is for floats only.
__global__ void KNearestNeighborBackwardKernel(
    const float* __restrict__ p1, // (N, P1, D)
    const float* __restrict__ p2, // (N, P2, D)
    const int64_t* __restrict__ lengths1, // (N,)
    const int64_t* __restrict__ lengths2, // (N,)
    const int64_t* __restrict__ idxs, // (N, P1, K)
    const float* __restrict__ grad_dists, // (N, P1, K)
    float* __restrict__ grad_p1, // (N, P1, D)
    float* __restrict__ grad_p2, // (N, P2, D)
    const size_t P1,
    const size_t P2) {
  const size_t D = 3;
  const size_t N = 1;
  const size_t K = 1;
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = gridDim.x * blockDim.x;

  for (size_t i = tid; i < N * P1 * K * D; i += stride) {
    const size_t n = i / (P1 * K * D); // batch index
    size_t rem = i % (P1 * K * D);
    const size_t p1_idx = rem / (K * D); // index of point in p1
    rem = rem % (K * D);
    const size_t k = rem / D; // k-th nearest neighbor
    const size_t d = rem % D; // d-th dimension in the feature vector

    const size_t num1 = lengths1[n]; // number of valid points in p1 in batch
    const size_t num2 = lengths2[n]; // number of valid points in p2 in batch
    if ((p1_idx < num1) && (k < num2)) {
      const float grad_dist = grad_dists[n * P1 * K + p1_idx * K + k];
      // index of point in p2 corresponding to the k-th nearest neighbor
      const int64_t p2_idx = idxs[n * P1 * K + p1_idx * K + k];
      // If the index is the pad value of -1 then ignore it
      if (p2_idx == -1) {
        continue;
      }
      float diff = 2.0 * grad_dist *
            (p1[n * P1 * D + p1_idx * D + d] - p2[n * P2 * D + p2_idx * D + d]);
      atomicAdd(grad_p1 + n * P1 * D + p1_idx * D + d, diff);
      atomicAdd(grad_p2 + n * P2 * D + p2_idx * D + d, -1.0f * diff);
    }
  }
}

std::tuple<at::Tensor, at::Tensor> KNearestNeighborBackwardCuda(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    const at::Tensor& idxs,
    const at::Tensor& grad_dists) {
  // Check inputs are on the same device
  at::TensorArg p1_t{p1, "p1", 1}, p2_t{p2, "p2", 2},
      lengths1_t{lengths1, "lengths1", 3}, lengths2_t{lengths2, "lengths2", 4},
      idxs_t{idxs, "idxs", 5}, grad_dists_t{grad_dists, "grad_dists", 6};
  at::CheckedFrom c = "KNearestNeighborBackwardCuda";
  at::checkAllSameGPU(
      c, {p1_t, p2_t, lengths1_t, lengths2_t, idxs_t, grad_dists_t});
  at::checkAllSameType(c, {p1_t, p2_t, grad_dists_t});

  // This is nondeterministic because atomicAdd
  at::globalContext().alertNotDeterministic("KNearestNeighborBackwardCuda");

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(p1.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const auto P1 = p1.size(0);
  const auto P2 = p2.size(0);
  const auto D = 3;
  const auto K = 1;
  
  TORCH_CHECK(
      idxs.size(0) == P1, "KNN idxs must have the same point dimension as p1");
  TORCH_CHECK(grad_dists.size(0) == P1);

  auto grad_p1 = at::zeros({P1, D}, p1.options());
  auto grad_p2 = at::zeros({P2, D}, p2.options());

  if (grad_p1.numel() == 0 || grad_p2.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(grad_p1, grad_p2);
  }

  const int blocks = 64;
  const int threads = 512;

  KNearestNeighborBackwardKernel<<<blocks, threads, 0, stream>>>(
      p1.contiguous().data_ptr<float>(),
      p2.contiguous().data_ptr<float>(),
      lengths1.contiguous().data_ptr<int64_t>(),
      lengths2.contiguous().data_ptr<int64_t>(),
      idxs.contiguous().data_ptr<int64_t>(),
      grad_dists.contiguous().data_ptr<float>(),
      grad_p1.data_ptr<float>(),
      grad_p2.data_ptr<float>(),
      P1,
      P2);

  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(grad_p1, grad_p2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn_forward", &KNearestNeighborIdxCuda, "KNN forward (CUDA)");
    m.def("knn_backward", &KNearestNeighborBackwardCuda, "KNN backward (CUDA)");
}
