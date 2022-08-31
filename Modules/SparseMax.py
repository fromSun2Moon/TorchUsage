import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import numpy as np
from matplotlib import pyplot as plt


def _make_ix_like(X, dim):
    d = X.size(dim)
    rho = torch.arange(1, d+1, device=X.device, dtype=X.dtype)
    view = [1]*X.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _sparsemax_threshold_and_support(X, dim=-1, k=None):

    if k is None or k >= X.shape[dim]:  # do full sort
        # descending, values & indexes
        topk, _ = torch.sort(X, dim=dim, descending=True)
    else:
        topk, _ = torch.topk(X, k=k, dim=dim)

    #print("topk", topk)
    topk_cumsum = topk.cumsum(dim) - 1
    # [[[[1,2,3,4,5]]]] => size: X.dim() , 1,1,1,X.size(dim=-1)
    rhos = _make_ix_like(topk, dim)
    #print("rhos", rhos)
    #print("rhos*topk", rhos*topk)
    #print("topk_cumsum", topk_cumsum)

    # bool sum => int
    # topk는 X의 큰->작은 분포로 변환
    # topk cumsum은 뒤로 갈수록 작->큰 분포로 변환
    # row의 분포들을 바꿈
    support = rhos * topk > topk_cumsum
    #print("support", support)
    # 넘는 것들만 카운트해서 숫자를 row마다 1개씩 반환 (인덱스 역할)
    support_size = support.sum(dim=dim).unsqueeze(dim)
    #print("support_size", support_size)
    # topk_cumsum에서 경계선을 가름
    tau = topk_cumsum.gather(dim, support_size - 1)
    #print("tau1", tau)
    # [1,1,200,1]
    tau /= support_size.to(X.dtype)
    #print("tau2", tau)
    if k is not None and k < X.shape[dim]:
        unsolved = (support_size == k).squeeze(dim)

        if torch.any(unsolved):
            in_ = _roll_last(X, dim)[unsolved]
            tau_, ss_ = _sparsemax_threshold_and_support(in_, dim=-1, k=2 * k)
            _roll_last(tau, dim)[unsolved] = tau_
            _roll_last(support_size, dim)[unsolved] = ss_

    return tau, support_size


class SparsemaxFunction(Function):
    @classmethod
    def forward(cls, ctx, X, dim=-1, k=None):
        """
        ctx : 역전파 연산을 위한 정보 저장 object
        """
        ctx.dim = dim
        # max_val : column max 1
        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X - max_val  # same numerical stability trick as softmax
        #print("X", X)
        tau, supp_size = _sparsemax_threshold_and_support(X, dim=dim, k=k)
        # z -tau
        output = torch.clamp(X - tau, min=0)
        # backward
        ctx.save_for_backward(supp_size, output)
        return output

    @classmethod
    def backward(cls, ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None, None


def sparsemax(X, dim=-1, k=None):
    return SparsemaxFunction.apply(X, dim, k)


if __name__ == "__main__":
    x = torch.randn(10, 10)
    k = np.random.randint(10)

    for i in range(1, 4):
        k = np.random.randint(10)  # for distributions
        x[k] = x[k]*i

    print("Basic softmax", F.softmax(x[0]))
    print("Sparsemax", sparsemax(x[0]))
