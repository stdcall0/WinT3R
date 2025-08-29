# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# modified from DUSt3R

import torch


def fill_default_args(kwargs, func):
    import inspect  # a bit hacky but it works reliably

    signature = inspect.signature(func)

    for k, v in signature.parameters.items():
        if v.default is inspect.Parameter.empty:
            continue
        kwargs.setdefault(k, v.default)

    return kwargs


def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:

            module.requires_grad = False


def is_symmetrized(gt1, gt2):
    x = gt1["instance"]
    y = gt2["instance"]
    if len(x) == len(y) and len(x) == 1:
        return False  # special case of batchsize 1
    ok = True
    for i in range(0, len(x), 2):
        ok = ok and (x[i] == y[i + 1]) and (x[i + 1] == y[i])
    return ok


def flip(tensor):
    """flip so that tensor[0::2] <=> tensor[1::2]"""
    return torch.stack((tensor[1::2], tensor[0::2]), dim=1).flatten(0, 1)


def interleave(tensor1, tensor2):
    res1 = torch.stack((tensor1, tensor2), dim=1).flatten(0, 1)
    res2 = torch.stack((tensor2, tensor1), dim=1).flatten(0, 1)
    return res1, res2


def transpose_to_landscape(head, activate=True):
    """Predict in the correct aspect-ratio,
    then transpose the result in landscape
    and stack everything back together.
    """

    def wrapper_no(decout, true_shape, **kwargs):
        B = len(true_shape)
        assert true_shape[0:1].allclose(true_shape), "true_shape must be all identical"
        H, W = true_shape[0].cpu().tolist()
        res = head(decout, (H, W), **kwargs)
        return res

    def wrapper_yes(decout, true_shape, **kwargs):
        B = len(true_shape)

        H, W = int(true_shape.min()), int(true_shape.max())

        height, width = true_shape.T
        is_landscape = width >= height
        is_portrait = ~is_landscape

        if is_landscape.all():
            return head(decout, (H, W), **kwargs)
        if is_portrait.all():
            return transposed(head(decout, (W, H), **kwargs))

        def selout(ar):
            return [d[ar] for d in decout]

        if "pos" in kwargs:
            kwargs_landscape = kwargs.copy()
            kwargs_landscape["pos"] = kwargs["pos"][is_landscape]
            kwargs_portrait = kwargs.copy()
            kwargs_portrait["pos"] = kwargs["pos"][is_portrait]
        l_result = head(selout(is_landscape), (H, W), **kwargs_landscape)
        p_result = transposed(head(selout(is_portrait), (W, H), **kwargs_portrait))

        result = {}
        for k in l_result | p_result:
            x = l_result[k].new(B, *l_result[k].shape[1:])
            x[is_landscape] = l_result[k]
            x[is_portrait] = p_result[k]
            result[k] = x

        return result

    return wrapper_yes if activate else wrapper_no


def transposed(dic):
    return {k: v.swapaxes(1, 2) if v.ndim > 2 else v for k, v in dic.items()}


def invalid_to_nans(arr, valid_mask, ndim=999):
    if valid_mask is not None:
        arr = arr.clone()
        arr[~valid_mask] = float("nan")
    if arr.ndim > ndim:
        arr = arr.flatten(-2 - (arr.ndim - ndim), -2)
    return arr


def invalid_to_zeros(arr, valid_mask, ndim=999):
    if valid_mask is not None:
        arr = arr.clone()
        arr[~valid_mask] = 0
        nnz = valid_mask.view(len(valid_mask), -1).sum(1)
    else:
        nnz = arr.numel() // len(arr) if len(arr) else 0  # number of point per image
    if arr.ndim > ndim:
        arr = arr.flatten(-2 - (arr.ndim - ndim), -2)
    return arr, nnz


def move_to_device(batch, device):
    """ set parameter to gpu or cpu """
    if torch.cuda.is_available():
        if torch.is_tensor(batch):
            # 如果是张量，转移到设备
            return batch.to(device)
        elif isinstance(batch, list):
            # 如果是列表，递归处理每个元素
            return [move_to_device(item, device) for item in batch]
        elif isinstance(batch, dict):
            # 如果是字典，递归处理每个键值对
            return {key: move_to_device(value, device) for key, value in batch.items()}
        else:
            # 如果是其他类型（如 int、float、str），直接返回（不转移）
            return batch
    return batch
