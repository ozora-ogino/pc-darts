from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from genotypes import PRIMITIVES, Genotype
from operations import *


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    Channel shuffle proposed in ShuffleNetV2.
    https://arxiv.org/abs/1807.11164
    """
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class MixedOp(nn.Module):
    def __init__(self, C: int, stride: int, k: int = 4):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.mp = nn.MaxPool2d(2, 2)
        self.k = k
        for primitive in PRIMITIVES:
            # Partial connection.
            op = OPS[primitive](C // self.k, stride, False)
            if "pool" in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C // self.k, affine=False))
            self._ops.append(op)

    def forward(self, x: torch.Tensor, weights_alpha: torch.Tensor) -> torch.Tensor:
        # Partial Connection: channel proportion k=4.
        dim_2 = x.shape[1]
        op_inputs = x[:, : dim_2 // self.k, :, :]
        skip = x[:, dim_2 // self.k :, :, :]

        # Input node is Weighted by alpha.
        op_outs = sum(w * op(op_inputs) for w, op in zip(weights_alpha, self._ops))
        # Reduction cell needs pooling before concat.
        skip = self.mp(skip) if op_outs.shape[2] != x.shape[2] else skip
        ans = torch.cat([op_outs, skip], dim=1)
        ans = channel_shuffle(ans, self.k)
        return ans


class Cell(nn.Module):
    def __init__(
        self,
        steps: int,
        multiplier: int,
        C_prev_prev: int,
        C_prev: int,
        C: int,
        reduction: bool,
        reduction_prev: bool,
    ):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)

        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(
        self,
        s0: torch.Tensor,
        s1: torch.Tensor,
        weights_alpha: torch.Tensor,
        weights_beta: torch.Tensor,
    ) -> torch.Tensor:
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for _ in range(self._steps):
            s = sum(
                weights_beta[offset + j] * self._ops[offset + j](h, weights_alpha[offset + j])
                for j, h in enumerate(states)
            )
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier :], dim=1)


class Network(nn.Module):
    def __init__(
        self,
        C: int,
        num_classes: int,
        layers: int,
        criterion: Callable,
        steps: int = 4,
        multiplier: int = 4,
        stem_multiplier: int = 3,
    ):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(nn.Conv2d(3, C_curr, 3, padding=1, bias=False), nn.BatchNorm2d(C_curr))

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(
                steps,
                multiplier,
                C_prev_prev,
                C_prev,
                C_curr,
                reduction,
                reduction_prev,
            )
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        s0 = s1 = self.stem(input)
        for _, cell in enumerate(self.cells):
            weights_alpha, weights_beta = self._forward_search(cell)
            s0, s1 = s1, cell(s0, s1, weights_alpha, weights_beta)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _forward_search(self, cell: Cell) -> Tuple[torch.Tensor, torch.Tensor]:
        if cell.reduction:
            weights_alpha = F.softmax(self.alphas_reduce, dim=-1)
            n = 3
            start = 2
            weights_beta = F.softmax(self.betas_reduce[0:2], dim=-1)
            for _ in range(self._steps - 1):
                end = start + n
                tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
                start = end
                n += 1
                weights_beta = torch.cat([weights_beta, tw2], dim=0)
        else:
            weights_alpha = F.softmax(self.alphas_normal, dim=-1)
            n = 3
            start = 2
            weights_beta = F.softmax(self.betas_normal[0:2], dim=-1)
            for _ in range(self._steps - 1):
                end = start + n
                tw2 = F.softmax(self.betas_normal[start:end], dim=-1)
                start = end
                n += 1
                weights_beta = torch.cat([weights_beta, tw2], dim=0)
        return weights_alpha, weights_beta

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.betas_normal = Variable(1e-3 * torch.randn(k).cuda(), requires_grad=True)
        self.betas_reduce = Variable(1e-3 * torch.randn(k).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
            self.betas_normal,
            self.betas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        def _parse(weights_alpha, weights_beta):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W_ALPHA = weights_alpha[start:end].copy()
                W_BETA = weights_beta[start:end].copy()
                for j in range(n):
                    W_ALPHA[j, :] = W_ALPHA[j, :] * W_BETA[j]
                edges = sorted(
                    range(i + 2),
                    key=lambda x: -max(W_ALPHA[x][k] for k in range(len(W_ALPHA[x])) if k != PRIMITIVES.index("none")),
                )[:2]

                for j in edges:
                    k_best = None
                    for k in range(len(W_ALPHA[j])):
                        if k != PRIMITIVES.index("none"):
                            if k_best is None or W_ALPHA[j][k] > W_ALPHA[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        n = 3
        start = 2
        weightsr2 = F.softmax(self.betas_reduce[0:2], dim=-1)
        weightsn2 = F.softmax(self.betas_normal[0:2], dim=-1)
        for _ in range(self._steps - 1):
            end = start + n
            tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
            tn2 = F.softmax(self.betas_normal[start:end], dim=-1)
            start = end
            n += 1
            weightsr2 = torch.cat([weightsr2, tw2], dim=0)
            weightsn2 = torch.cat([weightsn2, tn2], dim=0)
        gene_normal = _parse(
            F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(),
            weightsn2.data.cpu().numpy(),
        )
        gene_reduce = _parse(
            F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(),
            weightsr2.data.cpu().numpy(),
        )

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal,
            normal_concat=concat,
            reduce=gene_reduce,
            reduce_concat=concat,
        )
        return genotype
