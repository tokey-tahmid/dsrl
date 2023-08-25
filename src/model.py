import os
import torch
import random

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SurrGradSpike(torch.autograd.Function):
    scale = 100.0

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad


class DSNN(nn.Module):
    def __init__(self, architecture, seed, alpha, beta, weight_scale, batch_size, threshold,
                 simulation_time, learning_rate, reset_potential=0):
        self.architecture = architecture

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.alpha = alpha
        self.beta = beta
        self.weight_scale = weight_scale
        self.batch_size = batch_size
        self.threshold = threshold
        self.simulation_time = simulation_time
        self.reset_potential = reset_potential

        self.spike_fn = SurrGradSpike.apply

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.weights = []
        for i in range(len(architecture) - 1):
            self.weights.append(torch.empty((self.architecture[i], self.architecture[i + 1]),
                                            device=device, dtype=torch.float, requires_grad=True))
            torch.nn.init.normal_(self.weights[i], mean=0.0,
                                  std=self.weight_scale/np.sqrt(self.architecture[i]))

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, inputs):
        syn = []
        mem = []
        spk_count = []

        for l in range(0, len(self.weights)):
            syn.append(torch.zeros((self.batch_size, self.weights[l].shape[1]), device=device,
                                   dtype=torch.float))
            mem.append(torch.zeros((self.batch_size, self.weights[l].shape[1]), device=device,
                                   dtype=torch.float))
        mem_rec = []
        spk_rec = []
        for t in range(self.simulation_time):
            mem_rec.append([])
            spk_rec.append([])

            if t == 0:
                for l in range(len(self.weights)):
                    mem_rec[-1].append(mem[l])
                    spk_rec[-1].append(mem[l])
                continue
            input = inputs.detach().clone()
            for l in range(len(self.weights)):
                if l == 0:
                    h = torch.einsum("ab,bc->ac", [input, self.weights[0]])
                    new_syn = 0 * syn[l] + h
                else:
                    h = torch.einsum("ab,bc->ac", [spk_rec[-1][l - 1], self.weights[l]])
                    new_syn = self.alpha * syn[l] + h

                new_mem = self.beta*mem[l] + new_syn
                if l < (len(self.weights) - 1):
                    mthr = new_mem
                    mthr = mthr - self.threshold
                    out = self.spike_fn(mthr)
                    c = (mthr > 0)
                    new_mem[c] = self.reset_potential
                    spk_rec[-1].append(out)

                mem[l] = new_mem
                syn[l] = new_syn

                mem_rec[-1].append(mem[l])
        return mem_rec[-1][-1], mem_rec, spk_rec

    def load_state_dict(self, layers):
        """Method to load weights and biases into the network"""
        weights = layers[0]
        for l in range(0,len(weights)):
            self.weights[l] = weights[l].detach().clone().requires_grad_(True)

    def state_dict(self):
        """Method to copy the layers of the SQN. Makes explicit copies, no references."""
        weights_copy = []
        bias_copy = []
        for l in range(0, len(self.weights)):
            weights_copy.append(self.weights[l].detach().clone())
        return weights_copy, bias_copy

    def parameters(self):
        parameters = []
        for l in range(0, len(self.weights)):
            parameters.append(self.weights[l])

        return parameters

