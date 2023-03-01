import math
from typing import Tuple, Union

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, TensorDataset
import PrepareDataset
from UNET import UNet


class VariationalDropout(nn.Module):

    def _parse_shape(self, shape):
        try:
            self.shape = tuple(shape)
        except TypeError:
            pass
        if self.shape is not None:
            return
        self.shape = (int(shape),)

    def __init__(self, shape: Union[int, Tuple[int, ...]], alpha: float = 0.99, p_init=0.5):
        """
        Variational Gaussian Dropout as described in https://proceedings.mlr.press/v70/molchanov17a.html
        :param shape: Shape of the dropout weight tensors
        :param p_init: Initial dropout rate
        """
        super(VariationalDropout, self).__init__()
        self.shape = None
        self._parse_shape(shape)
        assert isinstance(self.shape, tuple)
        self.register_parameter(
            '_log_beta', nn.Parameter(
                torch.zeros(1, *self.shape),
                requires_grad=True
            )
        )
        self.register_parameter(
            '_log_var', nn.Parameter(
                torch.full_like(self._log_beta.data, self._get_init_log_var(p_init)),
                requires_grad=True
            )
        )

    @staticmethod
    def _get_init_log_var(p_init):
        # alpha = p / (1 - p) = sigma ** 2 / beta ** 2, beta_init = 1 -> log(sigma ** 2) = log(p / (1 - p))
        return float(np.log(p_init / (1. - p_init)))

    def forward(self, x: Tensor) -> Tensor:
        unsqueeze = [1 for _ in x.shape[(len(self.shape) + 1):]]
        w = self.beta.view(1, *self.shape, *unsqueeze)
        w = w + torch.exp(self._log_var / 2.).view(1, *self.shape, *unsqueeze) * torch.randn_like(x)
        return x * w

    @property
    def beta(self):
        return torch.exp(self._log_beta)

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    @property
    def log_alpha(self):
        return self._log_var - 2 * self._log_beta

    @property
    def p(self):
        """
        Dropout rate
        :return: tensor of dropout rates
        """
        alpha = self.alpha
        return alpha / (1. + alpha)


class DropoutKLDivergence(nn.Module):
    """
    Approximate KL Divergence for Gaussian dropout as described in https://proceedings.mlr.press/v70/molchanov17a.html
    """

    _k1 = 0.63576
    _k2 = 1.87320
    _k3 = 1.48695

    def __init__(self):
        super(DropoutKLDivergence, self).__init__()
        self.penalties = []

    def _collect_penalty(self, module: nn.Module):
        if isinstance(module, VariationalDropout):
            log_alpha = module.log_alpha
            t1 = self._k1 * torch.sigmoid(self._k2 + self._k3 * log_alpha)
            t2 = 0.5 * F.softplus(-log_alpha, beta=1.)
            self.penalties.append(torch.sum(- t1 + t2 + self._k1))

    def forward(self, model: nn.Module):
        model.apply(self._collect_penalty)
        if len(self.penalties):
            loss = sum(self.penalties)
            self.penalties = []
        else:
            loss = 0.
        return loss


class _ILogLikelihood(nn.Module):
    """
    Interface for computing log-likelihoods:
    """

    def get_prediction_channels(self, p: int) -> int:
        """
        Compute number of expected prediction channels for a field of p parameters
        :param p: number of predicted parameters
        :return: number of expected prediction channels
        """
        raise NotImplementedError()

    def forward(self, target: Tensor, prediction: Tensor) -> Tensor:
        """
        Compute expected log likelihood of a batch of targets, given a batch of predictions
        :param prediction: batch of predictions
        :param target: batch of targets
        :return: expected log likelihood
        """
        raise NotImplementedError()


class GaussianLogLikelihood(_ILogLikelihood):
    """
    Log likelihood based on (locally independent) Gaussian posteriors
    for quantities in (-infty, +infty),
    """

    _log_norm = math.log(2 * np.pi) / 2.

    def get_prediction_channels(self, p: int) -> int:
        # Gaussian log likelihood requires predictions of mean and log-var for each parameter
        return 2 * int(p)

    def forward(self, target: Tensor, prediction: Tensor) -> Tensor:
        mu, log_var = torch.chunk(prediction, 2, dim=1)
        x = (target - mu) / torch.exp(log_var / 2.)
        ll = - 0.5 * x ** 2 - self._log_norm - 0.5 * log_var
        # compute batch mean and sum over all spatial locations and parameters
        return torch.sum(torch.mean(ll, dim=0))


class BetaLogLikelihood(_ILogLikelihood):
    """
    Log likelihood based on (locally independent) Beta posteriors
    for quantities in (0, 1)
    """

    def __init__(self, eps=1.e-6):
        super(BetaLogLikelihood, self).__init__()
        self.eps = float(eps)
        self._log_bound = math.log(self.eps)

    def get_prediction_channels(self, p: int) -> int:
        # Beta distribution expects predictions for parameters a and b of the Beta distribution for each parameter
        return 3 * int(p)

    def forward(self, target: Tensor, prediction: Tensor) -> Tensor:
        # parameterization ensures unimodal shape due to a >= 1 or b >= 1
        n, pa, pb,  = torch.chunk(torch.exp(prediction), 3, dim=1)
        n = n + 1.
        t = pa + pb
        a = n * pa / t
        b = n * pb / t
        norm = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
        # clamp target to (0, 1) to avoid numerical issues
        ll = (
                (a - 1.) * torch.clip(torch.log(target), min=self._log_bound)
                + (b - 1.) * torch.clip(torch.log1p(- target), min=self._log_bound)
        ) - norm
        return torch.sum(torch.mean(ll, dim=0))


class TruncatedLogisticLogLikelihood(_ILogLikelihood):
    """
    Log likelihood with (locally independent) truncated logistic posterior for quantities in [lower_bound, +infty)
    """

    def __init__(self, lower_bound=0., eps=1.e-6):
        super(TruncatedLogisticLogLikelihood, self).__init__()
        self.lower_bound = lower_bound
        self.eps = eps

    def get_prediction_channels(self, p: int) -> int:
        # Logistic distribution requires location parameter mu and scale parameter sigma
        return 2 * int(p)

    def _compute_ll(self, target: Tensor, prediction: Tensor) -> Tensor:
        mu, sigma = torch.chunk(F.softplus(prediction), 2, dim=1)
        sigma = sigma + self.eps
        target_red = (target - mu) / sigma
        lower_bound = (self.lower_bound - mu) / sigma
        log_cdf_0 = - lower_bound - F.softplus(-lower_bound)
        log_pdf = - target_red - 2. * F.softplus(-target_red) - torch.log(sigma)
        ll = log_pdf - log_cdf_0
        return ll

    def forward(self, target: Tensor, prediction: Tensor) -> Tensor:
        ll = self._compute_ll(target, prediction)
        return torch.sum(torch.mean(ll, dim=0))


class TruncatedLogisticOrLowerLogLikelihood(TruncatedLogisticLogLikelihood):
    """
    Log likelihood with two cases: Value below threshold or value distributed as truncated logistic
    """

    def __init__(self, threshold=1.e-4, eps=1.e-6):
        super(TruncatedLogisticOrLowerLogLikelihood, self).__init__(lower_bound=threshold, eps=eps)
        self.threshold = threshold

    def get_prediction_channels(self, p: int) -> int:
        # requires 2 parameters for case probabilities and 2 values for TL distribution
        return 4 * int(p)

    def forward(self, target: Tensor, prediction: Tensor) -> Tensor:
        prediction_p, prediction_tl = torch.chunk(prediction, 2, dim=1)
        ll_tl = self._compute_ll(target, prediction_tl)
        log_p_higher, log_p_lower = torch.chunk(prediction_p, 2, dim=1)
        # reduce max for numerical stability
        _max = torch.maximum(log_p_higher, log_p_lower)
        log_norm = _max + torch.log(torch.exp(log_p_lower - _max) + torch.exp(log_p_higher - _max))
        ll = torch.where(
            target <= self.threshold,
            log_p_lower,
            ll_tl + log_p_higher
        )
        ll = ll - log_norm
        return torch.sum(torch.mean(ll, dim=0))

    def get_log_p_higher(self, predictions: Tensor):
        log_p_higher, log_p_lower, *_ = torch.chunk(predictions, 4, dim=1)
        _max = torch.maximum(log_p_higher, log_p_lower)
        log_norm = _max + torch.log(torch.exp(log_p_lower - _max) + torch.exp(log_p_higher - _max))
        return log_p_higher - log_norm




def build_dummy_data(
        num_inputs, num_targets,
        num_samples=2000, resolution=(32, 64), noise_level=1.e-4,
        activation= None
) -> Dataset:
    """
    Build random dummy dataset
    -> To be replaced with the actual WeatherBench dataset
    :param num_inputs: number of input parameters
    :param num_outputs: number of target parameters
    :param num_samples: number of samples in the dataset
    :param resolution: spatial resolution
    :return: dataset
    """
    inputs = torch.randn(num_samples, num_inputs, *resolution)
    mapping = torch.randn(num_inputs, num_targets)
    targets = torch.einsum('bihw,ij->bjhw', inputs, mapping) + noise_level * torch.randn(num_samples, num_targets, *resolution)
    if activation is not None:
        targets = activation(targets)
    return TensorDataset(inputs, targets)












input_indices = torch.zeros([7,6], dtype=torch.int)
out_index = torch.zeros([7,1], dtype=torch.int)
for i in range(7):
    out_index[i] = i
    count = 0
    for j in range(7):
        if(i == j):
            continue
        input_indices[i][count] = j
        count += 1



def main():
    num_parameters = 7
    num_targets = 1
    num_inputs = num_parameters - num_targets

    mode = 'unconstrained' # check other modes as well


    train_dataset = PrepareDataset.ValDataset()
    train_loader = torch.utils.data.DataLoader(train_dataset)


    if mode == 'unconstrained':
        # Data range (-infty, + infty) and Gaussian posterior: e.g. temperature, pressure, u10, v10, ...
        log_likelihood = GaussianLogLikelihood()
    elif mode == 'positive':
        # positive data and truncated logistic posterior: e.g. precipitation
        log_likelihood = TruncatedLogisticOrLowerLogLikelihood()
        # for precipitation the explicit treatment of the no-rain case may be useful.
        # But the implementation of the LL-computation in TruncatedLogisticOrLowerLogLikelihood needs double-checking
        # i.e. if used, it should be verified that learned probabilities are correct/sensible (see get_log_p_higher(...))
    elif mode == 'bounded':
        # unit interval data: e.g. cloudcover
        log_likelihood = BetaLogLikelihood()
    else:
        raise NotImplementedError()

    kl_divergence = DropoutKLDivergence()
    kl_weight = 1.
    # default weight should be one 1., but can be reduced or increased if too much or not enough dropout is observed

    p_init = 0.5
    dropout = VariationalDropout(num_inputs, p_init=p_init)
    # initial dropout rate may be lowered to admit more information to pass through the network at training start

    num_outputs = log_likelihood.get_prediction_channels(num_targets)
    model = nn.Sequential(
        dropout,
        UNet(num_inputs, num_outputs),
    )


    optimizer = Adam(model.parameters(), lr=1.e-4)
    loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

    def train():
        model.train()
        for batch in loader:
            model.zero_grad()

            inputs = batch.index_select(dim=1, index=input_indices[0]).to(dtype=torch.float32)
            targets = batch.index_select(dim=1, index=out_index[0]).to(dtype=torch.float32)

            predictions = model(inputs)

            ll = log_likelihood(targets, predictions)
            kl_loss = kl_divergence(model)
            # Note the negative sign of the log-likelihood term in the loss function!
            loss = - ll + kl_weight * kl_loss

            loss.backward()
            optimizer.step()

            # print('Batch loss:', loss.item())

    num_epochs = 20
    for e in range(num_epochs):
        train()
        print(f'Epoch: {e}')
        print(dropout.p)

    print('Done')


if __name__ == '__main__':
    main()
