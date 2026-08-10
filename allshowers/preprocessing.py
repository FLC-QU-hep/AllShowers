import math

import torch
from torch import nn

__all__ = [
    "Transformation",
    "Sequence",
    "Identity",
    "Log",
    "LogIt",
    "Affine",
    "StandardScaler",
    "RobustScaler",
    "MinMaxScaler",
    "compose",
]


class Transformation(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class Sequence(Transformation):
    def __init__(self, modules: list[Transformation]) -> None:
        super().__init__()
        self.sub_modules = nn.ModuleList(modules)

    def fit(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for module in self.sub_modules:
            if isinstance(module, Transformation):
                x = module.fit(x, mask)
            else:
                raise TypeError(
                    f"Expected module of type Transformation, got {type(module)}"
                )
        return x

    def forward(self, x: torch.Tensor):
        for module in self.sub_modules:
            x = module.forward(x)
        return x

    def inverse(self, x: torch.Tensor):
        for i in range(len(self.sub_modules) - 1, -1, -1):
            module = self.sub_modules[i]
            if isinstance(module, Transformation):
                x = module.inverse(x)
            else:
                raise TypeError(
                    f"Expected module of type Transformation, got {type(module)}"
                )
        return x


class Identity(Transformation):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Log(Transformation):
    def __init__(self, alpha: float = 1e-6, base: float = math.e) -> None:
        super().__init__()
        self.alpha = alpha
        self.log_base = math.log(base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x + self.alpha) / self.log_base

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_base * x) - self.alpha


class LogIt(Transformation):
    def __init__(self, alpha: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (1 - 2 * self.alpha) * x + self.alpha
        return torch.log(x / (1 - x))

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sigmoid(x)
        return (x - self.alpha) / (1 - 2 * self.alpha)


class Affine(Transformation):
    def __init__(self, scale: float = 1.0, shift: float = 0.0) -> None:
        super().__init__()
        self.a = scale
        self.b = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a * x + self.b

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b) / self.a


class StandardScaler(Transformation):
    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("std", torch.ones(shape))
        self.shape = shape

    def fit(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.bool)
        x = torch.where(mask, x, 0.0)
        dims = tuple(torch.where(torch.tensor(self.shape) == 1)[0].tolist())
        mean = torch.sum(x * mask, dim=dims, keepdim=True)
        mean /= torch.sum(mask, dim=dims, keepdim=True)
        self.mean = mean
        x = x - mean
        std = torch.sqrt(torch.sum(x**2 * mask, dim=dims, keepdim=True))
        std /= torch.sqrt(torch.sum(mask, dim=dims, keepdim=True) - 1)
        std[std == 0] = 1
        self.std = std
        x = x / std
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


class RobustScaler(Transformation):
    """Scales using median and IQR, robust to outliers."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.register_buffer("median", torch.zeros(shape))
        self.register_buffer("iqr", torch.ones(shape))
        self.shape = shape

    def fit(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)

        # Squeeze mask if it has trailing dimension
        if mask.dim() == x.dim():
            mask = mask.squeeze(-1)

        # x shape: (batch, points, features) or (batch, features)
        # mask shape: (batch, points) or (batch,)
        n_features = x.shape[-1]
        medians = []
        iqrs = []

        for i in range(n_features):
            if x.dim() == 3:
                valid = x[:, :, i][mask]
            else:
                valid = x[:, i][mask] if mask.dim() > 0 else x[:, i]

            median = torch.quantile(valid.float(), 0.5)
            q75 = torch.quantile(valid.float(), 0.75)
            q25 = torch.quantile(valid.float(), 0.25)
            iqr = q75 - q25
            if iqr == 0:
                iqr = torch.tensor(1.0, device=x.device)

            medians.append(median)
            iqrs.append(iqr)

        self.median = torch.stack(medians).reshape(self.shape)
        self.iqr = torch.stack(iqrs).reshape(self.shape)

        return (x - self.median) / self.iqr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.median) / self.iqr

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.iqr + self.median


class MinMaxScaler(Transformation):
    """Scales data to a target range [target_min, target_max]."""

    def __init__(
        self,
        shape: tuple[int, ...],
        target_min: float = 0.0,
        target_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.register_buffer("data_min", torch.zeros(shape))
        self.register_buffer("data_max", torch.ones(shape))
        self.target_min = target_min
        self.target_max = target_max
        self.shape = shape

    def fit(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.bool)
        dims = tuple(torch.where(torch.tensor(self.shape) == 1)[0].tolist())
        x_masked = torch.where(mask, x, torch.inf)
        data_min = torch.amin(x_masked, dim=dims, keepdim=True)
        x_masked = torch.where(mask, x, -torch.inf)
        data_max = torch.amax(x_masked, dim=dims, keepdim=True)
        # Avoid division by zero
        data_range = data_max - data_min
        data_range[data_range == 0] = 1
        self.data_min = data_min
        self.data_max = data_max
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        data_range = self.data_max - self.data_min
        data_range[data_range == 0] = 1
        x_scaled = (x - self.data_min) / data_range
        return x_scaled * (self.target_max - self.target_min) + self.target_min

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        data_range = self.data_max - self.data_min
        x_scaled = (x - self.target_min) / (self.target_max - self.target_min)
        return x_scaled * data_range + self.data_min


def compose(transformation: list[list[str | dict | list | None]] | None) -> Sequence:
    if transformation is None:
        return Sequence([Identity()])
    trafo_list = []
    attrs = globals()
    for element in transformation:
        if (element[0] not in __all__) or (
            element[0] in ["Transformation", "Sequence", "compose"]
        ):
            raise ValueError(f"Invalid transformation: {element[0]}")
        Trafo = attrs[element[0]]
        assert issubclass(Trafo, Transformation)
        if len(element) == 1 or element[1] is None:
            trafo_list.append(Trafo())
        elif isinstance(element[1], list):
            trafo_list.append(Trafo(*element[1]))
        elif isinstance(element[1], dict):
            trafo_list.append(Trafo(**element[1]))
        else:
            raise ValueError(
                f"argument for {element[0]} must be a list or a dict not {type(element[1])}"
            )

    return Sequence(trafo_list)
