from collections.abc import Callable
from functools import wraps
from typing import Any

import torch
from torch import Tensor

__all__ = ["euler_integrate", "heun_integrate", "midpoint_integrate", "integrators"]

integrators: dict[str, Callable[..., Tensor]] = {}


def register_integrator(
    name: str,
) -> Callable[[Callable[..., Tensor]], Callable[..., Tensor]]:
    def decorator(
        func: Callable[..., Tensor],
    ) -> Callable[..., Tensor]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Tensor:
            return func(*args, **kwargs)

        integrators[name] = wrapper
        return wrapper

    return decorator


@register_integrator("euler")
def euler_integrate(
    ode: Callable[[Tensor, Tensor], Tensor],
    x0: Tensor,
    t0: float,
    t1: float,
    steps: int,
    **kwargs,
) -> Tensor:
    t0_ = torch.as_tensor(t0, dtype=x0.dtype, device=x0.device)
    t1_ = torch.as_tensor(t1, dtype=x0.dtype, device=x0.device)

    x = x0
    t = t0_
    dt = (t1_ - t0_) / steps
    for _ in range(steps):
        x = x + dt * ode(t, x, **kwargs)
        t = t + dt

    return x


@register_integrator("heun")
def heun_integrate(
    ode: Callable[[Tensor, Tensor], Tensor],
    x0: Tensor,
    t0: float,
    t1: float,
    steps: int,
    **kwargs,
) -> Tensor:
    t0_ = torch.as_tensor(t0, dtype=x0.dtype, device=x0.device)
    t1_ = torch.as_tensor(t1, dtype=x0.dtype, device=x0.device)

    x = x0
    t = t0_
    dt = (t1_ - t0_) / steps
    for _ in range(steps):
        df = ode(t, x, **kwargs)
        y_ = x + dt * df
        x = x + dt / 2 * (df + ode(t + dt, y_, **kwargs))
        t = t + dt

    return x


@register_integrator("midpoint")
def midpoint_integrate(
    ode: Callable[[Tensor, Tensor], Tensor],
    x0: Tensor,
    t0: float,
    t1: float,
    steps: int,
    **kwargs,
) -> Tensor:
    t0_ = torch.as_tensor(t0, dtype=x0.dtype, device=x0.device)
    t1_ = torch.as_tensor(t1, dtype=x0.dtype, device=x0.device)

    x = x0
    t = t0_
    dt = (t1_ - t0_) / steps
    for _ in range(steps):
        df = ode(t, x, **kwargs)
        y_ = x + dt / 2 * df
        x = x + dt * ode(t + dt / 2, y_, **kwargs)
        t = t + dt

    return x
