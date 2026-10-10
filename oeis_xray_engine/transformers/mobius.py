"""Exact arithmetic-function Dirichlet convolution and Möbius inversion.

Arrays in this module use one-based arithmetic-function indices: element zero
of an input represents the function value at ``n = 1``, element one represents
``n = 2``, and so on. No floating-point arithmetic is used.
"""

from __future__ import annotations

from collections.abc import Sequence

from . import TransformError


def _validate_integer_sequence(values: Sequence[int], name: str) -> tuple[int, ...]:
    """Validate and detach a finite sequence of exact Python integers."""

    if isinstance(values, (str, bytes, bytearray)) or not isinstance(
        values, Sequence
    ):
        raise TransformError(f"{name} must be a finite integer sequence")
    result = tuple(values)
    for index, value in enumerate(result):
        if type(value) is not int:
            raise TransformError(
                f"{name}[{index}] must be an integer",
                context={"actual_type": type(value).__name__},
            )
    return result


def _mobius_values(n_max: int) -> list[int]:
    """Return ``mu(0)..mu(n_max)`` using an exact linear sieve."""

    if type(n_max) is not int or n_max < 0:
        raise TransformError("n_max must be a non-negative integer")
    if n_max == 0:
        return [0]

    mu = [0] * (n_max + 1)
    least_prime = [0] * (n_max + 1)
    primes: list[int] = []
    mu[1] = 1

    for value in range(2, n_max + 1):
        if least_prime[value] == 0:
            least_prime[value] = value
            primes.append(value)
            mu[value] = -1

        for prime in primes:
            composite = value * prime
            if composite > n_max:
                break
            least_prime[composite] = prime
            if value % prime == 0:
                mu[composite] = 0
                break
            mu[composite] = -mu[value]

    return mu


def dirichlet_convolution(
    f: Sequence[int],
    g: Sequence[int],
    n_max: int | None = None,
) -> list[int]:
    """Compute the Dirichlet convolution ``(f * g)(n)`` for ``1 <= n <= N``.

    ``f`` and ``g`` are stored as ``[f(1), f(2), ...]`` and must contain every
    divisor value needed through ``N``. When ``n_max`` is omitted, both input
    sequences must have equal lengths and that length is used. When it is
    supplied, both inputs must contain at least ``n_max`` values.
    """

    left = _validate_integer_sequence(f, "f")
    right = _validate_integer_sequence(g, "g")
    if n_max is None:
        if len(left) != len(right):
            raise TransformError(
                "f and g must have equal lengths when n_max is omitted",
                context={"len_f": len(left), "len_g": len(right)},
            )
        n_max = len(left)
    if type(n_max) is not int or n_max < 0:
        raise TransformError("n_max must be a non-negative integer")
    if len(left) < n_max or len(right) < n_max:
        raise TransformError(
            "input does not contain every required divisor value",
            context={
                "n_max": n_max,
                "len_f": len(left),
                "len_g": len(right),
            },
        )

    result = [0] * n_max
    for divisor in range(1, n_max + 1):
        f_value = left[divisor - 1]
        for multiple in range(divisor, n_max + 1, divisor):
            quotient = multiple // divisor
            result[multiple - 1] += f_value * right[quotient - 1]
    return result


def mobius_invert(values: Sequence[int]) -> list[int]:
    """Invert a divisor-sum sequence under Dirichlet convolution.

    If ``values[n - 1] = sum(f[d - 1] for d | n)``, return ``f`` exactly.
    Every positive divisor required by the inversion is present in ``values``.
    """

    transformed = _validate_integer_sequence(values, "values")
    n_max = len(transformed)
    mu = _mobius_values(n_max)
    result = [0] * n_max

    for divisor in range(1, n_max + 1):
        mobius_value = mu[divisor]
        if mobius_value == 0:
            continue
        for multiple in range(divisor, n_max + 1, divisor):
            result[multiple - 1] += (
                mobius_value * transformed[multiple // divisor - 1]
            )
    return result


def divisor_sum_transform(source: Sequence[int]) -> list[int]:
    """Return ``F(n) = sum(f(d) for d | n)`` for each supplied ``n``.

    ``source`` is stored one-based as ``[f(1), ..., f(N)]``. The implementation
    visits divisor/multiple pairs and uses arbitrary-precision integer sums.
    """

    function = _validate_integer_sequence(source, "source")
    n_max = len(function)
    result = [0] * n_max

    for divisor, value in enumerate(function, start=1):
        for multiple in range(divisor, n_max + 1, divisor):
            result[multiple - 1] += value
    return result
