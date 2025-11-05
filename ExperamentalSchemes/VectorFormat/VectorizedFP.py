from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN, getcontext
from typing import Sequence, Tuple, Union

NumberLike = Union[int, float, Decimal]

# We routinely scale values by base ** frac_limbs, so bump precision to avoid
# surprises when we build Decimal representations from floats.
getcontext().prec = 100


@dataclass(frozen=True)
class VectorizedFixedPointConfig:
    """Configuration for vectorised fixed-point numbers stored in base 2**limb_bits."""

    total_limbs: int
    frac_limbs: int = 0
    limb_bits: int = 8

    def __post_init__(self) -> None:
        if self.limb_bits not in (8, 16):
            raise ValueError("Only 8-bit and 16-bit limb widths are supported for now.")
        if self.total_limbs <= 0:
            raise ValueError("total_limbs must be positive.")
        if self.frac_limbs < 0:
            raise ValueError("frac_limbs must be non-negative.")
        if self.frac_limbs > self.total_limbs:
            raise ValueError("frac_limbs cannot exceed total_limbs.")

    @property
    def base(self) -> int:
        return 1 << self.limb_bits

    @property
    def dtype(self) -> np.dtype:
        return np.uint8 if self.limb_bits == 8 else np.uint16

    @property
    def scale(self) -> int:
        return int(self.base ** self.frac_limbs)

    def widen(self, extra_limbs: int) -> "VectorizedFixedPointConfig":
        if extra_limbs < 0:
            raise ValueError("extra_limbs must be non-negative.")
        return VectorizedFixedPointConfig(
            total_limbs=self.total_limbs + extra_limbs,
            frac_limbs=self.frac_limbs,
            limb_bits=self.limb_bits,
        )


def _coerce_decimal(value: NumberLike) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, np.integer)):
        return Decimal(int(value))
    if isinstance(value, float):
        return Decimal(str(value))
    raise TypeError(f"Unsupported type {type(value)!r} for conversion to Decimal.")


def _int_to_limbs(value: int, base: int) -> Sequence[int]:
    if value < 0:
        raise ValueError("Only non-negative values are supported at the moment.")
    if value == 0:
        return [0]
    digits: list[int] = []
    while value:
        value, rem = divmod(value, base)
        digits.append(int(rem))
    return digits


def _limbs_to_int(limbs: Sequence[int], base: int) -> int:
    total = 0
    for digit in reversed(limbs):
        total = total * base + int(digit)
    return total


def _normalize_limbs(
    raw_limbs: Sequence[int],
    *,
    base: int,
    max_len: int,
    dtype: np.dtype,
    drop_lsb: int = 0,
    truncate_msb: bool = False,
) -> Tuple[np.ndarray, bool]:
    raw = np.asarray(raw_limbs, dtype=np.int64)
    if raw.ndim != 1:
        raise ValueError("raw_limbs must be a 1-D sequence.")

    carry = 0
    digits: list[int] = []
    for val in raw:
        total = int(val) + carry
        digits.append(total % base)
        carry = total // base

    while carry:
        digits.append(carry % base)
        carry //= base

    dropped_nonzero = False
    if drop_lsb:
        remainder = 0
        for idx in range(drop_lsb - 1, -1, -1):
            remainder = remainder * base + digits[idx]
        dropped_nonzero = remainder != 0

        digits = digits[drop_lsb:]

        if remainder:
            half = base**drop_lsb // 2
            round_up = remainder > half
            if remainder == half:
                lsb = digits[0] if digits else 0
                round_up = bool(lsb & 1)

            if round_up:
                carry = 1
                for i in range(len(digits)):
                    total = digits[i] + carry
                    digits[i] = total % base
                    carry = total // base
                    if carry == 0:
                        break
                if carry:
                    digits.append(carry)

    if not digits:
        digits = [0]

    if len(digits) > max_len:
        if not truncate_msb:
            raise OverflowError(
                f"value requires {len(digits)} limbs but configuration allows only {max_len}"
            )
        overflow = digits[max_len:]
        digits = digits[:max_len]
        dropped_nonzero = dropped_nonzero or any(overflow)

    if len(digits) < max_len:
        digits.extend([0] * (max_len - len(digits)))

    return np.asarray(digits, dtype=dtype), dropped_nonzero


def _trim_trailing_zeros(limbs: np.ndarray) -> np.ndarray:
    if limbs.ndim != 1:
        raise ValueError("limb vector must be 1-D.")
    non_zero_indices = np.nonzero(limbs)[0]
    if non_zero_indices.size == 0:
        return limbs[:1].copy()  # Preserve a single zero digit.
    return limbs[: non_zero_indices[-1] + 1].copy()


class VectorizedFixedPoint:
    """Vectorised fixed-point numbers backed by uint8/uint16 limb arrays."""

    __slots__ = ("config", "limbs", "_sticky")

    def __init__(
        self,
        limbs: Sequence[int] | np.ndarray,
        config: VectorizedFixedPointConfig,
        *,
        normalized: bool = False,
    ) -> None:
        object.__setattr__(self, "config", config)
        if normalized:
            arr = np.asarray(limbs, dtype=config.dtype)
            if arr.ndim != 1:
                raise ValueError("Limbs must be 1-D.")
            if len(arr) != config.total_limbs:
                raise ValueError("Normalized limbs must match config.total_limbs.")
            object.__setattr__(self, "limbs", arr.copy())
            object.__setattr__(self, "_sticky", False)
            return

        arr, dropped = _normalize_limbs(
            limbs, base=config.base, max_len=config.total_limbs, dtype=config.dtype
        )
        object.__setattr__(self, "limbs", arr)
        object.__setattr__(self, "_sticky", dropped)

    @classmethod
    def zero(cls, config: VectorizedFixedPointConfig) -> "VectorizedFixedPoint":
        return cls(np.zeros(config.total_limbs, dtype=config.dtype), config, normalized=True)

    @classmethod
    def from_int(cls, value: int, config: VectorizedFixedPointConfig) -> "VectorizedFixedPoint":
        if value < 0:
            raise ValueError("Negative values are not supported yet.")
        scaled = int(value) * config.scale
        limbs = _int_to_limbs(scaled, config.base)
        return cls(limbs, config)

    @classmethod
    def from_number(
        cls,
        value: NumberLike,
        config: VectorizedFixedPointConfig,
    ) -> "VectorizedFixedPoint":
        if isinstance(value, (int, np.integer)):
            return cls.from_int(int(value), config)

        dec = _coerce_decimal(value)
        scaled = (dec * Decimal(config.scale)).to_integral_value(rounding=ROUND_HALF_EVEN)
        if scaled < 0:
            raise ValueError("Negative values are not supported yet.")
        limbs = _int_to_limbs(int(scaled), config.base)
        return cls(limbs, config)

    def copy(self) -> "VectorizedFixedPoint":
        return VectorizedFixedPoint(self.limbs.copy(), self.config, normalized=True)

    def to_scaled_int(self) -> int:
        return _limbs_to_int(self.limbs, self.config.base)

    def to_decimal(self) -> Decimal:
        return Decimal(self.to_scaled_int()) / Decimal(self.config.scale)

    def to_float(self) -> float:
        return float(self.to_decimal())

    def to_numpy(self) -> np.ndarray:
        return self.limbs.copy()

    def lost_precision(self) -> bool:
        return bool(self._sticky)

    def _coerce_other(self, other: NumberLike | "VectorizedFixedPoint") -> "VectorizedFixedPoint":
        if isinstance(other, VectorizedFixedPoint):
            if other.config != self.config:
                raise ValueError("Operands must share the same configuration.")
            return other
        return VectorizedFixedPoint.from_number(other, self.config)

    def __add__(self, other: NumberLike | "VectorizedFixedPoint") -> "VectorizedFixedPoint":
        rhs = self._coerce_other(other)
        width = self.config.total_limbs + 1  # allow an extra limb for overflow detection
        raw = np.zeros(width, dtype=np.int64)
        raw[: self.config.total_limbs] += self.limbs.astype(np.int64)
        raw[: rhs.config.total_limbs] += rhs.limbs.astype(np.int64)

        limbs, dropped = _normalize_limbs(
            raw,
            base=self.config.base,
            max_len=self.config.total_limbs,
            dtype=self.config.dtype,
            truncate_msb=True,
        )
        result = VectorizedFixedPoint(limbs, self.config, normalized=True)
        object.__setattr__(result, "_sticky", dropped)
        return result

    def __radd__(self, other: NumberLike | "VectorizedFixedPoint") -> "VectorizedFixedPoint":
        return self.__add__(other)

    def __mul__(self, other: NumberLike | "VectorizedFixedPoint") -> "VectorizedFixedPoint":
        rhs = self._coerce_other(other)

        a = _trim_trailing_zeros(self.limbs.astype(np.int64))
        b = _trim_trailing_zeros(rhs.limbs.astype(np.int64))
        raw = np.convolve(a, b)

        limbs, dropped = _normalize_limbs(
            raw,
            base=self.config.base,
            max_len=self.config.total_limbs,
            dtype=self.config.dtype,
            drop_lsb=self.config.frac_limbs,
            truncate_msb=True,
        )
        result = VectorizedFixedPoint(limbs, self.config, normalized=True)
        object.__setattr__(result, "_sticky", dropped)
        return result

    def __rmul__(self, other: NumberLike | "VectorizedFixedPoint") -> "VectorizedFixedPoint":
        return self.__mul__(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorizedFixedPoint):
            return NotImplemented
        return self.config == other.config and np.array_equal(self.limbs, other.limbs)

    def __repr__(self) -> str:
        return (
            f"VectorizedFixedPoint(value={self.to_decimal()}, "
            f"limbs={self.limbs.tolist()}, config={self.config}, "
            f"sticky={self._sticky})"
        )


def vector_add_with_carry(
    a: np.ndarray,
    b: np.ndarray,
    *,
    base: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add two limb matrices in parallel and return (sum, carry_out).

    This is a CPU reference for the parallel-prefix (Kogge-Stone style) adder.
    Each row stores a single number in little-endian order.
    """
    if a.shape != b.shape:
        raise ValueError("Shapes must match for element-wise addition.")
    if a.ndim != 2:
        raise ValueError("Inputs must be 2-D matrices (batch, limbs).")

    a_u = a.astype(np.int64, copy=False)
    b_u = b.astype(np.int64, copy=False)
    width = a.shape[1]

    partial = a_u + b_u
    result = np.empty_like(partial)
    carry = np.zeros((a.shape[0],), dtype=np.int64)

    for idx in range(width):
        total = partial[:, idx] + carry
        result[:, idx] = total % base
        carry = total // base

    return result.astype(a.dtype, copy=False), carry


def convolution_multiply(
    a: np.ndarray,
    b: np.ndarray,
    *,
    base: int = 256,
    drop_lsb: int = 0,
) -> Tuple[np.ndarray, bool]:
    """Multiply two limb vectors using discrete convolution and renormalise.

    Returns the limbs (little endian) together with a flag that signals if
    precision was lost while dropping the least-significant digits.
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Inputs must be 1-D limb vectors.")
    a_trim = _trim_trailing_zeros(a.astype(np.int64))
    b_trim = _trim_trailing_zeros(b.astype(np.int64))
    raw = np.convolve(a_trim, b_trim)
    limbs, dropped = _normalize_limbs(
        raw,
        base=base,
        max_len=max(len(a), len(b)),
        dtype=a.dtype,
        drop_lsb=drop_lsb,
        truncate_msb=True,
    )
    return limbs, dropped


if __name__ == "__main__":
    cfg = VectorizedFixedPointConfig(total_limbs=7, frac_limbs=5, limb_bits=8)
    lhs = VectorizedFixedPoint.from_number(10.515151515351435, cfg)
    rhs = VectorizedFixedPoint.from_number(10.535464135135143513, cfg)

    summed = lhs + rhs
    product = lhs * rhs

    print("lhs limbs:", lhs.limbs)
    print("rhs limbs:", rhs.limbs)
    print("lhs + rhs:", summed.to_decimal(), summed.limbs, "sticky:", summed.lost_precision())
    print("lhs * rhs:", product.to_decimal(), product.limbs, "sticky:", product.lost_precision())
