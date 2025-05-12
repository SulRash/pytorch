# mypy: allow-untyped-defs
from typing import Optional

import torch


class SobolEngine:
    r"""
    The :class:`torch.quasirandom.SobolEngine` is an engine for generating
    (scrambled) Sobol sequences. Sobol sequences are an example of low
    discrepancy quasi-random sequences.

    This implementation of an engine for Sobol sequences is capable of
    sampling sequences up to a maximum dimension of 21201. It uses direction
    numbers from https://web.maths.unsw.edu.au/~fkuo/sobol/ obtained using the
    search criterion D(6) up to the dimension 21201. This is the recommended
    choice by the authors.

    References:
      - Art B. Owen. Scrambling Sobol and Niederreiter-Xing points.
        Journal of Complexity, 14(4):466-489, December 1998.

      - I. M. Sobol. The distribution of points in a cube and the accurate
        evaluation of integrals.
        Zh. Vychisl. Mat. i Mat. Phys., 7:784-802, 1967.

    Args:
        dimension (Int): The dimensionality of the sequence to be drawn
        scramble (bool, optional): Setting this to ``True`` will produce
                                   scrambled Sobol sequences. Scrambling is
                                   capable of producing better Sobol
                                   sequences. Default: ``False``.
        seed (Int, optional): This is the seed for the scrambling. The seed
                              of the random number generator is set to this,
                              if specified. Otherwise, it uses a random seed.
                              Default: ``None``
        device (torch.device or str, optional): The device on which all
                              tensors and internal state are allocated
                              (e.g., "cuda:0"). Defaults to CPU.
    """

    MAXBIT = 30
    MAXDIM = 21201

    def __init__(
        self,
        dimension: int,
        scramble: bool = False,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        if dimension > self.MAXDIM or dimension < 1:
            raise ValueError(
                "Supported range of dimensionality "
                f"for SobolEngine is [1, {self.MAXDIM}]"
            )

        # Device handling
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"SobolEngine requested on CUDA device {self.device}, "
                "but CUDA is not available in this PyTorch build."
            )

        self.seed = seed
        self.scramble = scramble
        self.dimension = dimension

        dev = self.device

        # Initialize Sobol state on the target device
        self.sobolstate = torch.zeros(
            dimension, self.MAXBIT, device=dev, dtype=torch.long
        )
        torch._sobol_engine_initialize_state_(self.sobolstate, self.dimension)

        # Initial shift vector (zero or scrambled)
        if not self.scramble:
            self.shift = torch.zeros(dimension, device=dev, dtype=torch.long)
        else:
            self._scramble()

        # Copy into quasi state
        self.quasi = self.shift.clone(memory_format=torch.contiguous_format)
        self._first_point = (self.quasi / 2**self.MAXBIT).reshape(1, -1)
        self.num_generated = 0

    def draw(
        self,
        n: int = 1,
        out: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        r"""
        Function to draw a sequence of :attr:`n` points from a Sobol sequence.
        Note that the samples are dependent on the previous samples. The size
        of the result is :math:`(n, dimension)`.

        Args:
            n (Int, optional): The length of sequence of points to draw.
                               Default: 1
            out (Tensor, optional): The output tensor (must be on same device).
            dtype (:class:`torch.dtype`, optional): the desired data type of the
                                                    returned tensor.
                                                    Default: ``None``
        """
        if dtype is None:
            dtype = torch.get_default_dtype()

        # Produce result via C++/CUDA-dispatched kernel
        if self.num_generated == 0:
            if n == 1:
                result = self._first_point.to(dtype)
            else:
                result, self.quasi = torch._sobol_engine_draw(
                    self.quasi,
                    n - 1,
                    self.sobolstate,
                    self.dimension,
                    self.num_generated,
                    dtype=dtype,
                )
                result = torch.cat((self._first_point.to(dtype), result), dim=-2)
        else:
            result, self.quasi = torch._sobol_engine_draw(
                self.quasi,
                n,
                self.sobolstate,
                self.dimension,
                self.num_generated - 1,
                dtype=dtype,
            )

        self.num_generated += n

        if out is not None:
            if out.device != self.device:
                raise ValueError(
                    f"Output tensor must be on device {self.device}, "
                    f"but is on {out.device}"
                )
            out.resize_as_(result).copy_(result)
            return out

        return result

    def draw_base2(
        self,
        m: int,
        out: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        r"""
        Function to draw a sequence of :attr:`2**m` points from a Sobol sequence.
        Note that the samples are dependent on the previous samples. The size
        of the result is :math:`(2**m, dimension)`.

        Args:
            m (Int): The (base2) exponent of the number of points to draw.
            out (Tensor, optional): The output tensor
            dtype (:class:`torch.dtype`, optional): the desired data type of the
                                                    returned tensor.
                                                    Default: ``None``
        """
        n = 2**m
        total_n = self.num_generated + n
        if not (total_n & (total_n - 1) == 0):
            raise ValueError(
                "The balance properties of Sobol' points require "
                f"n to be a power of 2. {self.num_generated} points have been "
                f"previously generated, then: n={self.num_generated}+2**{m}={total_n}. "
                "If you still want to do this, please use "
                "'SobolEngine.draw()' instead."
            )
        return self.draw(n=n, out=out, dtype=dtype)

    def reset(self):
        r"""
        Function to reset the ``SobolEngine`` to base state.
        """
        self.quasi.copy_(self.shift)
        self.num_generated = 0
        return self

    def fast_forward(self, n: int):
        r"""
        Function to fast-forward the state of the ``SobolEngine`` by
        :attr:`n` steps. This is equivalent to drawing :attr:`n` samples
        without using the samples.

        Args:
            n (Int): The number of steps to fast-forward by.
        """
        if self.num_generated == 0:
            torch._sobol_engine_ff_(
                self.quasi, n - 1, self.sobolstate, self.dimension, self.num_generated
            )
        else:
            torch._sobol_engine_ff_(
                self.quasi, n, self.sobolstate, self.dimension, self.num_generated - 1
            )
        self.num_generated += n
        return self

    def _scramble(self):
        r"""
        Internal helper to generate a random shift and
        lower-triangular scramble matrices.
        """
        # Set up generator on correct device
        g: Optional[torch.Generator] = None
        if self.seed is not None:
            g = torch.Generator(device=self.device)
            g.manual_seed(self.seed)

        dev = self.device

        # Generate shift vector
        shift_ints = torch.randint(
            2, (self.dimension, self.MAXBIT), device=dev, generator=g
        )
        self.shift = torch.mv(
            shift_ints, torch.pow(torch.arange(0, self.MAXBIT, device=dev), 2)
        )

        # Generate lower triangular matrices (stacked across dimensions)
        ltm_dims = (self.dimension, self.MAXBIT, self.MAXBIT)
        ltm = torch.randint(2, ltm_dims, device=dev, generator=g).tril()

        torch._sobol_engine_scramble_(self.sobolstate, ltm, self.dimension)

    def __repr__(self):
        fmt = [f"dimension={self.dimension}"]
        if self.scramble:
            fmt.append("scramble=True")
        if self.seed is not None:
            fmt.append(f"seed={self.seed}")
        fmt.append(f"device={self.device}")
        return f"{self.__class__.__name__}(" + ", ".join(fmt) + ")"
