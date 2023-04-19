from __future__ import annotations

import jax
import jax_dataclasses as jdc
from jax import numpy as jnp
from overrides import overrides
from typing_extensions import Annotated

from . import _base, hints
from ._so2 import SO2
from .utils import get_epsilon, register_lie_group

@register_lie_group(
    matrix_dim=3,
    parameters_dim=4,
    tangent_dim=3,
    space_dim=2,
)
@jdc.pytree_dataclass
class R2SO2(jdc.EnforcedAnnotationsMixin, _base.RnSOnBase[SO2]):
    """"""

    unit_complex_xy: Annotated[
        jnp.ndarray,
        (..., 4),  # Shape.
        jnp.floating,  # Data-type.
    ]
    """Internal parameters. wxyz quaternion followed by xyz translation."""

    @overrides
    def __repr__(self) -> str:
        unit_complex = jnp.round(self.unit_complex_xy[..., :2], 5)
        xy = jnp.round(self.unit_complex_xy[..., 2:], 5)
        return f"{self.__class__.__name__}(unit_complex={unit_complex}, xy={xy})"

    @staticmethod
    def from_xy_theta(x: hints.Scalar, y: hints.Scalar, theta: hints.Scalar) -> R2SO2:
        """Construct a transformation from standard 2D pose parameters.

        Note that this is not the same as integrating over a length-3 twist.
        """
        cos = jnp.cos(theta)
        sin = jnp.sin(theta)
        return R2SO2(unit_complex_xy=jnp.array([cos, sin, x, y]))


    @staticmethod
    @overrides
    def from_rotation_and_translation(
        rotation: SO2,
        translation: hints.Array,
    ) -> R2SO2:
        assert translation.shape == (2,)
        return R2SO2(
            unit_complex_xy=jnp.concatenate([rotation.unit_complex, translation])
        )

    @overrides
    def rotation(self) -> SO2:
        return SO2(unit_complex=self.unit_complex_xy[..., :2])

    @overrides
    def translation(self) -> jnp.ndarray:
        return self.unit_complex_xy[..., 2:]

    @staticmethod
    @overrides
    def identity() -> R2SO2:
        return R2SO2(unit_complex_xy=jnp.array([1.0, 0.0, 0.0, 0.0]))

    @staticmethod
    @overrides
    def from_matrix(matrix: hints.Array) -> R2SO2:
        assert matrix.shape == (3, 3)
        # Currently assumes bottom row is [0, 0, 0, 1].
        return R2SO2.from_rotation_and_translation(
            rotation=SO2.from_matrix(matrix[:2, :2]),
            translation=matrix[:2, 2],
        )

    @overrides
    def as_matrix(self) -> jnp.ndarray:
        cos, sin, x, y = self.unit_complex_xy
        return jnp.array(
            [
                [cos, -sin, x],
                [sin, cos, y],
                [0.0, 0.0, 1.0],
            ]
        )

    @overrides
    def parameters(self) -> jnp.ndarray:
        return self.unit_complex_xy

    
    @staticmethod
    @overrides
    def exp(tangent: hints.Array) -> R2SO2:

        assert tangent.shape == (3,)

        theta = tangent[2]

        return R2SO2.from_rotation_and_translation(
            rotation=SO2.from_radians(theta),
            translation=tangent[:2],
        )

    
    @overrides
    def log(self) -> jnp.ndarray:

        theta = self.rotation().log()[0]

        tangent = jnp.concatenate([self.translation(), theta[None]])
        return tangent

    @overrides
    def adjoint(self) -> jnp.ndarray:
        raise NotImplementedError

    @staticmethod
    @overrides
    def sample_uniform(key: hints.KeyArray) -> R2SO2:
        key0, key1 = jax.random.split(key)
        return R2SO2.from_rotation_and_translation(
            rotation=SO2.sample_uniform(key0),
            translation=jax.random.uniform(
                key=key1, shape=(2,), minval=-1.0, maxval=1.0
            ),
        )
