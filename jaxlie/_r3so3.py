from __future__ import annotations

import jax
import jax_dataclasses as jdc
from jax import numpy as jnp
from overrides import overrides
from typing_extensions import Annotated

from . import _base, hints
from ._so3 import SO3
from .utils import get_epsilon, register_lie_group

@register_lie_group(
    matrix_dim=4,
    parameters_dim=7,
    tangent_dim=6,
    space_dim=3,
)
@jdc.pytree_dataclass
class R3SO3(jdc.EnforcedAnnotationsMixin, _base.RnSOnBase[SO3]):
    """"""

    wxyz_xyz: Annotated[
        jnp.ndarray,
        (..., 7),  # Shape.
        jnp.floating,  # Data-type.
    ]
    """Internal parameters. wxyz quaternion followed by xyz translation."""

    @overrides
    def __repr__(self) -> str:
        quat = jnp.round(self.wxyz_xyz[..., :4], 5)
        trans = jnp.round(self.wxyz_xyz[..., 4:], 5)
        return f"{self.__class__.__name__}(wxyz={quat}, xyz={trans})"

    @staticmethod
    @overrides
    def from_rotation_and_translation(
        rotation: SO3,
        translation: hints.Array,
    ) -> R3SO3:
        assert translation.shape == (3,)
        return R3SO3(wxyz_xyz=jnp.concatenate([rotation.wxyz, translation]))

    @overrides
    def rotation(self) -> SO3:
        return SO3(wxyz=self.wxyz_xyz[..., :4])

    @overrides
    def translation(self) -> jnp.ndarray:
        return self.wxyz_xyz[..., 4:]

    def vec(self) -> jnp.ndarray:
        return self.wxyz_xyz

    @staticmethod
    @overrides
    def identity() -> R3SO3:
        return R3SO3(wxyz_xyz=jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    @staticmethod
    def orthogonalize(matrix: hints.Array):
        if matrix.shape == (9,):
            # 6d rotation + 3d translation
            rot = matrix[:6]
            tran = matrix[6:]
            rot = SO3.orthogonalize(rot)
        else:
            rot = SO3.orthogonalize(matrix[:3, :3])
            tran = matrix[:3, 3]

        return (
            jnp.eye(4)
            .at[:3, :3]
            .set(rot)
            .at[:3, 3]
            .set(tran)
        )

    @staticmethod
    @overrides
    def from_matrix(matrix: hints.Array) -> R3SO3:
        assert matrix.shape == (4, 4)
        # Currently assumes bottom row is [0, 0, 0, 1].
        return R3SO3.from_rotation_and_translation(
            rotation=SO3.from_matrix(matrix[:3, :3]),
            translation=matrix[:3, 3],
        )

    @overrides
    def as_matrix(self) -> jnp.ndarray:
        return (
            jnp.eye(4)
            .at[:3, :3]
            .set(self.rotation().as_matrix())
            .at[:3, 3]
            .set(self.translation())
        )

    @overrides
    def parameters(self) -> jnp.ndarray:
        return self.wxyz_xyz

    
    @staticmethod
    @overrides
    def exp(tangent: hints.Array) -> R3SO3:
        # (x, y, z, omega_x, omega_y, omega_z)
        assert tangent.shape == (6,)

        rotation = SO3.exp(tangent[3:])

        return R3SO3.from_rotation_and_translation(
            rotation=rotation,
            translation=tangent[:3],
        )

    
    @overrides
    def log(self) -> jnp.ndarray:
        omega = self.rotation().log()

        return jnp.concatenate([self.translation(), omega])

    @overrides
    def adjoint(self) -> jnp.ndarray:
        raise NotImplementedError

    @staticmethod
    @overrides
    def sample_uniform(key: hints.KeyArray) -> R3SO3:
        key0, key1 = jax.random.split(key)
        return R3SO3.from_rotation_and_translation(
            rotation=SO3.sample_uniform(key0),
            translation=jax.random.uniform(
                key=key1, shape=(3,), minval=-1.0, maxval=1.0
            ),
        )
