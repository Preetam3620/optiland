"""Component Property Module

This module contains value functions for :class:`PropertyColorizer` that color
lens elements by a property of their glass, such as the Abbe number or the
refractive index.

Example:
    >>> from optiland.visualization.coloring import (
    ...     ColorScale,
    ...     PropertyColorizer,
    ...     abbe_number,
    ... )
    >>> optic.draw(
    ...     component_colorizer=PropertyColorizer(
    ...         abbe_number, ColorScale(cmap="coolwarm_r", label="Abbe number")
    ...     )
    ... )

Preetam Maske, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.materials.base import BaseMaterial

if TYPE_CHECKING:
    from collections.abc import Callable

    from optiland.optic import Optic
    from optiland.visualization.coloring.components import ComponentRef


def abbe_number(optic: Optic, ref: ComponentRef) -> float | None:
    """Abbe number of a lens element's glass.

    Always V_d, computed from the material's indices at the d, F and C lines
    by :meth:`BaseMaterial.abbe`, so every material is colored on the same
    scale. For ``AbbeMaterial`` this can differ slightly from the Abbe number
    it was given, since the indices come from a fitted model, and for
    ``AbbeMaterialE`` it is V_d rather than the V_e it was given.

    Args:
        optic (Optic): The system being drawn.
        ref (ComponentRef): The component to evaluate.

    Returns:
        float | None: The Abbe number, or None for components that are not lens
        elements and for materials without dispersion (e.g. ``IdealMaterial``),
        so they keep the default color.

    Example:
        >>> optic.draw(
        ...     component_colorizer=PropertyColorizer(
        ...         abbe_number, ColorScale(cmap="coolwarm_r", label="Abbe number")
        ...     )
        ... )
    """
    material = _element_material(optic, ref)
    if material is None:
        return None
    # Call the base method directly: AbbeMaterial and AbbeMaterialE shadow
    # ``abbe`` with the stored input value. A dispersionless material gives
    # n_F - n_C = 0; return None rather than warn about the division by zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        abbe = BaseMaterial.abbe(material)
    return _finite_or_none(abbe)


def refractive_index(
    wavelength: float | None = None,
) -> Callable[[Optic, ComponentRef], float | None]:
    """Build a value function for the refractive index of a lens element's glass.

    Args:
        wavelength (float | None): Wavelength in microns. None means the
            optic's primary wavelength at draw time.

    Returns:
        Callable[[Optic, ComponentRef], float | None]: Value function for
        :class:`PropertyColorizer`. It returns None for components that are not
        lens elements, so they keep the default color.

    Example:
        >>> optic.draw(
        ...     component_colorizer=PropertyColorizer(
        ...         refractive_index(0.5876), ColorScale(label="n_d")
        ...     )
        ... )
    """

    def value_fn(optic: Optic, ref: ComponentRef) -> float | None:
        material = _element_material(optic, ref)
        if material is None:
            return None
        wl = optic.primary_wavelength if wavelength is None else wavelength
        return _finite_or_none(material.n(wl))

    return value_fn


def _element_material(optic: Optic, ref: ComponentRef) -> BaseMaterial | None:
    """Return the glass of a lens element, or None if ``ref`` is not one.

    The glass is the medium after the element's first surface. A medium with
    n <= 1 at the primary wavelength is air, matching how the system viewer
    decides which surfaces bound a lens.
    """
    if ref.kind != "lens_element":
        return None
    material = optic.surfaces[ref.surface_indices[0]].material_post
    if _to_float(material.n(optic.primary_wavelength)) <= 1:
        return None
    return material


def _finite_or_none(value) -> float | None:
    """Convert a backend scalar to a Python float, mapping NaN and inf to None."""
    value = _to_float(value)
    return value if np.isfinite(value) else None


def _to_float(value) -> float:
    """Convert a backend scalar or one-element array to a detached Python float."""
    return np.asarray(be.to_numpy(value), dtype=float).item()
