"""Tests for the glass-property value functions used with PropertyColorizer."""

from __future__ import annotations

import warnings
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from optiland.materials import AbbeMaterial, AbbeMaterialE, IdealMaterial, get_nd_vd
from optiland.optic import Optic
from optiland.samples.simple import CementedAchromat
from optiland.samples.telescopes import HubbleTelescope
from optiland.visualization.coloring import (
    ColorScale,
    ComponentRef,
    PropertyColorizer,
    abbe_number,
    refractive_index,
)
from optiland.visualization.system import OpticViewer3D

matplotlib.use("Agg")

COOLWARM_R = matplotlib.colormaps["coolwarm_r"]

# CementedAchromat: N-BAK1 (crown) between surfaces 1-2, SF2 (flint) between 2-3.
CROWN = ComponentRef("lens_element", (1, 2))
FLINT = ComponentRef("lens_element", (2, 3))

# d, F and C lines (um) that define V_d.
VD_LINES = (0.5875618, 0.4861327, 0.6562725)


def _singlet(material) -> Optic:
    optic = Optic()
    optic.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
    optic.surfaces.add(index=1, radius=20.0, thickness=3.0, material=material)
    optic.surfaces.add(index=2, radius=-20.0, thickness=20.0, is_stop=True)
    optic.surfaces.add(index=3)
    optic.set_aperture(aperture_type="EPD", value=5.0)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0)
    optic.wavelengths.add(value=0.55, is_primary=True)
    return optic


class TestAbbeNumber:
    def test_doublet_matches_catalog_vd(self, set_test_backend):
        optic = CementedAchromat()
        for ref, glass in [(CROWN, "N-BAK1"), (FLINT, "SF2")]:
            _, vd = get_nd_vd(glass)
            value = abbe_number(optic, ref)
            assert isinstance(value, float)
            assert value == pytest.approx(vd, rel=1e-4)

    def test_crown_has_higher_abbe_and_colors_follow(self, set_test_backend):
        optic = CementedAchromat()
        assert abbe_number(optic, CROWN) > abbe_number(optic, FLINT)

        colorizer = PropertyColorizer(abbe_number, ColorScale(cmap="coolwarm_r"))
        colorizer.prepare(optic, [CROWN, FLINT])
        # Limits come from the two values, so the crown sits at the top of the
        # scale and the flint at the bottom.
        np.testing.assert_allclose(colorizer.color_for(CROWN), COOLWARM_R(1.0))
        np.testing.assert_allclose(colorizer.color_for(FLINT), COOLWARM_R(0.0))
        assert colorizer.scale.vmin == pytest.approx(abbe_number(optic, FLINT))
        assert colorizer.scale.vmax == pytest.approx(abbe_number(optic, CROWN))

    def test_dispersionless_material_returns_none_without_warning(
        self, set_test_backend
    ):
        optic = _singlet(IdealMaterial(n=1.5))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert abbe_number(optic, ComponentRef("lens_element", (1, 2))) is None

    @pytest.mark.parametrize(
        "material",
        [
            AbbeMaterial(n=1.5168, abbe=64.17, model="buchdahl"),
            AbbeMaterialE(n=1.51872, abbe=63.96),
        ],
        ids=["AbbeMaterial", "AbbeMaterialE"],
    )
    def test_abbe_materials_report_vd_from_indices(self, set_test_backend, material):
        optic = _singlet(material)
        nd, nf, nc = (float(np.squeeze(material.n(w))) for w in VD_LINES)
        value = abbe_number(optic, ComponentRef("lens_element", (1, 2)))
        assert value == pytest.approx((nd - 1) / (nf - nc))


class TestRefractiveIndex:
    def test_returns_nd_for_each_element(self, set_test_backend):
        optic = CementedAchromat()
        value_fn = refractive_index(0.5876)
        for ref, glass in [(CROWN, "N-BAK1"), (FLINT, "SF2")]:
            nd, _ = get_nd_vd(glass)
            value = value_fn(optic, ref)
            assert isinstance(value, float)
            assert value == pytest.approx(nd, rel=1e-5)

    def test_default_uses_primary_wavelength(self, set_test_backend):
        optic = CementedAchromat()  # primary is the d line (0.5876 um)
        value_fn = refractive_index()
        primary = value_fn(optic, FLINT)
        assert primary == refractive_index(optic.primary_wavelength)(optic, FLINT)

        # The wavelength is read at call time: switching the primary to the
        # F line (0.4861 um) gives the flint's higher index there.
        optic.wavelengths.primary_index = 0
        assert value_fn(optic, FLINT) > primary

    def test_ideal_material_has_an_index(self, set_test_backend):
        optic = _singlet(IdealMaterial(n=1.5))
        value = refractive_index()(optic, ComponentRef("lens_element", (1, 2)))
        assert value == pytest.approx(1.5)


class TestNonGlassComponents:
    @pytest.mark.parametrize(
        "value_fn", [abbe_number, refractive_index()], ids=["abbe", "index"]
    )
    def test_mirrors_return_none(self, set_test_backend, value_fn):
        optic = HubbleTelescope()
        mirrors = [
            k for k, s in enumerate(optic.surfaces) if s.interaction_model.is_reflective
        ]
        assert mirrors
        for k in mirrors:
            assert value_fn(optic, ComponentRef("mirror", (k,))) is None

    @pytest.mark.parametrize(
        "value_fn", [abbe_number, refractive_index()], ids=["abbe", "index"]
    )
    def test_air_returns_none(self, set_test_backend, value_fn):
        optic = CementedAchromat()
        # Surface 3 is followed by air: neither a surface ref nor an
        # air-gap "element" gets a value.
        assert value_fn(optic, ComponentRef("surface", (3,))) is None
        assert value_fn(optic, ComponentRef("lens_element", (3, 4))) is None


class TestDraw:
    @pytest.mark.parametrize(
        "value_fn", [abbe_number, refractive_index(0.5876)], ids=["abbe", "index"]
    )
    def test_draw_colors_both_elements_with_colorbar(self, set_test_backend, value_fn):
        optic = CementedAchromat()
        colorizer = PropertyColorizer(value_fn, ColorScale(label="glass"))
        fig, ax = optic.draw(show=False, component_colorizer=colorizer)
        faces = [p.get_facecolor() for p in ax.patches]
        assert len(faces) == 2
        assert faces[0] != faces[1]
        assert len(fig.axes) == 2  # lens axes + colorbar
        plt.close(fig)

    def test_2d_doublet_colors_follow_abbe(self, set_test_backend):
        colorizer = PropertyColorizer(abbe_number, ColorScale(cmap="coolwarm_r"))
        _, ax = CementedAchromat().draw(show=False, component_colorizer=colorizer)
        faces = [p.get_facecolor() for p in ax.patches]
        np.testing.assert_allclose(faces[0][:3], COOLWARM_R(1.0)[:3])  # crown
        np.testing.assert_allclose(faces[1][:3], COOLWARM_R(0.0)[:3])  # flint
        plt.close("all")

    def test_3d_doublet_colors_follow_abbe(self, set_test_backend):
        colorizer = PropertyColorizer(abbe_number, ColorScale(cmap="coolwarm_r"))
        viewer = OpticViewer3D(CementedAchromat())
        with (
            patch.object(viewer.iren, "Start"),
            patch.object(viewer.ren_win, "Render"),
        ):
            viewer.view(num_rays=6, component_colorizer=colorizer)
        actors = viewer.ren_win.GetRenderers().GetFirstRenderer().GetActors()
        actors.InitTraversal()
        glass = {
            tuple(round(c, 4) for c in actor.GetProperty().GetColor())
            for actor in (
                actors.GetNextItemAsObject() for _ in range(actors.GetNumberOfItems())
            )
            if actor.GetProperty().GetOpacity() == 0.5
        }
        expected = {tuple(round(c, 4) for c in COOLWARM_R(f)[:3]) for f in (0.0, 1.0)}
        assert expected <= glass
