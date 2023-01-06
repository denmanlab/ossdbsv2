
from typing import List
from ossdbs.brain_imaging.mri import MagneticResonanceImage
from ossdbs.brainsubstance import Material
from ossdbs.voxels import Voxels
import ngsolve
import numpy as np


class Mesh:
    """Class for interacting with the mesh for FEM.

    Parameters
    ----------
    geometry : netgen.libngpy._NgOCC.OCCGeometry

    order : int
        Order of mesh elements.
    """
    def __init__(self,
                 geometry,
                 order: int) -> None:
        self.__mesh = ngsolve.Mesh(ngmesh=geometry.GenerateMesh())
        self.__mesh.Curve(order=order)
        self.__order = order
        self.__complex = False

    def get_boundaries(self) -> List:
        """Return all boundary names.

        Returns
        -------
        list
            Collection of strings.
        """
        return list(set(self.__mesh.GetBoundaries()) - set(['default']))

    def boundary_coefficients(self, boundaries) \
            -> ngsolve.fem.CoefficientFunction:
        """Return a boundary coefficient function.

        Returns
        -------
        ngsolve.fem.CoefficientFunction
        """
        return self.__mesh.BoundaryCF(values=boundaries)

    def flux_space(self) -> ngsolve.comp.HDiv:
        """Return a flux space based on the mesh.

        Returns
        -------
        ngsolve.comp.HDiv
        """
        return ngsolve.HDiv(mesh=self.__mesh,
                            order=self.__order-1,
                            complex=self.__complex)

    def ngsolvemesh(self) -> ngsolve.comp.Mesh:
        """Return mesh as a ngsolve object.

        Returns
        -------
        ngsolve.comp.Mesh
        """
        return self.__mesh

    def refine(self) -> None:
        """Refine the mesh."""
        self.__mesh.Refine()
        self.__mesh.Curve(order=self.__order)

    def set_complex(self, state: bool) -> None:
        """Set the data type to complex.

        Parameters
        ----------
        state : bool
            True for complex data type, False otherwise.
        """
        self.__complex = state

    def is_complex(self) -> bool:
        """Check complex data type.

        Returns
        -------
        bool
            True if complex, False otherwise.
        """
        return self.__complex

    def sobolev_space(self) -> ngsolve.comp.H1:
        """Return a sobolev space based on the mesh.

        Returns
        -------
        ngsolve.comp.H1
        """
        dirichlet = '|'.join(boundary for boundary in self.get_boundaries())
        return ngsolve.H1(mesh=self.__mesh,
                          order=self.__order,
                          dirichlet=dirichlet,
                          complex=self.__complex,
                          wb_withedges=False)

    def refine_by_mri(self, mri: MagneticResonanceImage) -> None:
        """Refine the mesh by magnetic resonance imaging.

        Parameters
        ----------
        mri : MagneticResonanceImage
            Image which represents the distributiion of brain substances.
        """

        maximum_size = min(mri.voxel_size())
        csf_voxel = mri.material_distribution(Material.CSF)
        flags = np.logical_and(self.__elements_at_position(csf_voxel),
                               self.__element_sizes() > maximum_size)
        while np.any(flags) and self.sobolev_space().ndof < 1e5:
            self.__set_volume_refinement_flags(flags)
            self.refine()
            csf_voxel = mri.material_distribution(Material.CSF)
            flags = np.logical_and(self.__elements_at_position(csf_voxel),
                                   self.__element_sizes() > maximum_size)

    def refine_by_boundaries(self, boundaries: list) -> None:
        """Refine the mesh by the boundaries.

        Parameters
        ----------
        boundaries : list of str
            Collection of boundary names.
        """

        elements = self.__mesh.Elements(ngsolve.BND)
        flags = [element.mat in boundaries for element in elements]
        for element, flag in zip(self.__mesh.Elements(ngsolve.BND), flags):
            self.__mesh.SetRefinementFlag(ei=element, refine=flag)
        self.refine()

    def refine_by_error(self, error: ngsolve.fem.CoefficientFunction) -> List:
        """Refine the mesh by the error at each mesh element.

        Parameters
        ----------
        error : ngsolve.fem.CoefficientFunction
            Function holding all errors for each mesh element.
        """

        errors = ngsolve.Integrate(cf=error,
                                   mesh=self.__mesh,
                                   VOL_or_BND=ngsolve.VOL,
                                   element_wise=True).real
        limit = 0.5 * max(errors)
        flags = [errors[el.nr] > limit for el in self.__mesh.Elements()]
        for element, flag in zip(self.__mesh.Elements(ngsolve.BND), flags):
            self.__mesh.SetRefinementFlag(ei=element, refine=flag)
        self.refine()

    def __set_volume_refinement_flags(self, flags: List[bool]) -> None:
        for element, flag in zip(self.__mesh.Elements(ngsolve.VOL), flags):
            self.__mesh.SetRefinementFlag(ei=element, refine=flag)

    def __elements_at_position(self, position: Voxels) -> None:
        space = ngsolve.L2(self.__mesh, order=0)
        grid_function = ngsolve.GridFunction(space=space)
        cf = ngsolve.VoxelCoefficient(start=tuple(position.start),
                                      end=tuple(position.end),
                                      values=position.data.astype(float),
                                      linear=False)
        grid_function.Set(cf)
        return grid_function.vec.FV().NumPy()

    def __element_sizes(self) -> List:
        cf = ngsolve.CoefficientFunction(1)
        volumes = ngsolve.Integrate(cf=cf, mesh=self.__mesh, element_wise=True)
        return (6 * volumes.NumPy()) ** (1 / 3)