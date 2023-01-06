
from ossdbs.conductivity import Conductivity
from ossdbs.mesh import Mesh
import ngsolve
import numpy as np


class Potential():
    """Represents the electrical potential.

    Parameters
    ----------
    gridfunction : ngsolve.GridFunction
    mesh : Mesh
    """

    def __init__(self, gridfunction: ngsolve.GridFunction, mesh: Mesh) -> None:
        self.gridfunction = gridfunction
        self.mesh = mesh

    def error(self):
        """Evaluate the error of the potential.

        Returns
        -------
        ngsolve.fem.CoefficientFunction
            Error values.
        """
        flux = ngsolve.grad(self.gridfunction)
        space = self.mesh.flux_space()
        flux_potential = ngsolve.GridFunction(space=space)
        flux_potential.Set(coefficient=flux)
        difference = flux - flux_potential
        return difference * ngsolve.Conj(difference)

    def __add__(self, other) -> 'Potential':
        self.gridfunction.vec.data += other._gridfunction.vec.data

    def __mul__(self, value) -> 'Potential':
        self.gridfunction.vec.data *= value


class VolumeConductor():
    """Model for representing a volume conductor which evaluates the potential.

    Parameters
    ----------
    mesh : Mesh

    conductivity : Conductivity
    """

    def __init__(self, mesh: Mesh, conductivity: Conductivity) -> None:
        self.__conductivity = conductivity
        self.mesh = mesh

    def evaluate_potential(self, boundaries: dict, frequency: float) \
            -> ngsolve.comp.GridFunction:
        """Evaluate electrical potential of volume conductor.

        Parameters
        ----------
        boundaries : dict
            Dictionary containing the values for accordingly boundaries.
        frequency : float
            Frequency [Hz] of the input signal.

        Returns
        -------
        tuple
            potential and current density of volume conductor
        """

        conductivities = self.__conductivity.conductivity(frequency)
        values = conductivities.data
        start, end = conductivities.start, conductivities.end
        if not self.mesh.is_complex():
            values = np.real(values)
        sigma = ngsolve.VoxelCoefficient(start, end, values, linear=False)
        space = self.mesh.sobolev_space()
        potential = ngsolve.GridFunction(space=space)
        coefficient = self.mesh.boundary_coefficients(boundaries=boundaries)

        potential.Set(coefficient=coefficient, VOL_or_BND=ngsolve.BND)
        equation = LaplaceEquation(space=space, sigma=sigma)
        potential.vec.data = equation.solve_bvp(input=potential)
        return potential, sigma * ngsolve.grad(potential)


class LaplaceEquation:
    """Solves boundary value problem.

    Parameters
    ----------
    space : ngsolve.comp.H1
    sigma : ngsolve.fem.CoefficientFunction
    """

    def __init__(self,
                 space: ngsolve.comp.H1,
                 sigma: ngsolve.fem.CoefficientFunction) -> None:

        u = space.TrialFunction()
        v = space.TestFunction()
        equation = sigma * ngsolve.grad(u) * ngsolve.grad(v) * ngsolve.dx
        self.__a = ngsolve.BilinearForm(space=space, symmetric=True)
        self.__a += equation
        self.__f = ngsolve.LinearForm(space=space)
        self.__preconditioner = ngsolve.Preconditioner(bf=self.__a,
                                                       type="bddc",
                                                       coarsetype="local")

    def solve_bvp(self, input: ngsolve.comp.GridFunction) \
            -> ngsolve.la.DynamicVectorExpression:
        """Solve the boundary value problem.

        Parameters
        ----------
        input : ngsolve.comp.GridFunction
            Input values of the Laplace function.

        Returns
        -------
        ngsolve.la.DynamicVectorExpression
            Numerical solution of the boundary value problem.
        """

        self.__a.Assemble()
        self.__f.Assemble()
        inverse = ngsolve.CGSolver(mat=self.__a.mat,
                                   pre=self.__preconditioner.mat,
                                   printrates=True,
                                   maxsteps=10000,
                                   precision=1e-12)
        r = self.__f.vec.CreateVector()
        r.data = self.__f.vec - self.__a.mat * input.vec
        return input.vec.data + inverse * r