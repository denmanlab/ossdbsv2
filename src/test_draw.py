
import netgen.occ as occ
import ngsolve
import numpy as np

class LaplaceEquation:

    def __init__(self, 
                 space: ngsolve.comp.H1, 
                 coefficient: ngsolve.fem.CoefficientFunction) -> None:

        u = space.TrialFunction()
        v = space.TestFunction()
        self.__a = ngsolve.BilinearForm(space=space, symmetric=True)

        data = np.array([np.eye(3)]*8).reshape(2,2,2,3,3)
        # coefficient = ngsolve.VoxelCoefficient(start=(0,0,0), end=(1,1,1), values=data, linear=False)

        self.__a += coefficient * ngsolve.grad(u) * ngsolve.grad(v) * ngsolve.dx
        self.__f = ngsolve.LinearForm(space=space)
        self.__preconditioner = ngsolve.Preconditioner(bf=self.__a, 
                                                        type="bddc",
                                                        coarsetype="h1amg")

    def solve_bvp(self, input: ngsolve.comp.GridFunction) \
                                         -> ngsolve.la.DynamicVectorExpression:
        """Solve boundary value problem."""
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

class VolumeConductor:
    """Model for representing a volume conductor.

    Attributes
    ----------
    mesh : Mesh
    
    conductivity : dict

    Methods
    -------
    evaluate_potential(mesh: Mesh, conductivity: dict)
        Evaluate the electric potential of volume conductor.

    """
    
    def __init__(self, mesh: int, conductivity: dict) -> None:
        self.__mesh = mesh
        conductivities = [conductivity[mat] for mat in mesh.materials()]
        self.__sigma = ngsolve.CoefficientFunction(coef=conductivities) 
        
    def evaluate_potential(self, 
                            n_dof_limit : int = 1e6, 
                            n_refinements: int  = 1) -> tuple:
        """Evaluate electrical potential of volume conductor.
        
        Parameters
        ----------
        n_dof_limit : int
            Maximal number of Demensions of Freedom of FEM space.

        n_refinements: int
            Number of mesh refinements.

        Returns
        -------

        return : tuple
            Postprocessed data: lectric_field, V_contact, Power, potential
        
        """
        potential = self.__solve_bvp()
        iterations = 0
        while (self.__mesh.space().ndof < n_dof_limit and 
                                                iterations < n_refinements):
            self.__mesh.refine_by_error(error=self.__error(potential))
            potential = self.__solve_bvp()
            iterations = iterations + 1
        return self.__postprocess(potential=potential)

    def __solve_bvp(self) -> ngsolve.comp.GridFunction:
        potential = ngsolve.GridFunction(space=self.__mesh.space())
        potential.Set(coefficient=self.__mesh.boundary_coefficients(),
                      VOL_or_BND=ngsolve.BND)
        equation = LaplaceEquation(space=self.__mesh.space(), 
                                   coefficient=self.__sigma)
        potential.vec.data = equation.solve_bvp(input=potential)
        return potential

    def __error(self, potential: ngsolve.comp.GridFunction) \
                                            -> ngsolve.fem.CoefficientFunction:
        flux = ngsolve.grad(potential)
        flux_potential = ngsolve.GridFunction(space=self.__mesh.flux_space())
        flux_potential.Set(coefficient=flux)
        difference = flux - flux_potential
        return difference * ngsolve.Conj(difference)

    def __postprocess(self, potential: ngsolve.comp.GridFunction) -> tuple:
        # MappedIntegrationPoint
        electric_field = ngsolve.sqrt(ngsolve.grad(potential) * \
                                                        ngsolve.grad(potential))
        electric_field = electric_field(
                            self.__mesh.ngsolvemesh()(0, 0, 0.02 / 2)) * 1e3  
        V_contact = ngsolve.Integrate(potential,
                                      self.__mesh.ngsolvemesh(), 
                                      definedon=self.__mesh.boundaries
                                                                    ("contact"))

        P = ngsolve.Integrate(ngsolve.grad(potential) * \
                        ngsolve.Conj(self.__sigma * ngsolve.grad(potential)), 
                        self.__mesh.ngsolvemesh())
        return electric_field, V_contact, P, potential




class Mesh:

    def __init__(self,
                geometry: int, 
                order: int, 
                boundaries: dict) -> None:
        self.__mesh = ngsolve.Mesh(ngmesh=geometry.ng_mesh())
        self.__mesh.Curve(order=order)
        self.__order = order
        self.__boundaries = boundaries

    def boundaries(self, name: str) -> ngsolve.comp.Region:
        return self.__mesh.Boundaries(pattern=name)

    def boundary_coefficients(self) -> ngsolve.fem.CoefficientFunction:
        return self.__mesh.BoundaryCF(values=self.__boundaries)

    def centroids_of_elements(self) -> list:
        shape = (self.__mesh.ne, 4, 3)
        vertices = np.array([self.__mesh[v].point 
                             for element in self.__mesh.Elements()
                             for v in element.vertices]).reshape(shape)
        return [list(c) for c in np.sum(vertices, axis=1) / 4]

    def flux_space(self) -> ngsolve.comp.HDiv:
        return ngsolve.HDiv(mesh=self.__mesh, 
                            order=self.__order-1,
                            complex=True)    

    def materials(self) -> tuple:
        return self.__mesh.GetMaterials()

    def ngsolvemesh(self) -> ngsolve.comp.Mesh:
        return self.__mesh

    def refine(self) -> None:
        self.__mesh.Refine()
        self.__mesh.Curve(order=self.__order)

    def refine_by_error(self, error: ngsolve.fem.CoefficientFunction) -> None:
        self.mark_elements_by_error(error)
        self.__mesh.Refine()
        self.__mesh.Curve(order=self.__order)

    def mark_elements_by_error(self, 
                               error: ngsolve.fem.CoefficientFunction) -> None:
        errors = ngsolve.Integrate(cf=error,
                                    mesh=self.__mesh, 
                                    VOL_or_BND=ngsolve.VOL,
                                    element_wise=True).real
        limit = 0.5 * max(errors)
        flags = [errors[el.nr] > limit for el in self.__mesh.Elements()] 
        for index, element in enumerate(self.__mesh.Elements()):
            self.__mesh.SetRefinementFlag(ei=element, refine=flags[index])

    def element_sizes(self) -> list:
        volumes = ngsolve.Integrate(cf=ngsolve.CoefficientFunction(1), 
                                    mesh=self.__mesh,
                                    element_wise=True).NumPy()
        return list((6 * volumes) ** 1 / 3)
        
    def space(self) -> ngsolve.comp.H1:
        dirichlet = '|'.join(str(key) for key in self.__boundaries.keys())
        return ngsolve.H1(mesh=self.__mesh, 
                          order=self.__order, 
                          dirichlet=dirichlet, 
                          complex=False, 
                          wb_withedges=False)

    def elements_in_region(self, point: tuple, r: int) -> np.ndarray:
        shape = (self.__mesh.ne, 4, 3)
        vertices = np.array([self.__mesh[v].point 
                             for element in self.__mesh.Elements()
                             for v in element.vertices]).reshape(shape)

        distance = np.sum((vertices - point) ** 2, axis=2)
        return list(np.any(distance <= r ** 2, axis=1))

    def foo(self):

        data = np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]).reshape(2,2,2)
        print(data)

       # data = np.array([np.eye(3)]*8).reshape(2,2,2,3,3)
        
        cf = ngsolve.VoxelCoefficient(start=(0,0,0), end=(1,1,1), values=data, linear=False)

        ngsolve.Draw(cf, self.__mesh, 'foo')


        volumes = ngsolve.Integrate(cf=cf, 
                                    mesh=self.__mesh,
                                    element_wise=True).NumPy()
        print(volumes)
 




class GeometryDummy:
    def __init__(self) -> None:
        model = occ.Box(p1=occ.Pnt(0,0,0), p2=occ.Pnt(1,1,1))
        model.bc('contact')
        model.mat('saline')
        self.__geometry = occ.OCCGeometry(model)

    def ng_mesh(self):
        return self.__geometry.GenerateMesh()


geo = GeometryDummy()
mesh = Mesh(geometry=geo, order=2, boundaries={"contact": 1.0, "wire": 0.0})
mesh.foo()


# model = VolumeConductor(mesh, {"saline" : 1278*1e-6/1e-2,})
# model.evaluate_potential()