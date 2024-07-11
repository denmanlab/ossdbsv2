# # Copyright 2023, 2024 Konstantin Butenko, Shruthi Chakravarthy
# # Copyright 2023, 2024 Jan Philipp Payonk, Johannes Reding
# # Copyright 2023, 2024 Tom Reincke, Julius Zimmermann
# # SPDX-License-Identifier: GPL-3.0-or-later


# from dataclasses import dataclass, asdict
# import logging

# import netgen
# import netgen.occ as occ
# import numpy as np

# from .electrode_model_template import ElectrodeModel, abstractmethod
# from .utilities import get_highest_edge, get_lowest_edge


# _logger = logging.getLogger(__name__)
# @dataclass
# class NeuronexusParameters:
#     """Electrode geometry parameters."""

#     # dimensions [mm]
    
#     def __init__(self, tip_length,contact_length, contact_spacing, lead_diameter, total_length):
#         self.tip_length=tip_length
#         self.contact_length=contact_length
#         self.contact_spacing=contact_spacing
#         self.lead_diameter=lead_diameter
#         self.total_length=total_length
    

#     def get_center_first_contact(self) -> float:
#         """Returns distance between electrode tip and center of first contact."""
#         return 0.5 * self.tip_length

#     def get_distance_l1_l4(self) -> float:
#         """Returns distance between first level contact and fourth level contacts."""
#         return 3 * (self.contact_length + self.contact_spacing)


# class NeuronexusModel(ElectrodeModel):
#     """Deep Brain Simulation electrode.

#     Attributes
#     ----------
#     rotation : float
#         Rotation angle in degree of electrode.

#     direction : tuple
#         Direction vector (x,y,z) of electrode.

#     position : tuple
#         Position vector (x,y,z) of electrode tip.

#     n_contacts: int
#         Hard-coded number of electrode contacts.

#     index: int
#         Index of the electrode. Important for the model
#         generation later and unambigous naming of boundaries.
#     """

#     _n_contacts= 16
    
#     def __init__(
#         self,
#         parameters: dataclass,
#         rotation: float = 0,
#         direction: tuple = (0, 0, 1),
#         position: tuple = (0, 0, 0),
#     )-> None:
#         self._position = position
#         self._rotation = rotation
#         norm = np.linalg.norm(direction)
#         self._direction = tuple(direction / norm) if norm else (0, 0, 1)

#         self._boundaries = {"Body": "Body"}
#         for idx in range(1, 17):
#             self._boundaries[f"Contact_{idx}"] = f"Contact_{idx}"

#         self._parameters = parameters
#         #self.parameter_check()

#         self._geometry = self._construct_geometry()
#         self._encapsulation_thickness = 0.0
#         self._index = 0
#         #added this line
#         self._encapsulation_geometry = self._construct_encapsulation_geometry(self._encapsulation_thickness)
        
#     '''
#     def parameter_check(self):
#         """Check electrode parameters."""
#         # ensure that all parameters are at least 0
#         for param in asdict(self._parameters).values():
#             if param < 0:
#                 raise ValueError("Parameter values cannot be less than zero")
#         # check that number of contacts has been set correctly
#         if not isinstance(self.n_contacts, int):
#             raise ValueError(
#                 "The number of contacts has to be supplied as an integer value."
#             )
#     '''
#     @property
#     def n_contacts(self) -> int:
#         """Returns number of contacts."""
#         return self._n_contacts

#     @property
#     def boundaries(self) -> dict:
#         """Returns names of boundaries."""
#         return self._boundaries

#     @property
#     def geometry(self) -> netgen.libngpy._NgOCC.TopoDS_Shape:
#         """Return geometry of electrode.

#         Returns
#         -------
#         netgen.libngpy._NgOCC.TopoDS_Shape
#         """
#         return self._geometry

#     @property
#     def encapsulation_thickness(self) -> float:
#         """Thickness of encapsulation layer."""
#         return self._encapsulation_thickness

#     @encapsulation_thickness.setter
#     def encapsulation_thickness(self, thickness: float) -> None:
#         self._encapsulation_geometry = self._construct_encapsulation_geometry(thickness)
#         self._encapsulation_thickness = thickness

#     def encapsulation_geometry(
#         self, thickness: float
#     ) -> netgen.libngpy._NgOCC.TopoDS_Shape:
#         """Generate geometry of encapsulation layer around electrode.

#         Parameters
#         ----------
#         thickness : float
#             Thickness of encapsulation layer.

#         Returns
#         -------
#         netgen.libngpy._NgOCC.TopoDS_Shape
#         """
#         if np.less(thickness, 1e-3):
#             raise ValueError(
#                 "The specified thickness is too small. Choose a larger, positive value."
#             )
#         if not np.isclose(thickness, self._encapsulation_thickness):
#             return self._construct_encapsulation_geometry(thickness)
#         return self._encapsulation_geometry

#     #@abstractmethod
#     def _construct_geometry(self) -> netgen.libngpy._NgOCC.TopoDS_Shape:
#         contacts = self._contacts()
#         # TODO check
#         electrode = occ.Glue([self.__body() - contacts, contacts])
#         print(electrode)
#         print(type(electrode))
#         axis = occ.Axis(p=(0, 0, 0), d=self._direction)
#         rotated_electrode = electrode#electrode.Rotate(axis=axis, ang=self._rotation)
#         return electrode#rotated_electrode.Move(v=self._position)
    
#     def __body(self) -> netgen.libngpy._NgOCC.TopoDS_Shape:
#         radius = self._parameters.lead_diameter * 0.5
#         center = tuple(np.array(self._direction) * radius)
#         height = self._parameters.total_length - self._parameters.tip_length
#         body = occ.Cylinder(p=center, d=self._direction, r=radius, h=height)
#         body.bc(self._boundaries["Body"])
#         return body
#     def _contacts(self) -> netgen.libngpy._NgOCC.TopoDS_Shape:
#         radius = self._parameters.lead_diameter * 0.5
#         direction = self._direction
#         center = tuple(np.array(direction) * radius)
#         # define half space at tip_center
#         # to construct a hemisphere as part of the contact tip
#         half_space = netgen.occ.HalfSpace(p=center, n=direction)
#         contact_tip = occ.Sphere(c=center, r=radius) * half_space
#         h_pt2 = self._parameters.tip_length - radius
#         contact_pt2 = occ.Cylinder(p=center, d=direction, r=radius, h=h_pt2)
#         contact_1 = contact_tip + contact_pt2

#         vectors = []
#         distance = self._parameters.tip_length + self._parameters.contact_spacing
#         for _ in range(0, 3):
#             vectors.append(tuple(np.array(self._direction) * distance))
#             distance += (
#                 self._parameters.contact_length + self._parameters.contact_spacing
#             )

#         point = (0, 0, 0)
#         height = self._parameters.contact_length
#         axis = occ.Axis(p=point, d=self._direction)
#         contact_8 = occ.Cylinder(p=point, d=self._direction, r=radius, h=height)
#         contact_directed = self._contact_directed()

#         contacts = [
#             contact_1,
#             contact_directed.Move(v=vectors[0]),
#             contact_directed.Rotate(axis, 120).Move(v=vectors[0]),
#             contact_directed.Rotate(axis, 240).Move(v=vectors[0]),
#             contact_directed.Move(v=vectors[1]),
#             contact_directed.Rotate(axis, 120).Move(v=vectors[1]),
#             contact_directed.Rotate(axis, 240).Move(v=vectors[1]),
#             contact_8.Move(v=vectors[2]),
#         ]

#         for index, contact in enumerate(contacts, 1):
#             name = self._boundaries[f"Contact_{index}"]
#             contact.bc(name)
#             # Label max z value and min z value for contact_8
#             if name == "Contact_8":
#                 min_edge = get_lowest_edge(contact)
#                 min_edge.name = name
#             # Only label contact edge with maximum z value for contact_1
#             #if name == "Contact_1" or name == "Contact_8":
#                 #max_edge = get_highest_edge(contact)
#                 #max_edge.name = name
#             else:
#                 # Label all the named contacts appropriately
#                 for edge in contact.edges:
#                     if edge.name is not None:
#                         edge.name = name
#         return netgen.occ.Fuse(contacts)
#     def _contact_directed(self) -> netgen.libngpy._NgOCC.TopoDS_Shape:
#         point = (0, 0, 0)
#         radius = self._parameters.lead_diameter * 0.5
#         height = self._parameters.contact_length
#         body = occ.Cylinder(p=point, d=self._direction, r=radius, h=height)
#         # tilted y-vector marker is in YZ-plane and orthogonal to _direction
#         new_direction = (0, self._direction[2], -self._direction[1])
#         eraser = occ.HalfSpace(p=point, n=new_direction)
#         delta = 15
#         angle = 30 + delta
#         axis = occ.Axis(p=point, d=self._direction)

#         contact = body - eraser.Rotate(axis, angle) - eraser.Rotate(axis, -angle)
#         # Centering contact to label edges
#         contact = contact.Rotate(axis, angle)
#         # TODO refactor / wrap in function
#         # Find  max z, min z, max x, and max y values and label min x and min y edge
#         max_z_val = max_y_val = max_x_val = float("-inf")
#         min_z_val = float("inf")
#         for edge in contact.edges:
#             if edge.center.z > max_z_val:
#                 max_z_val = edge.center.z
#             if edge.center.z < min_z_val:
#                 min_z_val = edge.center.z
#             if edge.center.x > max_x_val:
#                 max_x_val = edge.center.x
#                 max_x_edge = edge
#             if edge.center.y > max_y_val:
#                 max_y_val = edge.center.y
#                 max_y_edge = edge
#         max_x_edge.name = "max x"
#         max_y_edge.name = "max y"
#         # Label only the outer edges of the contact with min z and max z values
#         for edge in contact.edges:
#             if np.isclose(edge.center.z, max_z_val) and not (
#                 np.isclose(edge.center.x, radius / 2)
#                 or np.isclose(edge.center.y, radius / 2)
#             ):
#                 edge.name = "max z"
#             elif np.isclose(edge.center.z, min_z_val) and not (
#                 np.isclose(edge.center.x, radius / 2)
#                 or np.isclose(edge.center.y, radius / 2)
#             ):
#                 edge.name = "min z"

#         # TODO check that the starting axis of the contacts
#         # are correct according to the documentation
#         contact = contact.Rotate(axis, -angle)

#         return contact

#     #@abstractmethod
#     def _construct_encapsulation_geometry(
#         self, thickness: float
#     ) -> netgen.libngpy._NgOCC.TopoDS_Shape:
#         """Generate geometry of encapsulation layer around electrode.

#         Parameters
#         ----------
#         thickness : float
#             Thickness of encapsulation layer.

#         Returns
#         -------
#         netgen.libngpy._NgOCC.TopoDS_Shape
#         """
#         radius = self._parameters.lead_diameter * 0.5 + thickness
#         center = tuple(np.array(self._direction) * self._parameters.lead_diameter * 0.5)
#         height = self._parameters.total_length - self._parameters.tip_length
#         tip = netgen.occ.Sphere(c=center, r=radius)
#         lead = occ.Cylinder(p=center, d=self._direction, r=radius, h=height)
#         encapsulation = tip + lead
#         encapsulation.bc("EncapsulationLayerSurface")
#         encapsulation.mat("EncapsulationLayer")
#         return encapsulation.Move(v=self._position) - self.geometry
    
#     def set_contact_names(self, boundaries: dict) -> None:
#         """Set the names of electrode contacts.

#         Parameters
#         ----------
#         boundaries : dict
#             {'Body': 'body_name',
#              'Contact_1': 'contact_name',
#              'Contact_2': ...}
#         """
#         if self._boundaries == boundaries:
#             _logger.info("Boundary names remain unchanged")
#             return

#         # TODO discuss if stricter checking required
#         # currently, typos would not be catched, for example
#         # checking that the keys are equivalent could help
#         for face in self.geometry.faces:
#             old_name = face.name
#             if old_name in boundaries:
#                 face.name = boundaries[old_name]
#         for edge in self.geometry.edges:
#             old_name = edge.name
#             if old_name in boundaries:
#                 edge.name = boundaries[old_name]

#         self._boundaries.update(boundaries)
#         _logger.info("Boundary names updated")

#     @property
#     def index(self) -> int:
#         """Index of electrode, relevant if multiple electrodes used."""
#         return self._index

#     @index.setter
#     def index(self, index: int) -> None:
#         self._index = index

#     def get_max_mesh_size_contacts(self, ratio: float) -> float:
#         """Use electrode's contact size to estimate maximal mesh size.

#         Parameters
#         ----------
#         ratio: float
#             Ratio between characteristic contact size and maximal mesh size.

#         Notes
#         -----
#         For most of the electrodes, the electrode diameter is used.
#         Exemptions are:
#         * :class:`ossdbs.electrodes.MicroProbesSNEX100Model`

#         """
#         return self._parameters.lead_diameter / ratio

#     def export_electrode(self, output_path, brain_dict, n_electrode) -> None:
#         """Export electrode as VTK file."""
#         _logger.info("Export electrode as VTK file")
#         height = (
#             np.amax(
#                 [
#                     brain_dict["Dimension"]["x[mm]"],
#                     brain_dict["Dimension"]["y[mm]"],
#                     brain_dict["Dimension"]["z[mm]"],
#                 ]
#             )
#             / 2
#         )
#         try:
#             radius = self._parameters.lead_diameter / 2
#         except AttributeError:
#             radius = 1  # Set larger radius in case lead_diameter is not defined

#         cylinder = netgen.occ.Cylinder(
#             p=self._position,
#             d=self._direction,
#             r=radius,
#             h=height,
#         )

#         print(cylinder)
#         print(self.geometry)
#         occgeo = occ.OCCGeometry(cylinder * self.geometry)
#         mesh_electrode = Mesh(occgeo.GenerateMesh())
#         bnd_dict = {}
#         for idx, contact in enumerate(self.boundaries):
#             bnd_dict[contact] = idx
#         boundary_cf = mesh_electrode.BoundaryCF(bnd_dict, default=-1)

#         VTKOutput(
#             ma=mesh_electrode,
#             coefs=[boundary_cf],
#             names=["boundaries"],
#             filename=f"{output_path}/electrode_{n_electrode}",
#             subdivision=0,
#         ).Do(vb=BND)

# Copyright 2023, 2024 Konstantin Butenko, Shruthi Chakravarthy
# Copyright 2023, 2024 Jan Philipp Payonk, Johannes Reding
# Copyright 2023, 2024 Tom Reincke, Julius Zimmermann
# SPDX-License-Identifier: GPL-3.0-or-later


from dataclasses import dataclass, asdict
import logging

import netgen
import netgen.meshing as ngm
import netgen.occ as occ
import numpy as np

from .electrode_model_template import ElectrodeModel, abstractmethod
from .utilities import get_highest_edge, get_lowest_edge


_logger = logging.getLogger(__name__)
@dataclass
class NeuronexusParameters:
    """Electrode geometry parameters."""

    # dimensions [mm]
    tip_length: float
    # tip_diameter: float
    contact_spacing: 50
    contact_length: float
    lead_diameter: 15
    total_length: 5000

    def get_center_first_contact(self) -> float:
        """Returns distance between electrode tip and center of first contact."""
        return 0.5 * self.tip_length

    def get_distance_l1_l4(self) -> float:
        """Returns distance between first level contact and fourth level contact."""
        return -1.0


class NeuronexusModel(ElectrodeModel):
    """MicroElectrode.

    Attributes
    ----------
    parameters : MicroElectrodeParameters
        Parameters for MicroElectrode geometry.

    rotation : float
        Rotation angle in degree of electrode.

    direction : tuple
        Direction vector (x,y,z) of electrode.

    position : tuple
        Position vector (x,y,z) of electrode tip.
    """

    _n_contacts = 16
    #note: changing n_contacts leads to n_contacts+1 electrodes
    def _construct_encapsulation_geometry(
        self, thickness: float
    ) -> netgen.libngpy._NgOCC.TopoDS_Shape:
        """Generate geometry of encapsulation layer around electrode.

        Parameters
        ----------
        thickness : float
            Thickness of encapsulation layer.

        Returns
        -------
        netgen.libngpy._NgOCC.TopoDS_Shape
        """
        center = tuple(np.array(self._direction) * self._parameters.tip_length * 0.5)
        radius = self._parameters.tip_length
        height = self._parameters.total_length - self._parameters.tip_length * 0.5
        #this tip statement is not used
        tip = occ.Sphere(c=center, r=radius)
        lead = occ.Cylinder(p=center, d=self._direction, r=radius, h=height)
        encapsulation = tip + lead
        encapsulation.bc("EncapsulationLayerSurface")
        encapsulation.mat("EncapsulationLayer")
        return encapsulation.Move(v=self._position) - self.geometry

    def _construct_geometry(self) -> netgen.libngpy._NgOCC.TopoDS_Shape:
        contact = self._contacts()
        electrode = netgen.occ.Glue([self.__body(), contact])
        return electrode.Move(v=self._position)

    def __body(self) -> netgen.libngpy._NgOCC.TopoDS_Shape:
        radius = self._parameters.tip_length
        radius_lead = self._parameters.lead_diameter * 0.5
        center = tuple(np.array(self._direction) * self._parameters.tip_length)
        height_lead = self._parameters.total_length - self._parameters.tip_length
        lead=occ.Box((0,-radius/2, 1),(radius_lead,radius/2, 1000))
        lead.bc(self._boundaries["Body"])
        return lead

    def _contacts(self) -> netgen.libngpy._NgOCC.TopoDS_Shape:
        point = netgen.libngpy._NgOCC.gp_Pnt(0, 0, 0)
        radius = self._parameters.lead_diameter * 0.5
        height = self._parameters.contact_length
        diff_direction = netgen.libngpy._NgOCC.gp_Dir(1, 0, 0)
        cyl = occ.Cylinder(p=point, d=diff_direction, h=radius, r=height) 
        contact = cyl
        contact.col = (0,0,0)
        distance = 50
        contacts = []
        for count in range(self._n_contacts):
            name = self._boundaries[f"Contact_{count + 1}"]
            contact.bc(name)
            contact.col = (0,0,0)
            min_edge = get_lowest_edge(contact)
            max_edge = get_highest_edge(contact)
            min_edge.name = name
            max_edge.name = name
            # axis = occ.Axis(p=(0, 0, 0), d=(0.,1.,0.))
            # contact = contact.Rotate(axis=axis,ang=90.).Move((0.,self._parameters.lead_diameter * 0.45,0.))
            vector = tuple(np.array(self._direction) * distance)
            contacts.append(contact.Move(vector))
            distance+=50
            #distance += (
            #    self._parameters.contact_length + self._parameters.contact_spacing
            #)
        #contacts.extend([box])
        glued= netgen.occ.Glue(contacts)
        return glued
