from ossdbs.bounding_box import BoundingBox
import nibabel
import numpy as np


class Nifti1Image:
    """Intrerface for Nifti1 image.

    Attributes
    ----------
    file_path : str
        File path of a Nifti1 image.
    """

    __N_DIMENSION = 3

    def __init__(self, file_path: str) -> None:
        self._image = self.__load_image(file_path)

    def data_map(self) -> np.memmap:
        """Return the data of the nifti1 image.

        Returns
        -------
        np.memmap
        """
        return self._image.get_fdata()

    def bounding_box(self) -> BoundingBox:
        """Return the bounding box of the voxel data.

        Returns
        -------
        BoundingBox
        """
        start = self.offset()
        shape = np.array(self.xyz_shape(), dtype=np.float64)
        ends = start + shape * self.voxel_size()
        return BoundingBox(tuple(start), tuple(ends))

    def header(self) -> nibabel.nifti1.Nifti1Header:
        """Return the header of the nifti1 image.

        Returns
        -------
        nibabel.nifti1.Nifti1Header
        """
        return self._image.header

    def offset(self) -> tuple:
        """Returns the lowest cartesian coordinates of the voxel data.

        Returns
        -------
        tuple
        """
        offset = np.array([self._image.header['qoffset_x'],
                           self._image.header['qoffset_y'],
                           self._image.header['qoffset_z']
                           ], dtype=np.float64)
        return offset * self.__scaling()

    def voxel_size(self) -> tuple:
        """Returns the sizes of a voxel in x-, y- and z-direction.

        Returns
        -------
        tuple
        """
        x, y, z = self._image.header.get_zooms()[:self.__N_DIMENSION]
        return tuple(np.array((x, y, z), dtype=np.float64) * self.__scaling())

    def xyz_shape(self) -> tuple:
        """Returns the number of voxels in x-, y- and z-direction.

        Returns
        -------
        tuple
        """
        return self._image.header.get_data_shape()[:self.__N_DIMENSION]

    @staticmethod
    def __load_image(file_path: str) -> nibabel.nifti1.Nifti1Image:
        try:
            return nibabel.load(file_path)
        except FileNotFoundError:
            raise IOError('File Not Found.')

    def __scaling(self) -> float:
        xyz_unit = self._image.header.get_xyzt_units()[0]
        return {'unknown': 1.0,
                'meter': 1.0e3,
                'mm': 1.0,
                'micron': 1.0e-3}[xyz_unit]