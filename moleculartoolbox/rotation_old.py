# -*- coding: utf-8 -*-
import sys
# import re
# import os
import numpy as np
# from chemphysconst import PeriodicTable
from chemphysconst import Constants
# from collections import OrderedDict
# from operator import itemgetter
from numpy import linalg

FLOAT = np.float128
CONST = Constants()
# PT = PeriodicTable()


class Rotation(object):
    """docstring for Rotation"""
    def __init__(self, geometry, **kwarg):
        """
        Geometry must be a moleculartoolbox.geometry type object.
        Possible keyword arguments are:

        """
        super(Rotation, self).__init__()
        self.geometry = geometry
        self.kwarg = self.check_keywords(kwarg)

    def determinant_3x3(self, matA):
        '''
        Returns the determinant of a 3x3 matrix. While we could use the
        np.linalg.det routine, it only allows for np.float64. This is an
        analytical form not limited by precision.
        '''
        e = CONST.third_order_LeviCevita_Tensor()
        det = np.einsum('ijk,i,j,k', e, matA[0], matA[1], matA[2])
        return det

    def diagonalise_3x3_symmetric(self, matA):
        '''
        Diagonalises a 3x3 symmetric real matrix and returns the eigenvalues
        and eigenvectors. While we could use the np.linalg.eigh routine, it
        only allows for np.float64. These are analytical forms not limited
        by precision.
        '''
        p1 = matA[0, 1]**2 + matA[0, 2]**2 + matA[1, 2]**2
        if p1 == 0.0:
            # A is diagonal.
            eig1 = matA[0, 0]
            eig2 = matA[1, 1]
            eig3 = matA[2, 2]
        else:
            matI = np.eye(3, dtype=FLOAT)
            q = np.trace(matA) / 3.0
            p2 = ((matA[0, 0] - q)**2 +
                  (matA[1, 1] - q)**2 +
                  (matA[2, 2] - q)**2 + 2 * p1)
            p = np.sqrt(p2 / 6.0)
            B = (1.0 / p) * (matA - q * matI)
            r = self.determinant_3x3(B) / 2.0

            # In exact arithmetic for a symmetric matrix  -1 <= r <= 1
            # but computation error can leave it slightly outside this range.
            if r <= -1:
                phi = np.pi / 3.0
            elif r >= 1:
                phi = 0.0
            else:
                phi = np.arccos(r) / 3.0

            # the eigenvalues satisfy eig3 <= eig2 <= eig1
            eig1 = q + 2.0 * p * np.cos(phi)
            eig3 = q + 2.0 * p * np.cos(phi + (2.0 * np.pi / 3.0))
            eig2 = 3.0 * q - eig1 - eig3  # since trace(A) = eig1 + eig2 + eig3
        # We want it in reversed order though
        eig = np.array([eig3, eig2, eig1], dtype=FLOAT)
        print (matA - eig1 * matI) * (matA - eig2 * matI)
        return eig

    def moment_of_inertia_tensor(self):
        '''
        Returns the moment of inertia tensor [unit = u*(Angstrom^2)] where
         I_ij = e_ikm * e_jlm * (m_a * r_ak * r_al) (Einstein sum notation)
        '''
        masses = self.mass_array()
        coords = self.geom_array()
        e = CONST.third_order_LeviCevita_Tensor()
        moI = np.einsum('ikm,jlm,a,ak,al->ij', e, e, masses, coords, coords,
                        casting='same_kind', dtype=FLOAT)
        # The above short-cut the following nasty cross product
        # i_xx = masses.dot(coords[:, 1] ** 2 + coords[:, 2] ** 2)
        # i_yy = masses.dot(coords[:, 0] ** 2 + coords[:, 2] ** 2)
        # i_zz = masses.dot(coords[:, 0] ** 2 + coords[:, 1] ** 2)
        # i_xy = masses.dot(coords[:, 0] * coords[:, 1])
        # i_xz = masses.dot(coords[:, 0] * coords[:, 2])
        # i_yz = masses.dot(coords[:, 1] * coords[:, 2])

        # moI = np.array([[i_xx, -i_xy, -i_xz],
        #                 [-i_xy, i_yy, -i_yz],
        #                 [-i_xz, -i_yz, i_zz]], dtype=np.float64)
        return moI

    def rotate_to_principal_axes_frame(self):
        '''
        Moves/rotates the coordinates into their principle axis frame
            1. shift centre of mass
            2. diagonalise moment_of_inertia_tensor
            3. rotate around resulting X_rot
            4. Update coordinates
        Returns the rotational Matrix X_rot and the centre of mass
        '''
        geom = self.geometry
        coords = geom.geom_array()
        # 1. shift centre of mass
        r_com = geom.center_of_mass()
        coords -= r_com
        # 2. diagonalise moment_of_inertia_tensor
        moI = self.moment_of_inertia_tensor()
        x_rot = np.array(np.linalg.eigh(np.float64(moI.astype(np.float64)))[1])
        # print self.diagonalise_3x3_symmetric(moI)
        # print X_rot
        # sys.exit()
        # 3. Rotate to get the principle axis frame coordinates
        coords = np.transpose(np.transpose(x_rot).dot(np.transpose(coords)))
        # 4. Update coordinates
        check = 0
        for i, atom in enumerate(self.atoms):
            check += atom.update_coordinates(coords[i])
        # Only return those if we updated all coordinates
        if check == len(coords):
            return x_rot, r_com
        else:
            sys.exit('Failed to update rotated coordinates.')

    def check_if_moI_diagonal(self, moI):
        '''
        Returns true if the moment of inertia tensor is diagonalised
        '''
        rel_tol = 1.0e-9
        abs_tol = 1.0e-6
        moI_diag = np.array(linalg.eigh(moI.astype(np.float64))[0])
        if np.allclose(np.diag(moI_diag), moI, rtol=rel_tol, atol=abs_tol):
            return True

    def rotational_symmetry(self):
        '''
        Returns an array of sorted rotational constants [unit = u*(Angstrom^2)]
         as well as a the rotational symmetry type.
        Four types:
            1. Symmetric top
                a) prolate symmetric top (I_xx = I_yy > I_zz)
                   [I_xx -> I_B, I_zz -> I_A]
                b) oblate  symmetric top (I_xx = I_yy < I_zz)
                   [I_xx->I_B, I_zz->I_C]
            2. Linear (I_xx = I_yy, I_zz = 0)
                   [I_xx -> I_B]
            3. Spherical top (I_xx = I_yy = I_zz)
                   [I_xx -> I_B]
            4. Asymmetric top (I_xx != I_yy != I_zz)
                   [sorted([I_xx, I_yy, I_zz]) = [I_A, I_B, I_C]]
        '''
        tol = 1.0e-6
        moI = self.moment_of_inertia_tensor()
        I_xx, I_yy, I_zz = moI[0, 0], moI[1, 1], moI[2, 2]
        if np.isclose(I_xx, I_yy, rtol=tol):
            if np.isclose(0.0, I_zz, rtol=tol):
                self.is_linear = True
                return np.array([0.0, I_xx, 0.0]), "Linear"
            elif np.isclose(I_xx, I_zz, rtol=tol):
                return np.array([0.0, I_xx, 0.0]), "Spherical top"
            elif I_xx > I_zz:
                return np.array([I_zz, I_xx, 0.0]), "Prolate symmetric top"
            else:
                return np.array([0.0, I_zz, I_xx]), "Oblate symmetric top"
        else:
            return np.array(sorted([I_xx, I_yy, I_zz],
                                   reverse=True)), "Asymmetric top"


