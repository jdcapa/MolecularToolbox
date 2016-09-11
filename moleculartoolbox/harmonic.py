# -*- coding: UTF-8 -*-
"""This module contains functions related to the harmonic approximation."""
import sys
import numpy as np
from chemphysconst import Constants
from numpy import linalg
from . import printfunctions as PF

FLOAT = np.float128
CONST = Constants()
COMPLEX = np.complex256


class Harmonic(object):
    """
    This class does the rigid-rotor double harmonic approximation analysis.

    It requires at least a geometry and a Hessian matrix as input.
    Further properties can be included, e.g. the dipole derivatives.
    """

    def __init__(self, geometry, hessian=None, **kwargs):
        """
        Initiate the class with at least geometry and a Hessian object.

        geometry: N-atomic geometry object
        gradient: Nx3 numpy array with gradients (Eh/bohr)
        hessian:  3Nx3N numpy Hessian array (Eh/(bohr^2))

        kwargs["dipole derivatives"] : 3Nx3N numpy array of the dipole
                                       derivatives required for intensities
                                       (Eh*bohr)^1/2)
        """
        super(Harmonic, self).__init__()
        try:
            geometry.atoms
        except AttributeError:
            sys.exit("Harmonic.__init__(): "
                     "A valid geometry is necessary")
        else:
            self.geometry = geometry

        if type(hessian) == np.ndarray:
            self.hessian = hessian
        else:
            sys.exit("Harmonic.__init__(): "
                     "A valid Hessian is necessary")

        if 'gradient' in kwargs:
            if type(kwargs["gradient"]) == np.ndarray:
                self.gradient = kwargs["gradient"]
            else:
                sys.exit("Harmonic.__init__(): "
                         "Valid gradient are necessary")
        else:
            self.gradient = None

        if 'dipole_derivatives' in kwargs:
            dipole_derivatives = kwargs["dipole_derivatives"]
            if type(dipole_derivatives) == np.ndarray:
                self.dipole_derivatives = dipole_derivatives
            else:
                print("Harmonic.__init__(): "
                      "Valid dipole derivatives are missing")
                self.dipole_derivatives = None
        else:
            self.dipole_derivatives = None

        self.diag_Hess, self.mat_L = self.diagonalise_Hessian()
        self.rotation = self.geometry.rot_prop
        self.rot_const_inv_cm = self.rotation.rigid_rotational_constants(
            "1/cm", False)
        self.freq_inv_cm = self.harmonic_frequencies(self.diag_Hess, "1/cm")

    def norm_of_grad(self):
        """Return the norm of the gradient."""
        if np.any(self.gradient):
            return linalg.norm(self.gradient.astype(np.float64))
        else:
            return 0

    def rms_of_grad(self):
        """Return the norm of the gradient."""
        if np.any(self.gradient):
            return self.norm_of_grad() / (3.0 * len(self.geometry.atoms))
        else:
            return 0.0

    def rotate_Hessian(self, x_rot):
        """
        Rotate the Hessian according to a rotational matrix.

        This matrix is usually obtained by diagonalising the moment of inertia
         tensor (or any other 3x3 rotational matrix).
        """
        nAtoms = self.geometry.nAtoms
        hessian = self.hessian
        # We need to operate on sets of 3 in the 3Nx3N Hessian matrix
        for i in range(3 * nAtoms):
            for s in range(nAtoms):
                b_c, e_c = 3 * s, 3 * s + 3
                xyz = hessian[i, b_c: e_c]
                hessian[i, b_c: e_c] = np.dot(x_rot, xyz)
        for j in range(3 * nAtoms):
            for t in range(nAtoms):
                b_l, e_l = 3 * t, 3 * t + 3
                xyz = hessian[b_l: e_l, j]
                hessian[b_l: e_l, j] = np.dot(x_rot, xyz)
        # Double check this, it might be done only once.
        return hessian

    def mass_weight_Hessian(self, hessian):
        """
        Return a mass-weighted Hessian (3N x 3N numpy array).

        The unit of the matrix elements is Eh/(amu*bohr^2).
        """
        masses_3N = self.geometry.threeN_mass_array()
        inv_sq_masses = 1.0 / np.sqrt(np.array(masses_3N))
        return ((inv_sq_masses * hessian).T * inv_sq_masses).T

    def diagonalise_Hessian(self):
        """
        Return the diagonalised mass-weighted Hessian and a matrix L.

        The latter matrix contains the normalised eigenvectors in its columns.
        The elements of the diagonalised Hessian have the dimension
         Eh/(amu*bohr^2) while the elements of L are unitless.
        """
        # 1. mass-weight Hessian
        mw_hessian = self.mass_weight_Hessian(self.hessian)
        # 2. diagonalise mass-weighted Hessian
        diag_Hess, mat_L = linalg.eigh(mw_hessian.astype(np.float64))

        return (np.array(diag_Hess, dtype=FLOAT),
                np.array(mat_L, dtype=FLOAT))

    def harmonic_frequencies(self, diag_Hess, unit="1/cm"):
        """
        Return the harmonic frequencies.

        This assumes a diagonalised Hessian and comes in units Eh/(amu*bohr^2).
        Units:
            1/cm, cm**-1 or cm^-1
            1/s or Hz
            1/s^2 or 1/s^2
        """
        hartee_to_joule = CONST.hartree_energy("J")
        bohr_to_meter = CONST.bohr_radius()
        u_to_kg = CONST.atomic_mass_constant()
        c = CONST.speed_of_light()  # m/s

        conversion_factor = hartee_to_joule / (4 * np.pi**2 *
                                               bohr_to_meter**2 * u_to_kg)

        if unit in ("1/cm", "cm**-1", "cm^-1"):
            conversion_factor /= (c**2 * 1e4)
        elif unit in ('1/s^2', 'Hz^2'):
            return diag_Hess * conversion_factor
        elif not (unit in ('1/s', 'Hz')):
            sys.exit("Harmonic.harmonic_frequencies(): "
                     "Unknown unit passed to function")
        complex_diag_Hess = np.array(diag_Hess, dtype=COMPLEX)
        harm_freq = np.sqrt(complex_diag_Hess * conversion_factor)
        return harm_freq

    def harmonic_intensities(self):
        """
        Return the harmonic intensities in km/mol.

        This assumes the dipole derivatives comes in units of (Eh*bohr)^1/2.
        """
        if not np.any(self.dipole_derivatives):
            return
        sqrt_masses = np.sqrt(self.geometry.threeN_mass_array())
        u_to_kg = CONST.atomic_mass_constant()
        c = CONST.speed_of_light()
        avogardro = CONST.avogadro_constant()
        bohr_to_meter = CONST.bohr_radius()
        hartee_to_joule = CONST.hartree_energy("J")

        conversion_factor = (np.pi * bohr_to_meter * hartee_to_joule *
                             avogardro * 1e-3 / (3 * c**2 * u_to_kg))
        mw_DipDer = (self.dipole_derivatives.T / sqrt_masses).T
        normCo_deriv = np.einsum('ij,jx->ix', self.mat_L.T, mw_DipDer,
                                 casting='same_kind', dtype=FLOAT)
        harm_int = (conversion_factor *
                    np.array([np.sum(vec**2) for vec in normCo_deriv]))
        return harm_int

    def normal_coordinates(self, diag_Hess, mat_L):
        """Return the dimensionless normal coordinates as a matrix Q."""
        u_to_kg = CONST.atomic_mass_constant()
        h_bar = CONST.planck_constant() / (2 * np.pi)

        masses = np.sqrt(self.geometry.threeN_mass_array() * u_to_kg)
        harm_freq = self.harmonic_frequencies(diag_Hess, "1/s")
        threeN = 3 * self.geometry.nAtoms

        normal_coordinates = np.zeros((threeN, threeN), dtype=COMPLEX)

        for i in range(threeN):
            for n in range(threeN):
                normal_coordinates[n, i] = mat_L[n, i] * \
                    (masses[n] * 1e-10 * np.sqrt(harm_freq[i] / h_bar))

        return normal_coordinates

    def coriolis_zeta(self):
        """Generate three Coriolis zeta matrices (one for each x, y, z)."""
        nAtoms = self.geometry.nAtoms
        threeN = 3 * nAtoms

        e = CONST.third_order_LeviCevita_Tensor()
        l = np.zeros((3, nAtoms, threeN), dtype=FLOAT)
        for i in range(3):
            l[i] = self.mat_L[i::3, :]
        # This is a much cleaner solution than those awful for-loops
        cz = np.einsum('abc,bik,cil->akl', e, l, l,
                       casting='same_kind', dtype=FLOAT)

        # The following is deprecated (it's still kept in case I need to port):
        # nTransRot = self.geometry.nTransRot()
        # offset1 = [1, 2, 0]
        # offset2 = [2, 0, 1]
        # L = self.mat_L
        # cz = []
        # for h in range(3):
        #     cz.append(np.zeros((threeN, threeN), dtype=FLOAT))

        # for i in range(3):
        #     for r in range(nTransRot, threeN):
        #         for s in range(nTransRot, threeN):
        #             z = 0.0
        #             i1 = offset1[i]
        #             i2 = offset2[i]
        #             for a in range(nAtoms):
        #                 z += (L[3 * a + i1, r] * L[3 * a + i2, s] -
        #                       L[3 * a + i2, r] * L[3 * a + i1, s])
        #             cz[i][r, s] = z
        # print cz[0][nTransRot]
        return np.around(cz, decimals=13)

    def inertia_derivatives(self):
        """Return the inertia derivatives a_k^ab in [u^1/2 * Angs].

        (3Nx3x3 tensor) calculated according to
            Papousek/Alijev, 1982, isbn: 9780444997371, p. 277
        """
        nAtoms = self.geometry.nAtoms
        nTransRot = self.geometry.nTransRot()
        nVib = self.geometry.nVib()
        sqrt_masses = np.sqrt(self.geometry.mass_array())
        coords = self.geometry.geom_array()
        e = CONST.third_order_LeviCevita_Tensor()
        l = np.zeros((3, nAtoms, nVib), dtype=FLOAT)
        for i in range(3):
            l[i] = self.mat_L[i::3, nTransRot:]
        inertia_derivatives = 2 * np.einsum('ace,bde,i,ic,dik->kab',
                                            e, e, sqrt_masses, coords, l,
                                            casting='same_kind', dtype=FLOAT)
        print("Inertia derivatives")
        for k in range(nVib):
            for a in range(3):
                for b in range(3):
                    print (k, a, b, "{:20.10f}".format(inertia_derivatives[k,a,b]))
        sys.exit()
        return inertia_derivatives

    def cartesian_displacements(self, anharm_displacement=0.05, unit="bohr",
                                verbose=False):
        """
        Calculate the Cartesian displacements.

        At these displaced geometries another harmonic Hessian need to be
         generated; this is necessary in order to generated the numerical third
         (and some fourth) order derivatives needed for VPT2.

        Output: 0: cartesian_displacements, 1: unit
        """
        # Normalised eigenvectors from the Hessian diagonalisation
        mat_L = self.mat_L
        nTransRot = self.geometry.nTransRot()
        harm_freq = self.harmonic_frequencies(self.diag_Hess, "1/s")
        harm_vib = harm_freq[nTransRot:].real
        masses = self.geometry.threeN_mass_array()
        if not (harm_vib > 0).all():
            sys.exit("Harmonic.cartesian_displacements(): "
                     "Encountered vibrations smaller than zero. Check!")
        # Constants
        u_to_kg = CONST.atomic_mass_constant()
        #   Note the extra 1/2PI, that's cfour (I don't know why)
        h_bar = CONST.planck_constant() / (4 * np.pi**2)
        #   Note the factor -1, that's a cfour diagonalisation correction
        len_fac = -1 / CONST.bohr_radius()
        if unit in ["A", "Angs", "angs", "Angstroem", "angstroem"]:
            len_fac = -1e10
        sqrt_inv_masses = 1 / np.sqrt(masses * u_to_kg)
        sqrt_inv_vib = np.sqrt(h_bar / harm_vib)

        # This is the actual calculation
        cart_disps = (mat_L.T[nTransRot:] * len_fac * anharm_displacement *
                      np.outer(sqrt_inv_vib, sqrt_inv_masses))
        if verbose:
            print("h_bar\n", h_bar)
            print("harm_vib\n", harm_vib)
            print("sqrt_inv_masses\n", sqrt_inv_masses)
            print("sqrt_inv_vib\n", sqrt_inv_vib)
            print("tmpOuter\n", np.outer(sqrt_inv_vib, sqrt_inv_masses))
            print("Cartesian displacements in {}:\n".format(unit))
            print(PF.print_np_2Dmatrix(cart_disps, 8))

        return cart_disps, unit

    def displaced_geometries(self, print_header=True,
                             unit="bohr", precision=9, verbose=False):
        """
        Return a list of strings of back transformed displaced normal coords.

        The coordinates (positive and negative) are in either Bohr or
         Angstroem.
        This is used as a coordinate input for new Hessian calculations.
            cart_disps: 3N-nTransRot x 3N displacement np.array in [unit].
            print_header: Prints an xyz-file header if True.
            unit: Bohr or Angstroem
            precision: This should be around 9 (pretty print control)
        """
        # Printing set-up
        output_strings = []
        vec_str = " {{:> {},.{}f}}".format(5 + precision, precision)
        freq_str = "{}\nMode {}: w = {} cm^-1 - "

        if unit in ["A", "Angs", "angs", "Angstroem", "angstroem"]:
            len_fac = 1.0
        else:
            len_fac = 1.0 / CONST.bohr_radius("Angstroem")

        nAtoms = self.geometry.nAtoms
        nTransRot = self.geometry.nTransRot()
        harm_inv_cm = self.freq_inv_cm[nTransRot:].real
        cart_disps = self.cartesian_displacements(0.05, unit, verbose)

        for i, disp in enumerate(cart_disps[0]):
            # Header setup
            freq = PF.print_complex(harm_inv_cm[i], 1.0, precision)
            header_str = freq_str.format(nAtoms, i + 1, freq)
            if print_header:
                tmp_pos = header_str + "positive displacement\n"
                tmp_neg = header_str + "negative displacement\n"
            else:
                tmp_pos, tmp_neg = "", ""
            # Geometry String
            for a, atom in enumerate(self.geometry.atoms):
                v = np.zeros((3,))
                for j in range(3):
                    v[j] = disp[3 * a + j]
                #  atom.coordinates are given in Angstroem
                v_positive = atom.coordinates * len_fac + v
                v_negative = atom.coordinates * len_fac - v
                tmp_pos += atom.element \
                    + (vec_str * 3).format(*v_positive) + "\n"
                tmp_neg += atom.element \
                    + (vec_str * 3).format(*v_negative) + "\n"
            output_strings.append(tmp_pos[:-1])
            output_strings.append(tmp_neg[:-1])
        return output_strings
