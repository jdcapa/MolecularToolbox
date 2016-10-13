# -*- coding: UTF-8 -*-
"""This module contains functions related to the VPT2 approximation."""
import sys
import os
import numpy as np
from chemphysconst import Constants
from . import Geometry
from . import Harmonic
# from numpy import linalg
# from . import printfunctions as PF

# Module globals
FLOAT = np.float128
COMPLEX = np.complex256
CONST = Constants()
CORIOLIS_RESONANCE_THRESH = 10.0  # 1/cm


class VPT2_ForceFields(object):
    """This constructs higher-order force fields and property derivatives."""

    def __init__(self, path, harmonic, **kwargs):
        super(VPT2_ForceFields, self).__init__()
        self.path = path
        self.harmonic = harmonic
        self.geometry = harmonic.geometry
        self.kwargs = kwargs

        if 'anharm_displacement' not in kwargs:
            self.anharm_displacement = 0.05
        else:
            self.anharm_displacement = kwargs['anharm_displacement']

        self.has_cubic = False
        self.has_semiquartic = False

    def transform_to_normal_coordinate_fc(self, hessian):
        """
        Transform a Cartesian Hessian to normal mode force constants.

        Coming from a normal mode displacement we assume that the atom ordering
         stays the same!
        The Hessian should have the units Eh/(bohr^2).
        """
        # Get constants
        # h_bar = CONST.planck_constant() / (2 * np.pi)
        hartee_to_joule = CONST.hartree_energy("J")
        bohr_to_meter = CONST.bohr_radius()
        u_to_kg = CONST.atomic_mass_constant()  # 1.66053904e-27 kg
        c = CONST.speed_of_light()  # m/s
        threeN = len(hessian)

        # Mass-weight the Hessian
        mw_Hessian = self.harmonic.mass_weight_Hessian(hessian)
        mat_L = self.harmonic.mat_L
        diag_Hess = self.harmonic.diag_Hess
        harm_freq = self.harmonic.harmonic_frequencies(diag_Hess, "1/s")
        conversion_factor = hartee_to_joule / (4 * np.pi**2 * c * u_to_kg *
                                               bohr_to_meter**2 * 1e2)
        # Transformation
        nc_hessian = np.zeros((threeN, threeN), COMPLEX)
        for i in range(threeN):
            for j in range(threeN):
                for m in range(threeN):
                    for n in range(threeN):
                        nc_hessian[i, j] += (mat_L[m, i] * mat_L[n, j] *
                                             mw_Hessian[m, n])
                nc_hessian[i, j] *= 1 / np.sqrt(harm_freq[i] * harm_freq[j])
        # nc_hessian has the unit of J * s^2 * Eh / (amu * bohr^2)
        return nc_hessian * conversion_factor

    def transform_displaced_Hessians(self, hessians):
        """Return a list of Hessians transformed into the norm-coord-domain."""
        nc_Hessians = []
        for hessian in hessians:
            nc_Hessians.append(self.transform_to_normal_coordinate_fc(hessian))
        return nc_Hessians

    def check_cubic(self, phi):
        """
        Check if the cubic force constants fulfil the Schwarz equation.

         Phi_ijk = Phi_jik = Phi_jki = Phi_ikj = Phi_kij = Phi_kji
        >> Doesn't do any checks at the moment.
        >> Need to implement printing control and proper error handling.
        """
        nVib = self.geometry.nVib()
        # precision = 9
        # i_str = "{:> 3d} "
        # const_str = " {{:> {},.{}f}}".format(7 + precision, precision)
        # print_str = i_str * 3 + const_str * 6  # + '\n'
        for i in range(nVib):
            for j in range(i, nVib):
                for k in range(j, nVib):
                    # vec = [phi[i, j, k],
                    #        phi[j, i, k],
                    #        phi[j, k, i],
                    #        phi[i, k, j],
                    #        phi[k, i, j],
                    #        phi[k, j, i]]
                    # if np.abs(phi[i, j, k]) > 1e-4:
                    #     print print_str.format(i + 7, j + 7, k + 7, *vec)
                    phi[j, i, k] = phi[i, j, k]
                    phi[j, k, i] = phi[i, j, k]
                    phi[i, k, j] = phi[i, j, k]
                    phi[k, i, j] = phi[i, j, k]
                    phi[k, j, i] = phi[i, j, k]
        return phi

    def calculate_cubic_force_field(self, nc_Hessians):
        """
        Generate a cubic force-field from a set of displacement Hessians.

         nc_Hessians holds all 2*3*N Hessians
         Phi_ijk = (Phi_ij^+ - Phi_ij^-)/(2 * Delta q)
        """
        def check(cub1, cub2):
            if np.abs(cub1 - cub2) > 1e-4:
                return np.abs(cub1 - cub2)
            else:
                return False

        nTransRot = self.geometry.nTransRot()
        nVib = self.geometry.nVib()
        disp = 2 * self.anharm_displacement
        cubic = np.zeros((nVib, nVib, nVib), FLOAT)

        for i in range(nVib):
            for j in range(nVib):
                for k in range(nVib):
                    k *= 2
                    pos = nc_Hessians[k][i + nTransRot, j + nTransRot].real
                    neg = nc_Hessians[k + 1][i + nTransRot, j + nTransRot].real
                    cubic[i, j, int(k / 2)] = (pos - neg) / disp
        cubic = self.check_cubic(cubic)
        self.cubic = cubic
        self.has_cubic = True
        return 1

    def calculate_semiquartic_force_field(self, nc_Hessians):
        """
        Generate a semi-quartic force field from a set of displaced Hessians.

        Phi_ijkk = (Phi_ij^+ + Phi_ij^- 2Phi_ij^0)/(q_k)^2
        """
        nTransRot = self.geometry.nTransRot()
        threeN = 3 * len(self.geometry.atoms)
        nVib = threeN - nTransRot
        hessian0 = self.harmonic.hessian
        nc_Hessian_zero = self.transform_to_normal_coordinate_fc(hessian0)
        semiquartic = np.zeros((nVib, nVib, nVib), FLOAT)
        for i in range(nVib):
            for j in range(nVib):
                for k in range(nVib):
                    pos = nc_Hessians[2 * k][i + nTransRot, j + nTransRot]
                    neg = nc_Hessians[2 * k + 1][i + nTransRot, j + nTransRot]
                    hess_zero = nc_Hessian_zero[i + nTransRot, j + nTransRot]
                    semiquartic[i, j, k] = ((pos.real + neg.real -
                                             2 * hess_zero.real) /
                                            (self.anharm_displacement**2))
        self.semiquartic = semiquartic
        self.has_semiquartic = True
        return 1


class VPT2_file(object):
    """
    This class reads in a .vpt2 file resulting from an ORCA VPT2 calculation.
    """
    def __init__(self, vpt2_file_path):
        super(VPT2_file, self).__init__()
        self.file_path = self.check_vpt2_file(vpt2_file_path)

        self.has_geometry = False
        self.has_harmonic = False
        self.has_cubic = False
        self.has_semiquartic = False

        self.geometry = self.get_geometry()
        self.harmonic = self.get_harmonic()
        self.cubic = self.get_cubic()
        self.semiquartic = self.get_semiquartic()

    def check_vpt2_file(self, file_path):
        """Return the VPT2 file as a string."""
        if not os.path.exists(file_path):
            sys.exit("VPT2_file.read_vpt2_file(): "
                     "Could not find vpt2 file.")
        return file_path

    def get_geometry(self):
        """Return the geometry object."""
        with open(self.file_path) as file_object:
            line = file_object.readline()
            while line:
                if "Atomic coordinates in Angstroem" in line:
                    line = file_object.readline()
                    raw_geom = []
                    for i in range(int(line.strip())):
                        split = file_object.readline().split()
                        raw_geom.append([i, split[0], split[2],
                                         split[3], split[4], split[5]])

                    self.has_geometry = True
                    break
                line = file_object.readline()
        if not self.has_geometry:
            sys.exit("VPT2_file.get_geometry(): "
                     "Could not find a valid geometry.")
        return Geometry(raw_geom, use_own_masses=True, distance_units="Angs")

    def get_harmonic(self):
        """Return the harmonic object."""
        with open(self.file_path) as file_object:
            line = file_object.readline()
            while line:
                if "Hessian[i][j] in Eh/(bohr**2)" in line:
                    line = file_object.readline()
                    size = tuple(int(i) for i in line.strip().split())
                    hessian = np.zeros(size, FLOAT)
                    for i in range(np.prod(size)):
                        split = file_object.readline().strip().split()
                        hessian[int(split[0]), int(split[1])] = FLOAT(split[2])
                    self.has_harmonic = True
                    break
                line = file_object.readline()
        if not self.has_harmonic:
            sys.exit("VPT2_file.get_geometry(): "
                     "Could not find a valid Hessian.")
        return Harmonic(self.geometry, hessian=hessian)

    def get_cubic(self):
        """Return the cubic force field as a nxnxn numpy matrix."""
        with open(self.file_path) as file_object:
            line = file_object.readline()
            while line:
                if "Cubic[i][j][k] force field in 1/cm" in line:
                    line = file_object.readline()
                    size = tuple(int(i) for i in line.strip().split())
                    cubic = np.zeros(size, FLOAT)
                    for i in range(np.prod(size)):
                        split = file_object.readline().strip().split()
                        cubic[int(split[0]),
                              int(split[1]),
                              int(split[2])] = FLOAT(split[3])
                    self.has_cubic = True
                    break
                line = file_object.readline()
        if not self.has_cubic:
            sys.exit("VPT2_file.get_geometry(): "
                     "Could not find a valid cubic force field.")
        return cubic

    def get_semiquartic(self):
        """Return the semiquartic force field as a nxnxn numpy matrix."""
        with open(self.file_path) as file_object:
            line = file_object.readline()
            while line:
                if "Semi-quartic[i][j][k][k] force field in 1/cm" in line:
                    line = file_object.readline()
                    size = tuple(int(i) for i in line.strip().split())
                    semiquartic = np.zeros(size, FLOAT)
                    for i in range(np.prod(size)):
                        split = file_object.readline().strip().split()
                        semiquartic[int(split[0]),
                                    int(split[1]),
                                    int(split[2])] = FLOAT(split[3])
                    self.has_semiquartic = True
                    break
                line = file_object.readline()
        if not self.has_semiquartic:
            sys.exit("VPT2_file.get_geometry(): "
                     "Could not find a valid semiquartic force field.")
        return semiquartic


class VPT2(object):
    """
    This handles all calculations related to a cubic/semi-quartic force field.

    This includes anharmonic constants, fundamentals, overtones as well as
     combination bands, VibRot constants and anharmonic properties.
    """

    def __init__(self, harmonic, cubic, semiquartic, **kwargs):
        super(VPT2, self).__init__()
        self.harmonic = harmonic
        self.geometry = harmonic.geometry
        self.kwargs = kwargs

        if type(cubic) == np.ndarray:
            self.cubic = cubic
        else:
            sys.exit("VibRot.VPT2.__init__(): "
                     "A valid cubic force field is necessary")
        if type(semiquartic) == np.ndarray:
            self.semiquartic = semiquartic
        else:
            sys.exit("VibRot.VPT2.__init__(): "
                     "A valid semi-quartic force field is necessary")
        if "print_level" in kwargs:
            self.print_level = kwargs["print_level"]
        else:
            self.print_level = 0
        # Common Variables
        self.nTransRot = harmonic.geometry.nTransRot()
        self.nVib = harmonic.geometry.nVib()
        self.harm_freq = harmonic.freq_inv_cm[self.nTransRot:].real
        self.mat_D = self.harmonic_VPT2_derivative()

    def anharmonic_constants(self):
        """
        Return the anharmonic constants chi as an (3N-nTransRot)**2 tensor.

        Calculated according to
            Papousek/Alijev, 1982, isbn: 9780444997371, 160 pp. and
            Amos/Handy/Jayatilaka (doi:10.1063/1.461259)
        """
        cubic = self.cubic
        semiquartic = self.semiquartic

        fermi_resonances_overview = self.detect_Fermi_resonances(self.mat_D)
        fermi_resonances = [set(f[0]) for f in fermi_resonances_overview]

        def omega(w_k, w_l, w_m):
            # Eq. 6c Amos/Handy/Jayatilaka (doi:10.1063/1.461259)
            omega_klm = (w_m * (w_k**2 + w_l**2 - w_m**2) / (2 *
                         ((w_k + w_l + w_m) *
                          (-w_k + w_l + w_m) *
                          (w_k - w_l + w_m) *
                          (w_k + w_l - w_m))))
            return omega_klm

        def omega_fermi(w_k, w_l, w_m):
            # Eq. 6d Amos/Handy/Jayatilaka (doi:10.1063/1.461259)
            omega_klm = 0.125 * (1 / (w_k + w_l + w_m) +
                                 1 / (-w_k + w_l + w_m) +
                                 1 / (w_k - w_l + w_m))
            return omega_klm

        def check_fermi(i, j, l):
            if set([i, j, l]) in fermi_resonances:
                return True

        # Collect the necessary variables
        nVib = self.nVib
        nTransRot = self.nTransRot
        w = self.harm_freq
        cz = self.harmonic.coriolis_zeta()
        b_e = self.harmonic.rot_const_inv_cm
        chi = np.zeros((nVib, nVib), FLOAT)

        for k in range(nVib):
            for l in range(k, nVib):
                if k == l:
                    # Term2 in Eq. 17.1.2 of Papousek/Alijev
                    chi_t2 = 0.0
                    for m in range(nVib):
                        if check_fermi(k, l, m):
                            if self.print_level:
                                print("Fermi resonance: w_%s~2w_%s" % (k, m))
                            # Eq. 6b Amos/Handy/Jayatilaka
                            chi_t2 += (0.125 * cubic[k, k, m]**2 *
                                       (1 / w[m] + 0.25 / (2 * w[k] + w[m])))
                        else:
                            # Eq. 6a Amos/Handy/Jayatilaka
                            chi_t2 += (cubic[k, k, m]**2 *
                                       (8 * w[k]**2 - 3 * w[m]**2) /
                                       (16 * w[m] * (4 * w[k]**2 - w[m]**2)))
                    # Eq. 17.1.2 of Papousek/Alijev
                    chi[k, k] = semiquartic[k, k, k] / 16 - chi_t2
                else:
                    # Term 1 in Eq. 17.1.3 of Papousek/Alijev
                    chi_t1 = 0.25 * semiquartic[k, k, l]
                    chi_t2 = 0.0
                    chi_t3 = 0.0
                    chi_t4 = 0.0
                    for m in range(nVib):
                        # Term 2 in Eq. 17.1.3 of Papousek/Alijev
                        chi_t2 -= cubic[k, k, m] * cubic[l, l, m] / w[m]
                        if check_fermi(k, l, m):
                            if self.print_level:
                                print("Fermi resonance: "
                                      "w_%s~w_%s+w_%s" % (k, l, m))
                            chi_t3 -= (cubic[k, l, m]**2 *
                                       omega_fermi(w[k], w[l], w[m]))
                        else:
                            # Term 3 in Eq. 17.1.3 of Papousek/Alijev
                            chi_t3 -= (cubic[k, l, m]**2 *
                                       omega(w[k], w[l], w[m]))
                    lz, kz = l + nTransRot, k + nTransRot
                    # Term 4 in Eq. 17.1.3 of Papousek/Alijev
                    for axis in range(3):
                        chi_t4 += (cz[axis, kz, lz]**2 *
                                   (w[k] / w[l] + w[l] / w[k]) *
                                   b_e[axis])
                    chi[k, l] = chi_t1 + chi_t2 / 4 + chi_t3 + chi_t4
                    chi[l, k] = chi[k, l]
        return chi

    def fundamental_transitions(self, chi):
        """Return the fundamental transitions in 1/cm."""
        nVib = self.nVib
        w = self.harm_freq
        fundamental_frequencies = np.zeros((nVib,), dtype=FLOAT)

        # Port this to an Einstein-sum version soon!
        for r in range(nVib):
            tmp = FLOAT(0)
            for s in range(nVib):
                if r != s:
                    tmp += chi[r, s]
            fundamental_frequencies[r] = w[r] + 2 * chi[r, r] + tmp / 2
        return fundamental_frequencies

    def vibRot_constants(self):
        """
        Return an array -alpha_k^beta (minus is important).

        It contains the components of the vibrational-rotational constants in
         1/cm.
        According to eq. 12 of Amos/Handy/Jayatilaka (doi:10.1063/1.461259)
        """
        fermi_resonances_overwiew = self.detect_Fermi_resonances(self.mat_D)
        strong_fermi_resonances = [set(f[0]) for f in fermi_resonances_overwiew
                                   if f[-1] == "strong"]

        def check_fermi(i, j):
            if set([i, j]) in strong_fermi_resonances:
                return True

        # Initialise constants
        h = CONST.planck_constant("J*s")
        c = CONST.speed_of_light()  # m/s
        u_to_kg = CONST.atomic_mass_constant()  # kg
        nVib = self.nVib
        nTransRot = self.nTransRot
        w = self.harm_freq

        cubic = self.cubic
        moI = self.geometry.rot_prop.moment_of_inertia_tensor()  # u*Angs^2
        moI_derivs = self.harmonic.inertia_derivatives()  # u^1/2*Angs
        # The moI derivative needs to be converted to the unit of cm:
        moI_deriv_conv = np.pi * np.sqrt(u_to_kg * c / h) * 1e-9
        cz = self.harmonic.coriolis_zeta()
        b_e = self.harmonic.rot_const_inv_cm

        coriolis_resonances = []
        negAlpha = np.zeros((3, nVib, 4), dtype=FLOAT)
        # Term 1
        for k in range(nVib):
            for b in range(3):
                for a in range(3):
                    negAlpha[b, k, 0] += (1.5 * b_e[b]**2 *
                                          moI_derivs[k, a, b]**2 /
                                          (w[k] * moI[a, a]))
        # Term 2 and 3
        for k in range(nVib):
            for b in range(3):
                for l in range(nVib):
                    lz, kz = l + nTransRot, k + nTransRot
                    if np.abs(w[k] - w[l]) > CORIOLIS_RESONANCE_THRESH:
                        negAlpha[b, k, 1] += (2 * b_e[b]**2 / w[k] *
                                              cz[b, kz, lz]**2 *
                                              (3 * w[k]**2 + w[l]**2) /
                                              (w[k]**2 - w[l]**2))
                    else:
                        coriolis_resonances.append((k, l))
                        negAlpha[b, k, 2] -= (b_e[b]**2 * cz[b, kz, lz]**2 *
                                              (w[k] - w[l])**2 /
                                              ((w[k] + w[l]) * w[k]**2 * w[l]))
        # Term 4
        for k in range(nVib):
            for b in range(3):
                for l in range(nVib):
                    if not check_fermi(l, k):
                        # it seems that this term needs to be negative when
                        # compared to cfour (moI_deriv definition?)
                        # print("{:.9f}".format(moI_derivs[l, b, b]))
                        negAlpha[b, k, 3] -= (2 * b_e[b]**2 * cubic[k, k, l] *
                                              moI_derivs[l, b, b] *
                                              moI_deriv_conv / w[l]**1.5)
        return -negAlpha, coriolis_resonances

    def b_0(self, alpha):
        """Return the corrected B_0 values in 1/cm."""
        b_e = self.harmonic.rot_const_inv_cm
        b_0 = np.zeros((3,), dtype=FLOAT)
        for a in range(3):
            b_temp = 0.0
            for i in range(self.nVib):
                b_temp += np.sum(alpha[a, i])
            b_0[a] = b_e[a] - 0.5 * b_temp
            # print "{:>12,.6f} {:>12,.6f}".format(b_e[a], b_0[a])
        return b_0

    def generate_state(self, ijk_quanta={}):
        """
        Return a list of states of length self.nVib.

        Here, at all positions found in ijk_quanta, the respective amount of
         quanta is inserted, e.g.:
         nVib = 3, ijk_quanta = {1:2, 2:1} --> state = np.array([0, 2, 1]).
        """
        state = np.zeros((self.nVib), dtype=np.int16)
        for i, quanta in ijk_quanta.items():
            state[i] = np.int16(quanta)
        return state

    def recursive_states(self, seed, n_qanta, states):
        """Recursively populate states with n_qanta."""
        if n_qanta == 1:
            return states
        new_states = []
        for element in seed:
            new_states += (element + states).tolist()
        states = np.array(new_states)
        n_qanta -= 1
        return self.recursive_states(seed, n_qanta, states)

    def generate_excited_states(self, initial_state, n_quantas):
        """
        Generate a list of possible excited Vibrational states.

        Here we start from an initial_state (constituting excitations
         of n_quanta).
        """
        eye = np.eye(self.nVib, dtype=np.int)
        seed = []
        pm = np.array([1, -1])
        for i in range(self.nVib):
            for m in pm:
                seed.append((m * eye[i]))
        excited_states = []
        concatenated = np.concatenate([pm * x for x in n_quantas])
        for n_quanta in n_quantas:
            for pre_state in self.recursive_states(seed, n_quanta, seed):
                if np.sum(pre_state) in concatenated:
                    excited_state = initial_state + pre_state
                    if np.min(excited_state) >= 0:
                        if excited_state.tolist() not in excited_states:
                            excited_states.append(excited_state.tolist())
        return excited_states

    def h0vib(self, state_i):
        """
        Return the energy of a harmonic transition.

        I.e. <i|H_0|j>, which is only greater zero if i == j.
        """
        return np.sum((FLOAT(state_i) + 0.5) * self.harm_freq)

    def qn_i(self, n, i, n_quanta):
        """Determine pre-factors resulting from the integrations."""
        if (n == 3 and n_quanta == 1):
            return np.sqrt(9.0 / 8.0 * FLOAT(i + 1)**3)
        elif (n == 2 and n_quanta == 0):
            return FLOAT(i) + 0.5
        elif (n_quanta == n and n_quanta > 0):
            q = [FLOAT(i + j) / 2 for j in range(1, n + 1)]
            return np.sqrt(np.prod(q))
        else:
            return 0.0

    def h1vib(self, state_i, state_j):
        """Return the energy of the 1st anharmonic transition,i.e. <i|H_1|j>."""
        h1 = 0.0
        if (len(state_i) == self.nVib and len(state_j) == self.nVib):
            state_diff = np.abs(state_j - state_i)
            nz = np.nonzero(state_diff)[0]
            # nz:  there could be up to 3 non-zero indices
            if np.sum(state_diff) == 3:
                if len(nz) == 1:
                    gs = min(state_i[nz[0]], state_j[nz[0]])
                    h1 += (self.qn_i(3, gs, 3) *
                           self.cubic[nz[0], nz[0], nz[0]] / 6)
                elif len(nz) == 2:
                    gs = [min(state_i[nz[0]], state_j[nz[0]]),
                          min(state_i[nz[1]], state_j[nz[1]])]
                    if state_diff[nz[0]] == 2:
                        h1 += (self.qn_i(1, gs[1], 1) *
                               self.qn_i(2, gs[0], 2) *
                               self.cubic[nz[0], nz[0], nz[1]] / 2)
                    elif state_diff[nz[1]] == 2:
                        h1 += (self.qn_i(1, gs[0], 1) *
                               self.qn_i(2, gs[1], 2) *
                               self.cubic[nz[1], nz[1], nz[0]] / 2)
                elif len(nz) == 3:
                    gs = [min(state_i[nz[0]], state_j[nz[0]]),
                          min(state_i[nz[1]], state_j[nz[1]]),
                          min(state_i[nz[2]], state_j[nz[2]])]
                    h1 += (self.qn_i(1, gs[0], 1) *
                           self.qn_i(1, gs[1], 1) *
                           self.qn_i(1, gs[2], 1) *
                           self.cubic[nz[0], nz[1], nz[2]])
            elif np.sum(state_diff) == 1:
                # print state_diff, state_i, state_j
                gs = min(state_i[nz[0]], state_j[nz[0]])
                for k in range(self.nVib):
                    # print k, nz[0], gs
                    if k == nz[0]:
                        h1 += (self.qn_i(3, gs, 1) *
                               self.cubic[nz[0], nz[0], nz[0]] / 6)
                    else:
                        h1 += (self.qn_i(1, gs, 1) *
                               self.qn_i(2, state_i[k], 0) *
                               self.cubic[nz[0], k, k] / 2)
            else:
                return 0.0
        else:
            sys.exit("VibRot.vpt2.h1vib(): "
                     "len(state_i) != len(state_j) != nVib.")

        return h1

    def harmonic_VPT2_derivative(self):
        """
        Return the D-matrix.

        This represents the harmonic derivative of the perturbative corrections
         to the fundamental frequencies d (dimensionless) according to
         Matthews.
            doi: 10.1080/00268970902769463 (equation 3,4)
        >> The routine is a bit slow for large systems, check if improvable!
        """
        def kron(a, b):
            if a == b:
                return 1.0
            else:
                return 0.0

        nVib = self.nVib
        nTransRot = self.nTransRot
        w = self.harm_freq
        cz = self.harmonic.coriolis_zeta()[:, nTransRot:, nTransRot:]
        b_e = self.harmonic.rot_const_inv_cm
        d = np.zeros((nVib, nVib), dtype=FLOAT)
        d_0 = np.zeros((nVib), dtype=FLOAT)

        state_0 = self.generate_state({})
        excited_states = self.generate_excited_states(state_0, [1, 3])

        for a in range(nVib):
            # d^0_a Term1
            for b in range(nVib):
                # if a == b, 1 / w[b] - w[b] / w[a]**2 is 0 (no contribution)
                coriolis = 0.0
                for alpha in range(3):
                    coriolis += (cz[alpha, a, b])**2 * b_e[alpha]
                # print("{:>12.4f}".format(coriolis))
                d_0[a] += 0.25 * (1 / w[b] - w[b] / w[a]**2) * coriolis
            # d^0_a Term2
            for excited_state_k in excited_states:
                h1vib_squared = self.h1vib(state_0, excited_state_k)**2
                delta_e_ik = (self.h0vib(state_0) -
                              self.h0vib(excited_state_k))
                d_0[a] += (h1vib_squared / delta_e_ik**2) * excited_state_k[a]
        # print(d_0)
        # sys.exit()
        for i in range(nVib):
            state_i = self.generate_state({i: 1})
            excited_states = self.generate_excited_states(state_i, [1, 3])
            for a in range(nVib):
                # Term 1
                for b in range(nVib):
                    if a != b:
                        coriolis = 0.0
                        for alpha in range(3):
                            coriolis += (cz[alpha, a, b])**2 * b_e[alpha]
                        weighting = (kron(i, a) + 0.5) * (kron(i, b) + 0.5)
                        d[i][a] += (weighting *
                                    (1 / w[b] - w[b] / w[a]**2) * coriolis)
                # Term 2
                for excited_state_k in excited_states:
                    h1vib_squared = self.h1vib(state_i, excited_state_k)**2
                    delta_e_ik = (self.h0vib(state_i) -
                                  self.h0vib(excited_state_k))
                    if not (h1vib_squared == 0.0 or np.abs(delta_e_ik) < 1e-3):
                        d[i][a] -= ((kron(i, a) - excited_state_k[a]) *
                                    h1vib_squared / delta_e_ik ** 2)
                # Term 3
                d[i][a] -= d_0[a]

        return d

    def zero_point_energy(self, anharmonic_constants):
        """Generate the zero point vibrational energy in 1/cm."""
        nVib = self.nVib
        nTransRot = self.nTransRot
        w = self.harm_freq
        cubic = self.cubic
        semiquartic = self.semiquartic
        cz = self.harmonic.coriolis_zeta()
        b_e = self.harmonic.rot_const_inv_cm

        harmonic_zpe = 0.5 * np.sum(w)
        anharmonic_zpe = 0.0
        for i in range(nVib):
            for j in range(nVib):
                if i >= j:
                    anharmonic_zpe += 0.25 * anharmonic_constants[i, j]

        # Term 1
        zpe = -0.25 * np.sum(b_e)

        for k in range(nVib):
            # Term 2
            zpe += semiquartic[k, k, k] / 64.0
            # Term 3
            zpe += -7.0 * cubic[k, k, k]**2 / (576.0 * w[k])
            # Term 4
            for l in range(nVib):
                if l != k:
                    zpe += ((3.0 * w[l] * cubic[k, k, l]**2) /
                            (64.0 * (4.0 * w[k]**2 - w[l]**2)))
            # Term 5
            for l in range(nVib):
                for m in range(nVib):
                    if (k < l and l < m):
                        zpe -= ((cubic[k, l, m]**2 * w[k] * w[l] * w[m]) /
                                (4.0 * ((w[k] + w[l] + w[m]) *
                                        (w[k] - w[l] - w[m]) *
                                        (w[k] + w[l] - w[m]) *
                                        (w[k] - w[l] + w[m]))))
            # Term 6
            for l in range(nVib):
                if k != l:
                    lz, kz = l + nTransRot, k + nTransRot
                    for axis in range(3):
                        zpe -= 0.125 * b_e[axis] * cz[axis, kz, lz]**2

        return zpe + harmonic_zpe + anharmonic_zpe

    def detect_Fermi_resonances(self, mat_D):
        """
        Return Fermi-resonances in an automaitic manner.

        It analyses the harmonic derivative of the perturbative corrections to
         the fundamental frequencies d_mat and retrun resonant states as well as
         their harmonic frequencies.

        Two cases:
            strong: w_i ~ 2 w_j     (D[i,i] = -X, D[i,j] = 2X)
            weak:   w_i ~ w_j + w_k (D[i,i] = -X, D[i,j] =  X, D[i,k] = X)
        """
        cTresh = 1  # np.around threshold

        def d_approx(i, j):
            return np.around(mat_D[i, i], cTresh)

        fermi_resonances = []
        # The following returns a list of indexes for which |D[i,i]| > 0.5
        relevant_D = np.nonzero(np.greater(np.abs(np.diag(mat_D)), 0.5))[0]
        w = self.harm_freq
        for i in relevant_D:
            d_index = np.nonzero(np.greater(np.abs(mat_D[i]), 0.5))[0]
            d_index = [j for j in d_index if i != j]
            if not d_index:
                continue
            if len(d_index) == 1:
                #  strong Fermi resonance
                j = d_index[0]
                f = [[i, j], [w[i], w[j]], [{j: 2}, {i: 1}],
                     [-mat_D[i, i], mat_D[i, j] / 2], "strong"]
                fermi_resonances.append(f)
            elif len(d_index) == 2:
                j, k = d_index[0], d_index[1]
                xTest = -d_approx(i, i)
                if (xTest == d_approx(i, j) and xTest == d_approx(i, k)):
                    f = [[i, j, k], [w[i], w[j], w[k]], [{j: 1, k: 1}, {i: 1}],
                         [-mat_D[i, i], mat_D[i, j], mat_D[i, k]], "weak"]
                    fermi_resonances.append(f)
            else:
                sys.exit("VibRot.vpt2.detect_Fermi_resonances(): "
                         "Matrix D seems too complicated.")
        return fermi_resonances
        # return []

    def effective_hamiltonian(self, anharmonic_const):
        """Return an effective Hamiltonian constructed from Fermi resonances."""
        resonances = self.detect_Fermi_resonances(self.mat_D)
        zpe_anharm = self.zero_point_energy(anharmonic_const)
        for resonance in resonances:
            for raw_state in resonance[2]:
                state = self.generate_state(raw_state)
                h0 = self.h0vib(state)
                print(zpe_anharm, h0, h0 - zpe_anharm)


# class VibRotErrors(Exception):
#     """Base class for exceptions in this module."""
#     def __init__(self, value):
#         self.value = value

#     def __str__(self):
#         return repr(self.value)
