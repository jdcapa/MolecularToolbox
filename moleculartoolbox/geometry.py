#!/usr/bin/python

import sys
import re
import numpy as np
from chemphysconst import PeriodicTable
from collections import OrderedDict
from operator import itemgetter
from system_tools import SystemTools
from constants import Constants
from numpy import linalg

FLOAT = np.float128
ST = SystemTools()
CONST = Constants()
PT = PeriodicTable()


class Geometry(object):
    '''
    This class contains tools for basic geometry operations.
    An input geometry contains a list of strings which represents either
     an xyz- or Z-matrix.
    Note that the internal xyz geometry is always in the unit of Angstroem.
    Conversion tools are provided through the Constants class.
    The Geometry class requires the appended Element class!
    '''

    def __init__(self, geometry, **kwarg):
        super(Geometry, self).__init__()
        self.kwarg = self.check_keywords(kwarg)
        self.set_defaults()
        self.atoms = self.analyse(geometry)
        if self.kwarg['pure']:
            self.atoms = self.purify_geometry()
        self.nAtoms = self.nAtoms()
        self.check_multiplicity()
        self.analyse_comment()

        self.raw_geometry = geometry

    def check_keywords(self, kwarg):
        '''
        checks if all necessary keywords are defined, if not it puts an empty
         one in it's stead.
        '''
        possible_keys = ['charge', 'mult', 'multiplicity', 'geom_type',
                         'distance_units', 'use_own_masses', 'pure']
        for key in possible_keys:
            if key not in kwarg:
                kwarg[key] = ''
        return kwarg

    def set_defaults(self):
        '''
        Analyses the kwargs and determines defaults.
        '''
        if self.kwarg["charge"]:
            self.charge = self.kwarg["charge"]
        else:
            self.charge = 0

        if (self.kwarg["multiplicity"] or self.kwarg["multiplicity"]):
            if self.kwarg["multiplicity"]:
                self.multiplicity = self.kwarg["multiplicity"]
            else:
                self.multiplicity = self.kwarg["mult"]
        else:
            self.multiplicity = 1

        if self.kwarg["geom_type"]:
            self.geom_type = self.kwarg["geom_type"]
        else:
            self.geom_type = "xyz"

        if self.kwarg["distance_units"]:
            self.distance_units = self.kwarg["distance_units"]
        else:
            self.distance_units = "Angs"

        if self.kwarg["use_own_masses"]:
            self.use_own_masses = self.kwarg["use_own_masses"]
        else:
            self.use_own_masses = True
        if not self.kwarg["pure"]:  # Default is to disregard Dummies/ECPs
            self.kwarg["pure"] = True
        # other defaults:
        self.comment = ""
        self.has_zmato = False
        self.rms_gradient = 1.0

    def analyse(self, geometry):
        '''
        The geometry string could be a raw xyz geometry or a Z-matrix.
        '''
        xyz_match = '\s*\w+\s+[-0-9]+\.\d+\s+[-0-9]+\.\d+\s+[-0-9]+\.\d+\s*'
        re_xyz = re.compile(xyz_match)

        if type(geometry) is str:
            # This is interpreted as a file name
            new_geometry = self.analyse(ST.file2list(geometry))
        elif type(geometry) is list:
            # print geometry[0], type(geometry[0])
            if type(geometry[0]) is str:
                # This is interpreted as a string list
                if re.match('\d+', geometry[0]):
                    # raw xyz file
                    self.comment = geometry[1]
                    new_geometry = self.analyse([line.split()
                                                 for line in geometry[2:]])
                elif re_xyz.match(geometry[0]):
                    # none-header part of a raw xyz file
                    new_geometry = self.analyse([line.split()
                                                 for line in geometry])
                elif (re.match('\w+', geometry[0]) or
                        len(geometry[0].split()) == 7):
                    # raw ZMat format
                    new_geometry = self.analyse([line.split()
                                                 for line in geometry])
                else:
                    sys.exit('Geometry.analyse(): Unknown geometry type.')
            elif type(geometry[0]) is list:
                # This  is interpreted as an already split-up list
                if len(geometry[0]) == 4:
                    # Standard xyz format [element , x, y, z]
                    new_geometry = self.parse_xyz(geometry)
                elif len(geometry[0]) == 6:
                    # Extended xyz format [index, element, mass , x, y, z]
                    new_geometry = self.parse_extended_xyz(geometry)
                elif len(geometry[0]) in [1, 7]:
                    # ZMat format
                    new_geometry = self.parse_zmat(geometry)
                else:
                    sys.exit('Geometry.analyse(): Unknown geometry type.')
        else:
            sys.exit('Geometry.analyse(): Unknown geometry type.')

        return new_geometry

    def parse_extended_xyz(self, geometry):
        '''
        This is the raw geometry coming from a yaml parsed input.
        It contains a list of a positional number, the element symbol,
        the mass used for the calculation followed by the xyz coordinates.
        '''
        if self.distance_units in ["Bohr", "bohr", "BOHR"]:
            scaling_factor = CONST.bohr_radius("A")
        else:
            scaling_factor = 1.0

        new_geometry = []
        for line in geometry:
            temp_xyz = [0.0, 0.0, 0.0]
            for i in range(3):
                # here we only take the last elements
                temp_xyz[i] = FLOAT(line[i - 3])
            if self.use_own_masses:
                mass = None
            else:
                mass = FLOAT(line[2])
            new_geometry.append(Element(line[1],
                                        np.array(temp_xyz, dtype=FLOAT) *
                                        scaling_factor,
                                        mass=mass))
        return new_geometry

    def purify_geometry(self):
        '''
        Removes all the 'non-real' atoms (Dummy/ECP).
        '''
        new_atoms = []
        for atom in self.atoms:
            if atom.type is 'Real':
                new_atoms.append(atom)
        return new_atoms

    def parse_xyz(self, geometry):
        '''
        Parses an xyz file, where geometry is a raw list of lines from the
         file.
        Returns a list of elements (class)
        '''
        if self.distance_units in ["Bohr", "bohr", "BOHR"]:
            scaling_factor = CONST.bohr_radius("A")
        else:
            scaling_factor = 1.0

        new_geometry = []
        for line in geometry:
            temp_xyz = [0.0, 0.0, 0.0]
            if len(line) == 4:
                for i in range(3):
                    temp_xyz[i] = FLOAT(line[i + 1]) * scaling_factor
                new_geometry.append(Element(line[0], np.array(temp_xyz)))
        return new_geometry

    def parse_zmat(self, geometry):
        '''
        Parses an zmat file,
         where geometry is a raw list of lines from the file.
        For the moment, we only get atoms.
        '''
        new_geometry = []
        for line in geometry:
            if (re.match('(\w+)\s*', line[:2]) and "="not in line):
                new_geometry.append(
                    Element(re.match('(\w+)\s*', line[:2]).group(1)))
        return new_geometry

    def set_zmato(self, orca_zmat):
        '''
        Adds an orca zmat.
        It cannot yet be verified. This is a hack.
        '''
        self.zmato = orca_zmat
        self.has_zmato = True

    def zmato_file_string(self):
        '''
        Returns a zmato file string
        '''
        if self.has_zmato:
            zmto_string = ""
            str_format = ("{:<2}  " + " {:<2,d}" * 3 + "\t{:>17,.12f}" +
                          " {:>17,.8f}" * 2 + "\n")
            for line in self.zmato:
                zmto_string += str_format.format(*line)
            return zmto_string

    def short_info(self):
        '''
        Returns a dictionary with concise information about the geometry.
        '''
        info = OrderedDict()
        info['Sum formula'] = self.sum_formula()
        info['Molecular weight'] = self.molecular_weight()
        info['Geometry type'] = self.geo_type
        info['Charge'] = self.charge
        info['Multiplicity'] = self.multiplicity
        geometry = []
        for atom in self.atoms:
            geometry.append([atom.symbol, atom.coordinates.tolist()])
        info['Coordinates'] = geometry
        return info

    def list_of_atoms(self, abridged_flag=True):
        '''
        Returns a list of all atoms in the geometry
        '''
        list_of_atoms = []
        for atom in self.atoms:
            if abridged_flag:
                if not (atom.element == "Dummy" or atom.element == "ECP"):
                    list_of_atoms.append(atom.symbol)
            else:
                list_of_atoms.append(atom.symbol)
        return list_of_atoms

    def sum_formula(self):
        '''
        Returns a sum formula for a given Geometry.
        '''
        from collections import Counter
        sum_formula = ""

        atoms_dict = Counter(self.list_of_atoms())
        atoms_list = atoms_dict.keys()
        atoms_list.sort()
        atoms_list.sort(key=len)
        for atoms in atoms_list:
            sum_formula += atoms
            if atoms_dict[atoms] > 1:
                sum_formula += str(atoms_dict[atoms])
        return sum_formula

    def molecular_weight(self, unit="g/mol"):
        '''
        Returns the molecular weight in g/mol or kg/mol.
        '''
        molecular_weight = 0.0
        multiplier = 1.0
        if unit == 'kg/mol':
            multiplier = 0.001
        for atom in self.atoms:
            molecular_weight += atom.mass
        return molecular_weight * multiplier

    def get_number_of_electrons(self):
        '''
        Returns the electron count of a molecule.
        '''
        electron_count = -1 * self.charge
        for atom in self.atoms:
            electron_count += atom.number_of_electrons
        return electron_count

    def analyse_comment(self):
        '''
        Checks the xyz file comment for further info
        '''
        re_norm = re.compile('Norm of gradient: ([-+\d.]+)')
        re_mult = re.compile('M\s*=\s*(\d+)')
        re_charge = re.compile('C\s*=\s*([-+\d]+)')
        if re_norm.search(self.comment):
            self.rms_gradient = (float(re_norm.search(self.comment).group(1)) /
                                 (3.0 * len(self.atoms)))
        if re_mult.search(self.comment):
            if (not self.kwarg['mult'] and not self.kwarg['multiplicity']):
                self.multiplicity = int(re_mult.search(self.comment).group(1))
        if re_charge.search(self.comment):
            if not self.kwarg['charge']:
                self.charge = int(re_charge.search(self.comment).group(1))

    def check_multiplicity(self):
        '''
        Checks whether a certain multiplicity is possible for a certain
         molecule and charge.
        If the multiplicity is not possible, it returns an alternative,
         possible multiplicity.
        '''
        def is_even(number):
            '''
            This one checks whether a number is even (True) or uneven (False).
            '''
            if number % 2:
                return False
            else:
                return True

        multiplicity = self.multiplicity
        if multiplicity < 1:
            sys.exit("Multiplicity can't be smaller than 1")
        if is_even(multiplicity):
            if not is_even(self.get_number_of_electrons()):  # if mult is even,
                return multiplicity          # electron_count must be uneven
            else:
                print "Multiplicity of {0} not possible, set to {1}".format(
                    multiplicity, multiplicity + 1)
                return multiplicity + 1
        else:
            if is_even(self.get_number_of_electrons()):  # if mult is uneven,
                return multiplicity      # electron_count must be even
            else:
                print "Multiplicity of {0} not possible, set to {1}".format(
                    multiplicity, multiplicity + 1)
                return multiplicity + 1

    def mass_array(self):
        '''
        Returns a list of masses ordered according to the geometry
        '''
        return np.array([atom.mass for atom in self.atoms], dtype=FLOAT)

    def geom_array(self):
        '''
        Returns a 3 x N array of coordinates.
        '''
        return np.array([atom.coordinates for atom in self.atoms], dtype=FLOAT)

    def threeN_geom_array(self):
        '''
        Returns a coordinates as a vector of 3N.
        '''
        threeN_array = []
        for atom in self.atoms:
            for coordinate in atom.coordinates:
                threeN_array.append(coordinate)
        return np.array(threeN_array, dtype=FLOAT)

    def threeN_mass_array(self):
        '''
        Returns a masses as a vector of 3N.
        '''
        threeN_array = []
        for atom in self.atoms:
            for i in range(3):
                threeN_array.append(atom.mass)
        return np.array(threeN_array, dtype=FLOAT)

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

    def center_of_mass(self):
        '''
        Returns the centre of mass as an array
        '''
        masses = self.mass_array()
        coords = self.geom_array()
        return coords.T.dot(masses) / np.sum(masses)

    def rotate_to_principal_axes_frame(self):
        '''
        Moves/rotates the coordinates into their principle axis frame
            1. shift centre of mass
            2. diagonalise moment_of_inertia_tensor
            3. rotate around resulting X_rot
            4. Update coordinates
        Returns the rotational Matrix X_rot and the centre of mass
        '''
        coords = self.geom_array()
        # 1. shift centre of mass
        r_com = self.center_of_mass()
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

    def is_linear(self):
        '''
        Checks if the geometry belongs to a linear symmetry group.
        '''
        rot_sym = self.rotational_symmetry()[1]
        if rot_sym == "Linear":
            return True

    def nTransRot(self):
        '''
        Returns 5 if the geometry is linear and 6 otherwise.
        '''
        if self.is_linear():
            return 5
        else:
            return 6

    def nVib(self):
        '''
        Returns the number of vibrational modes.
        '''
        return 3 * len(self.atoms) - self.nTransRot()

    def xyz_file_string(self, messenge=''):
        '''
        Returns an xyz file string
        '''
        xyz_string = "{0:d}\n{1}\n".format(len(self.atoms), messenge)
        for atom in self.atoms:
            xyz_string += atom.print_cartesian(12) + "\n"
        return xyz_string

    def nAtoms(self):
        '''
        Returns the number of atoms
        '''
        return len(self.atoms)

    def distance(self, i, j):
        '''
        Returns the distance between two atoms
        '''
        return linalg.norm(self.atoms[i].coordinates -
                           self.atoms[j].coordinates)

    def distance_matrix(self):
        '''
        Returns an NxN numpy array of distances between atoms
        '''
        n = self.nAtoms
        distances = np.zeros((n, n), dtype=FLOAT)
        for i in range(n):
            for j in range(n):
                if i != j:
                    distances[i, j] = self.distance(i, j)
        return distances

    def closest_neighbours(self):
        '''
        Returns a Nx(N-1) int array of the closest neighbours sorted by
         distance in ascending order.
        '''
        n = self.nAtoms
        if n < 2:
            return []
        dist = self.distance_matrix()
        neighbours = []
        for i in range(n):
            tmp_dist = [(j, d) for j, d in enumerate(dist[i]) if d > 0.0]
            tmp_dist.sort(key=itemgetter(1))
            neighbours.append(np.array(tmp_dist)[:, 0].astype(np.int))
        return neighbours

    def coordination_number(self, atomic_number, bonding=[], radius=2.0):
        '''
        Approximates the coordination number of an atom.
        The boning list allows to specify 'allowable bonds', effectively
         determining the surface atoms of a cluster if boning contains
         cluster atoms only.
        '''
        coordination_number = 0
        closest_neighbours = self.closest_neighbours()[atomic_number]
        if bonding:
            for i in closest_neighbours:
                if self.atoms[i].symbol in bonding:
                    if self.is_bound(atomic_number, i):
                        coordination_number += 1
        else:
            for i in closest_neighbours:
                if self.is_bound(atomic_number, i):
                    coordination_number += 1
        return coordination_number

    def is_bound(self, i, j):
        '''
        Return > 0 if i and j are bound.

        Atom numbers i and j are the numbers in positions in the self.atom
        list. Return 0 for non-bonded, 1 for single, 2 for double and 3 for
        triple-bond. This is determined through the use of Pyykkö's covalent
        radii.
        '''
        tollerance_factor = 1.1  # must be > 1
        maximum_covalent = 2.32 * 2 * tollerance_factor  # 2.32 is Caesium
        distance = self.distance(i, j)
        if distance > maximum_covalent:
            return 0
        # Let's first get the covalent radii for each element
        cr_i = self.atoms[i].covalent_radii()
        cr_j = self.atoms[j].covalent_radii()
        max_bond = min(len(cr_i), len(cr_j))
        cov_bonds = [cr_i[b] + cr_j[b] for b in range(max_bond)]
        for i, cov_bond in enumerate(cov_bonds):
            if (distance <= tollerance_factor * cov_bond and
                    distance >= (2 - tollerance_factor) * cov_bond):
                return i + 1
        if distance < (2 - tollerance_factor) * cov_bonds[-1]:
            # The distance is smaller than the tightest possible covalent bond.
            note = "Atom {}({}) is very close to atom {}({}). Check!"
            note = note.format(self.atoms[i].symbol, i + 1,
                               self.atoms[j].symbol, j + 1)
            print(note)

        return 0


class Element(object):
    """
    Contains the basic properties of an Element and its xyz position.
    """
    def __init__(self, element, coordinates=[0.0, 0.0, 0.0], **kwargs):
        super(Element, self).__init__()
        self.Const = Constants()
        self.setter(element)
        if "mass" in kwargs:
            self.mass = kwargs["mass"]
        if "pure" in kwargs:
            if kwargs["pure"]:
                self.purify_geometry()

        self.analyse_coordinates(coordinates)

    def setter(self, element):
        '''
        Analyses the Element in order to set up its properties
        '''
        self.symbol = element.title()
        self.basis = ''
        self.pt_entry = None
        self.mass = 0.0
        self.number_of_electrons = 0
        if self.symbol == "X":  # Dummy Atom
            self.element = "X"
            self.type = "Dummy"
        elif ">" in self.symbol:  # ECP
            self.element = self.symbol
            self.type = "ECP"
        else:  # Normal Atom
            self.pt_entry = PT.element(self.symbol)
            self.type = "Real"
            self.element = self.pt_entry.symbol
            self.mass = self.pt_entry.mass
            self.number_of_electrons = self.pt_entry.number

    def analyse_coordinates(self, coordinates=[0.0, 0.0, 0.0]):
        '''
        Analyses the coordinates and sets up the position.
        Sets self.coordinates as a numpy array of the xyz position.
        Sets self.additional_properties to whatever follows the xyz'z.
        '''
        other = []
        floats = [float, np.float64, FLOAT]

        def error():
            # print "Coordinates:\n", len(coordinates)
            sys.exit("Cannot parse the provided coordinates\n" +
                     "(Class Element, analyse_coordinates module)")

        if type(coordinates) is list:
            if (len(coordinates) < 3):
                error()
            if len(coordinates) == 3:
                if not type(coordinates[0]) in floats:
                    error()
            elif (len(coordinates) > 3 and type(coordinates[0]) is str):
                if len(coordinates) > 4:
                    other = coordinates[5:len(coordinates)]
                coordinates = coordinates[1:4]
            elif (len(coordinates) > 3 and type(coordinates[0]) in floats):
                other = coordinates[4:len(coordinates)]
                coordinates = coordinates[:3]
            else:
                error()
            self.coordinates = np.array(coordinates)
            self.additional_properties = other

        elif type(coordinates) is np.ndarray:
            self.coordinates = coordinates
            self.additional_properties = other
        else:
            error()
        return True

    def basis_from_general_basis(self, cardinal_number, core='', aug='a',
                                 other=''):
        '''
        Converts a general Dunning basis name like aVTZ to the element specific
         correlation consistent basis name (H:VTZ, S:AVT+dZ)
        '''
        extra_d = ''
        if (aug == 'a' and self.element.number == 1):
            aug = ''
        elif (aug == 'a' and self.element.number > 1):
            aug = 'A'
        if self.element.number >= 12:
            extra_d = '+d'
        elif self.element.number >= 18:
            sys.exit('More than 2nd row not yet implemented')
        if (extra_d and other):
            extra_d = ''
        if (extra_d and core):
            extra_d = ''
        if (self.element.number == 1 and core):
            core = ''
        basis = self.symbol.upper() + ":" + \
            aug + core + "V" + cardinal_number + extra_d + "Z" + other

        return basis

    def covalent_radii(self):
        """
        Return Pyykkö's covalent radii as a list.

        The list can contain one entry for elements that exhibit only single
         bonds (e.g. Hydrogen), two for elements that can be double bound
         (e.g. Oxygen) or elements that can form triple bonds.
        """
        raw_radii = [self.pt_entry.covalent_r_single,
                     self.pt_entry.covalent_r_double,
                     self.pt_entry.covalent_r_triple]
        radii = []
        for radius in raw_radii:
            if radius:
                # The radii are in pm (type INT)
                radii.append(FLOAT(radius) / 100.0)
            else:
                if radii:
                    return radii
                else:
                    error_msg = ("Element.covalent_radii(): " +
                                 "Could not find any covalent radius for " +
                                 "element {} .").format(self.symbol)
                    sys.exit(error_msg)
        return radii

    def convert_to_long_basis_name(self):
        '''
        Converts a Dunning short-hand notation into a long hand one:
            AVTZ  --> aug-cc-pVTZ
            wCVQZ --> cc-pwCQZ
        Non-Duninng bases are left untouched.
        '''
        re_Dunning = re.compile('([A]*)([wC]*)V([+dDTQ56]+)Z([-DK]*)')
        find = re_Dunning.search(self.basis)
        if find:
            if find.group(1):
                long_basis = 'aug-cc-p'
            else:
                long_basis = 'cc-p'
            if find.group(2):
                long_basis += find.group(2)
            long_basis += 'V'
            if find.group(3):
                if '+' in find.group(3):
                    long_basis += '(' + find.group(3) + ')'
                else:
                    long_basis += find.group(3)
            long_basis += 'V'
            if find.group(4):
                long_basis += '-DK'
            return long_basis
        else:
            return self.basis

    def set_basis(self, basis_string):
        '''
        Sets the simple basis name
        '''
        if not self.basis:
            self.basis = basis_string

    def check_if_basis_eists_in_GENBAS(self, genbas_path):
        '''
        Checks if self.basis exists in the provided GENBAS file
        '''
        if self.element == 'X':
            return True
        if self.basis:
            re_hit = re.compile("{}:{}".format(self.element, self.basis))
            with open(genbas_path) as genbas:
                for line in genbas:
                    if re_hit.search(line):
                        return True
            return False
        else:
            sys.exit('No basis given.')

    def update_coordinates(self, new_coordinates):
        '''
        Allows for a manual definition of the coordinate, where
         new_coordinates must be a numpy array with a length of 3.
        '''
        if (len(new_coordinates) == 3 and
                type(new_coordinates).__module__ == np.__name__):
            self.coordinates = new_coordinates
            return 1
        else:
            print "Could not update coordinates."
            return 0

    def print_cartesian(self, precision=9, unit="Angstroem", tab=''):
        '''
        Returns a string of "self.element x y z"
        '''
        if unit in ["A", "Angs", "angs", "Angstroem", "angstroem"]:
            len_fac = 1.0
        else:
            len_fac = 1.0 / self.Const.bohr_radius("Angstroem")

        vec_str = "  {{:> {},.{}f}}".format(5 + precision, precision) * 3
        vec = np.around((self.coordinates * len_fac), decimals=precision + 1)
        vec += 0
        vec_str = vec_str.format(*vec)
        return tab + self.symbol + vec_str
