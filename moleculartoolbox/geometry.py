# -*- coding: UTF-8 -*-
"""This module contains molecular geometry specific functions."""
from __future__ import print_function
from __future__ import division
import os
import re
import sys
import numpy as np
from atom import Atom
from rotation import Rotation
from chemphysconst import Constants
from numpy import linalg
from operator import itemgetter

FLOAT = np.float128
CONST = Constants()


class Geometry(object):
    """
    This class contains tools for basic geometry operations.

    An input geometry contains a list of strings which represents either
     an xyz- or Z-matrix.
    Note that the internal xyz geometry is always in the unit of Angstroem.
    Conversion tools are provided through the Constants class.
    The Geometry class requires the appended Element class!
    """

    def __init__(self, geometry, **kwarg):
        """
        Initiate with a geometry and some extra arguments.

        kwarg can contain:
        'charge' - type int (total charge of the molecule)
        'mult' | 'multiplicity' - type int (multiplicity of the molecule)
        'geom_type' - type str (type of the geometry provided [xyz|zmat])
        'distance_units' - type str (Bohr or Angstroem)
        'use_own_masses' - type bool (own masses are provided in geometry)
        'pure' - type bool (removes all non-real atoms [i.e. dummys and ECPs])
        """
        super(Geometry, self).__init__()
        self.kwarg = self.check_keywords(kwarg)
        self.set_defaults()
        self.atoms = self.analyse(geometry)
        self.multiplicity = self.check_mult()
        if self.kwarg['pure']:
            self.atoms = self.purify_geometry()
        self.nAtoms = self.nAtoms()
        # self.analyse_comment()
        self.raw_geometry = geometry
        self.rot_prop = Rotation(self)

    def check_keywords(self, kwarg):
        """Check if all necessary keywords are defined."""
        possible_keys = ['charge', 'mult', 'multiplicity', 'geom_type',
                         'distance_units', 'use_own_masses', 'pure']
        for key in possible_keys:
            if key not in kwarg:
                kwarg[key] = ""
        return kwarg

    def set_defaults(self):
        """Analyse the kwarg and determines defaults."""
        if self.kwarg["charge"]:
            self.charge = self.kwarg["charge"]
        else:
            self.charge = 0

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

    def check_mult(self, multiplicity=None):
        """
        Check whether this multiplicity is possible.

        This depends on the charge of the molecule.
        If the multiplicity is not possible, it returns an alternative,
         possible multiplicity.
        """
        def is_even(n):
            return [1, 0][n % 2]

        warn = True
        if not multiplicity:
            if (self.kwarg["mult"] or self.kwarg["multiplicity"]):
                if self.kwarg["multiplicity"]:
                    multiplicity = self.kwarg["multiplicity"]
                else:
                    multiplicity = self.kwarg["mult"]
            else:
                multiplicity = 1
                warn = False

        if multiplicity < 1:
            sys.exit("Geometry.check_mult(): " +
                     "Multiplicity can't be smaller than 1")
        if is_even(multiplicity):
            if not is_even(self.nElectrons()):  # if mult is even,
                return multiplicity          # electron_count must be uneven
            else:
                if warn:
                    print ("Multiplicity of {} not possible, set to {}".format(
                        multiplicity, multiplicity + 1))
                return multiplicity + 1
        else:
            if is_even(self.nElectrons()):  # if mult is uneven,
                return multiplicity      # electron_count must be even
            else:
                if warn:
                    print ("Multiplicity of {} not possible, set to {}".format(
                        multiplicity, multiplicity + 1))
                return multiplicity + 1

    def analyse(self, geometry):
        """
        Analyse the geometry.

        The geometry could be a string (file name) or a list (of lists)
        Also, geometry could represent a raw xyz geometry or a Z-matrix.
        """
        xyz_match = '\s*\w+\s+[-0-9]+\.\d+\s+[-0-9]+\.\d+\s+[-0-9]+\.\d+\s*'
        re_xyz = re.compile(xyz_match)

        if type(geometry) is str:
            # This is interpreted as a file name
            list_of_lines = []
            if not os.path.exists(geometry):
                sys.exit('{0} does not exist.'.format(geometry))
            with open(geometry) as gf:
                list_of_lines = [l.strip() for l in gf]
            new_geometry = self.analyse(list_of_lines)

        elif type(geometry) is list:
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
        """
        Parse an extended geometry where an element has 6 entries.

        This is the raw geometry coming from a yaml parsed input.
        It contains a list of a positional number, the element symbol,
        the mass used for the calculation followed by the xyz coordinates.
        """
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
            new_geometry.append(Atom(line[1],
                                     np.array(temp_xyz, dtype=FLOAT) *
                                     scaling_factor,
                                     mass=mass))
        return new_geometry

    def parse_xyz(self, geometry):
        """
        Parse an xyz file.

        Here geometry is a raw list of lines from the file.
        Returns a list of elements (class)
        """
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
                new_geometry.append(Atom(line[0], np.array(temp_xyz)))
        return new_geometry

    def parse_zmat(self, geometry):
        """
        Parse a zmat file.

        Here geometry is a raw list of lines from the file.
        For the moment, we only get atoms.
        """
        new_geometry = []
        for line in geometry:
            if (re.match('(\w+)\s*', line[:2]) and "="not in line):
                new_geometry.append(
                    Atom(re.match('(\w+)\s*', line[:2]).group(1)))
        return new_geometry

    def set_zmato(self, orca_zmat):
        """
        Add an orca zmat.

        It cannot yet be verified. This is a hack.
        """
        self.zmato = orca_zmat
        self.has_zmato = True

    def zmato_file_string(self):
        """Return a zmato file string."""
        if self.has_zmato:
            zmto_string = ""
            str_format = ("{:<2}  " + " {:<2,d}" * 3 + "\t{:>17,.12f}" +
                          " {:>17,.8f}" * 2 + "\n")
            for line in self.zmato:
                zmto_string += str_format.format(*line)
            return zmto_string

    def purify_geometry(self):
        """Remove all the 'non-real' atoms (Dummy/ECP)."""
        new_atoms = []
        for atom in self.atoms:
            if atom.type is 'Real':
                new_atoms.append(atom)
        return new_atoms

    def isotopic_substitution(self, substitutions):
        """
        Substitute certain elements with a certain isotope.

        The substitutions dictionary contains mappings, e.g. 'C': '13C'
         which causes all the masses of C to be changed to that of 13C
        """
        new_atoms = []
        for atom in self.atoms:
            if atom.element in substitutions:
                new_atoms.append(Atom(substitutions[atom.element],
                                      atom.coordinates))
            else:
                new_atoms.append(atom)
        return new_atoms

    def mass_array(self):
        """Return a list of masses ordered according to the geometry."""
        return np.array([atom.mass for atom in self.atoms], dtype=FLOAT)

    def geom_array(self):
        """Return a 3 x N array of coordinates."""
        return np.array([atom.coordinates for atom in self.atoms], dtype=FLOAT)

    def threeN_geom_array(self):
        """Return a coordinates as a vector of 3N."""
        threeN_array = []
        for atom in self.atoms:
            for coordinate in atom.coordinates:
                threeN_array.append(coordinate)
        return np.array(threeN_array, dtype=FLOAT)

    def threeN_mass_array(self):
        """Return a masses as a vector of 3N."""
        threeN_array = []
        for atom in self.atoms:
            for i in range(3):
                threeN_array.append(atom.mass)
        return np.array(threeN_array, dtype=FLOAT)

    def center_of_mass(self):
        """Return the centre of mass as an array."""
        masses = self.mass_array()
        coords = self.geom_array()
        return coords.T.dot(masses) / np.sum(masses)

    def rotate_to_principal_axes_frame(self):
        """Move and rotate the coordinates into their principle axis frame."""
        x_rot, r_com, coords = self.rot_prop.principal_axes_frame()
        check = 0
        for i, atom in enumerate(self.atoms):
            check += atom.update_coordinates(coords[i])
        # Only return those if we updated all coordinates
        if check == len(coords):
            return x_rot, r_com
        else:
            sys.exit('Geometry.rotate_to_principal_axes_frame(): ' +
                     'Failed to update rotated coordinates.')

    def is_linear(self):
        """Check if the geometry belongs to a linear symmetry group."""
        rot_sym = self.rot_prop.rotational_symmetry()[1]
        if rot_sym == "Linear":
            return True

    def nTransRot(self):
        """Return 5 if the geometry is linear and 6 otherwise."""
        if self.is_linear():
            return 5
        else:
            return 6

    def nVib(self):
        """Return the number of vibrational modes."""
        return 3 * len(self.atoms) - self.nTransRot()

    def xyz_file_string(self, messenge=''):
        """Return an xyz file string."""
        xyz_string = "{0:d}\n{1}\n".format(len(self.atoms), messenge)
        for atom in self.atoms:
            xyz_string += atom.print_cartesian(12) + "\n"
        return xyz_string

    def nAtoms(self):
        """Return the number of atoms."""
        return len(self.atoms)

    def nElectrons(self):
        """Return the electron count of a molecule."""
        electron_count = -1 * self.charge
        for atom in self.atoms:
            electron_count += atom.number_of_electrons
        return electron_count

    def distance(self, i, j):
        """Return the distance between two atoms."""
        return linalg.norm(self.atoms[i].coordinates -
                           self.atoms[j].coordinates)

    def distance_matrix(self):
        """Return an NxN numpy array of distances between atoms."""
        n = self.nAtoms
        distances = np.zeros((n, n), dtype=FLOAT)
        for i in range(n):
            for j in range(n):
                if i != j:
                    distances[i, j] = self.distance(i, j)
        return distances

    def closest_neighbours(self):
        """
        Return a Nx(N-1) int array of the closest neighbours.

        This array is  sorted by distance in ascending order.
        """
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
        """
        Approximate the coordination number of an atom.

        The bonding list allows to specify 'allowable bonds', effectively
         determining the surface atoms of a cluster if bonding contains
         cluster atoms only.
        """
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

    def is_bound(self, i, j, tollerance_factor=1.1):
        u"""
        Return > 0 if i and j are bound.

        Atom numbers i and j are the numbers in positions in the self.atom
        list. Return 0 for non-bonded, 1 for single, 2 for double and 3 for
        triple-bond. This is determined through the use of Pyykkö's covalent
        radii.
        """
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
