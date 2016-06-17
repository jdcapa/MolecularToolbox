# -*- coding: UTF-8 -*-
"""This module contains all the atom specific functions."""
import sys
import re
import numpy as np
from chemphysconst import PeriodicTable
from chemphysconst import Constants

FLOAT = np.float128
CONST = Constants()
PT = PeriodicTable()


class Atom(object):
    """Contains the basic properties of an Atom and its xyz position."""

    def __init__(self, symbol, coordinates=[0.0, 0.0, 0.0], **kwargs):
        """Initiat the atom with at least a symble and some coordinates."""
        super(Atom, self).__init__()

        self.setter(symbol, kwargs)
        self.analyse_coordinates(coordinates)

    def setter(self, symbol, kwargs):
        """Analyse the Element in order to set up its properties."""
        symbol = symbol.title()
        self.pt_entry = None
        self.mass = 0.0
        self.number_of_electrons = 0
        re_isotope = re.compile("(\d+)\^*(\w+)")
        if symbol == "X":  # Dummy Atom
            self.element = "X"
            self.type = "Dummy"
        elif ">" in symbol:  # ECP
            self.element = symbol
            self.type = "ECP"
        elif re_isotope.match(symbol):  # Isotope
            element = re_isotope.search(symbol).group(2)
            isotope = int(re_isotope.search(symbol).group(1))
            self.setter(element, kwargs)
            self.mass = self.pt_entry.isotope(isotope).atomic_mass
        elif symbol == "D":  # Special case of Deuterium
            self.setter("H", kwargs)
            self.mass = self.pt_entry.isotope(2).atomic_mass
        else:  # Normal Atom
            import time
            t0 = time.time()
            self.pt_entry = PT.element(symbol)
            t1 = time.time()
            self.type = "Real"
            self.element = self.pt_entry.symbol
            self.mass = self.pt_entry.mass
            self.number_of_electrons = self.pt_entry.number
            t2 = time.time()
            self.timing = np.array([t1 - t0, t2 - t1])
        # Keyword analysis
        floats = [float, np.float64, np.float128]
        if ("mass" in kwargs and type(kwargs["mass"]) in floats):
            # The keyword mass overwrites all other mass.
            self.mass = kwargs["mass"]

    def analyse_coordinates(self, coordinates=[0.0, 0.0, 0.0]):
            """
            Analyse the coordinates and sets up the position.

            Sets self.coordinates as a numpy array of the xyz position.
            Sets self.additional_properties to whatever follows the xyz's.
            """
            other = []
            floats = [float, np.float64, np.float128]

            def error():
                sys.exit("Atom.analyse_coordinates(): " +
                         "Cannot parse the provided coordinates.")

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
                radii.append(FLOAT(radius) / 100)
            else:
                if radii:
                    return radii
                else:
                    error_msg = ("Atom.covalent_radii(): " +
                                 "Could not find any covalent radius for " +
                                 "element {} .").format(self.element)
                    sys.exit(error_msg)
        return radii

    def update_coordinates(self, new_coordinates):
        """
        Allow for a manual definition of the coordinate.

        Here new_coordinates must be a numpy array with a length of 3.
        """
        if (len(new_coordinates) == 3 and
                type(new_coordinates).__module__ == np.__name__):
            self.coordinates = new_coordinates
            return 1
        else:
            print ("Could not update coordinates.")
            return 0

    def print_cartesian(self, precision=9, unit="Angstroem", tab=''):
        """Return a string of self.element x y z."""
        if unit in ["A", "Angs", "angs", "Angstroem", "angstroem"]:
            len_fac = 1
        else:
            len_fac = 1 / CONST.bohr_radius("Angstroem")

        vec_str = "  {{:> {},.{}f}}".format(5 + precision, precision) * 3
        vec = np.around((self.coordinates * len_fac), decimals=precision + 1)
        vec += 0
        vec_str = vec_str.format(*vec)
        return tab + self.element + vec_str
