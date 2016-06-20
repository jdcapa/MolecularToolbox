#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""These classes deal with the Orca input and output structure."""
import os
import re
import sys
import numpy as np
# from numpy import linalg
from .geometry import Geometry
# from . import systemtools as ST

TAB = " " * 4
FLOAT = np.float128


class CfourOutput:
    """This class is dealing with the cfour1.0/2.0 output structure."""

    def __init__(self, source_directory, outfile=None):
        """
        Initiate the class with the source_directory name.

        Everything eles should be detected automatically.
        If multiple jobs are present in that directory, a basename needs to be
         specified.
        """
        super(CfourOutput, self).__init__()
        # All we need initially are the directory and the file names.
        self.directory = source_directory
        self.filenames(source_directory, outfile)
        self.basic_info = self.get_BasicInfo()
        self.info = self.get_Calc_info()
        self.geometries = self.get_xyz_geometries()
        self.geometry = self.geometries[-1]
        self.basename = self.geometry.sum_formula()
        # if self.has_gradient:
        #     gradient = self.read_orca_gradient()
        #     if np.any(gradient):
        #         self.gradient = gradient
        if self.has_hessian:
            self.hessian = self.read_hessian()

    def filenames(self, source_directory, outfile):
        """
        Define all the cfour files (min: cfour output file).

        Currently supported:
            OUT or custom outfile
            FCM
        """
        # Output file
        if not outfile:
            self.OUT_FILE = os.path.join(source_directory, "OUT")
        else:
            self.OUT_FILE = os.path.join(source_directory, outfile)
        if not os.path.exists(self.OUT_FILE):
            sys.exit("CfourOutput.filenames(): "
                     "Could not find an Cfour output file.")
        else:
            with open(self.OUT_FILE) as out:
                self.OUT = out.read()
        # Hessian File
        self.HESSIAN_FILE = os.path.join(source_directory, "FCM")
        if os.path.exists(self.HESSIAN_FILE):
            self.has_hessian = True
        else:
            self.has_hessian = False
        if self.has_hessian:
            dipder_file = os.path.join(source_directory, "DIPDER")
            if os.path.exists(dipder_file):
                self.DIPDER_FILE = dipder_file
            else:
                self.DIPDER_FILE = ""

    def get_BasicInfo(self):
        """Read basic calculational details from the output file."""
        infodict = {"level": ["CALCLEVEL", "ICLLVL", ""],
                    "basis": ["BASIS", "IBASIS", ""],
                    "charge": ["CHARGE", "ICHRGE", ""],
                    "mult": ["MULTIPLICTY", "IMULTP", ""],
                    "ref": ["REFERENCE", "IREFNC", ""],
                    "geomet": ["GEO_METHOD", "INR", ""],
                    "scfconv": ["SCF_CONV", "ISCFCV", ""],
                    "geoconv": ["GEO_CONV", "ICONTL", ""],
                    "vib": ["VIBRATION", "IVIB", ""],
                    "anharm": ["ANHARMONIC", "IANHAR", ""],
                    "frozen": ["FROZEN_CORE", "IFROCO", ""],
                    "dropmo": ["DROPMO", "IDRPMO", ""],
                    "coords": ["COORDINATES", "ICOORD", ""]
                    }
        # filling the empty fields
        with open(self.OUT_FILE) as out_file:
            for line in out_file:
                for k in infodict.keys():
                    if re.search("\s+{0}\s+{1}\s+".format(infodict[k][0],
                                                          infodict[k][1]),
                                 line):
                        infodict[k][2] = [x for x in line.strip().split('   ')
                                          if x][2].strip()
                    elif "Job Title" in line:
                        break
        return infodict

    def get_Calc_info(self):
        """Summarise calculation set-up and return it as a dictionary."""
        infodict = {"program": "Cfour",
                    "version": self.get_CfourVersion(),
                    "calctype": self.get_CalcType(),
                    "method": self.get_Method(),
                    "basis": self.get_Basis(),
                    "charge": self.get_Charge(),
                    "mult": self.get_Multiplicity(),
                    "ref": self.get_Reference(),
                    "scfconv": self.get_SCF_Details(),
                    "geoconv": self.get_GeoOptDetails()}
        return infodict

    def get_CfourVersion(self):
        """Obtain Cfour version and revision."""
        re_version = re.compile("ersion ([\d.]+[\w]*)")
        version = "Unknown"
        with open(self.OUT_FILE) as out_file:
            for line in out_file:
                if re_version.search(line):
                    version = re_version.search(line).group(1)
                    break
        return version

    def get_CalcType(self):
        """
        Check the output file for various types of calculations.

        Possible types are:
            single point calculation:           'Energy_Calc',
            geometry optimisation:              'Geo_Opt',
            transition state searc:             'TS_Opt',
            frequency calculation:              'Hessian_Calc',
            anharmonic frequency calculation:   'VPT2_Calc'.
        """
        if self.basic_info['vib'][2] != 'NO':
            if self.basic_info['anharm'][2] == "OFF":
                return 'Hessian_Calc'
            else:
                return 'VPT2_Calc'
        elif self.basic_info['geomet'][2] == "NR":
            return 'Geo_Opt'
        elif self.basic_info['geomet'][2] == "TS":
            return 'TS_Opt'
        elif "SINGLE_POINT" in self.basic_info['geomet'][2]:
            return 'Energy_Calc'
        else:
            sys.exit("CfourOutput.get_CalcType(): "
                     "Type of calculation could not be determined")

    def get_Method(self):
        """
        Check the basic info for the method used in the calculation.

        Supporting: HF, MP2, CCSD, CCSD(T), CCSDT, CCSDT(Q)
        """
        if self.basic_info['level'][2] == 'MBPT(2)':
            return "MP2"
        elif self.basic_info['level'][2] == 'SCF':
            return "HF"
        else:
            return self.basic_info['level'][2]  # That one is simple

    def get_Basis(self):
        """
        Obtain the general Basis set.

        For atom specific definitions we should have a separate routine
        """
        pass

    def get_Reference(self):
        """Obtain the reference determinant type."""
        return self.basic_info["ref"][2]

    def get_Charge(self):
        """Read the charge."""
        return int(self.basic_info["charge"][2])

    def get_Multiplicity(self):
        """Read the multiplicity."""
        return int(self.basic_info["mult"][2])

    def get_SCF_Details(self):
        """Read SCF details from the Orca output file."""
        scfconv = self.basic_info['scfconv'][2]
        return FLOAT(re.sub('D', 'E', ''.join(scfconv.split())))

    def get_GeoOptDetails(self):
        """Read geometry gradient convergence criterion."""
        geoconv = self.basic_info['geoconv'][2]
        return FLOAT(re.sub('D', 'E', ''.join(geoconv.split())))

    def get_Symmetry(self):
        """Read the point group details."""
        re_mPG = re.compile("The full molecular point group is ([\w\s\d]+) .")
        re_cPG = re.compile("The computational point group is ([\w\s\d]+) .")

        point_group = ''
        comp_point_group = ''

        with open(self.OUT_FILE) as out_file:
            for line in out_file:
                if re_mPG.search(line):
                    point_group = re_mPG.search(line).group(1)
                elif re_cPG.search(line):
                    comp_point_group = re_cPG.search(line).group(1)
                    break
        if (point_group and comp_point_group):
            return point_group, comp_point_group
        else:
            print("Could not determine the point group.")
            return None, None

    def get_xyz_geometries(self):
        """
        Obtain the xyz geometries from the orca outfile.

        The coordinates are read in internal units (bohr) and then converted to
         Angstroem and stored in a Geometry object.
        An array of geometries is returned.
        """
        charge = self.get_Charge()
        mult = self.get_Multiplicity()
        geometries = []
        cue = "Z-matrix   Atomic            Coordinates (in bohr)"

        with open(self.OUT_FILE) as cfour_out:
            line = cfour_out.readline()
            while line:
                if cue in line:
                    line = cfour_out.readline()
                    line = cfour_out.readline()
                    line = cfour_out.readline()
                    geometry = []
                    while line.strip():
                        if len(line.split()) < 5:
                            break
                        v = line.split()
                        geometry.append([v[0], v[2], v[3], v[4]])
                        line = cfour_out.readline()
                    geometries.append(Geometry(geometry,
                                               charge=charge, mult=mult,
                                               distance_units='Bohr'))
                line = cfour_out.readline()
        if geometries:
            return geometries
        else:
            sys.exit("CfourOutput.get_xyz_geometries(): "
                     "No geometries found")

    def final_energies(self):
        """Return a list of all final energies."""
        re_final = re.compile("FINAL SINGLE POINT ENERGY\s+([-\d.]+)")
        return np.array([FLOAT(e) for e in re_final.findall(self.OUT)])

    def scf_energies(self):
        """Return a list of all SCF energies."""
        if self.basic_info["ref"][2] == 'ROHF':
            re_scf = re.compile("number:\s+[\d.]+\s+\d+\s+([-\d.]+)\s+[-\d.D]+")
        else:
            re_scf = re.compile(r'E\(SCF\)=\s+([-+\d.]+)')

        return np.array([FLOAT(e) for e in re_scf.findall(self.OUT)])

    def mp2_energies(self):
        """Return a list of all MP2 correlation energies."""
        re_mp2 = re.compile(r'Total MP2 energy\s+=\s+([-\d.]+)\s+a\.u\.')
        return np.array([FLOAT(e) for e in re_mp2.findall(self.OUT)])

    def ccsd_energies(self):
        """Return a list of all CCSD correlation energies."""
        pt1 = r"\d+\s+([-\d.]+)\s+[-\d.]+\s+DIIS\s+[-]+\s+"
        miracle = r"A miracle come to pass. The CC iterations have converged."
        pt2 = r"\s+Non-iterative perturbative"
        re_ccsd = re.compile(pt1 + miracle + pt2)
        # re_ccsd = re.compile(r"CCSD(T) energy\s+([-\d.]+)")
        return np.array([FLOAT(e) for e in re_ccsd.findall(self.OUT)])

    def pT_energies(self):
        """Return a list of all CCSD(T) correlation energies."""
        # re_pT = re.compile("Total perturbative triples energy:\s+([-\d.]+)")
        re_pT = re.compile(r"CCSD\(T\) energy\s+([-\d.]+)")
        e_scf = self.scf_energies()
        e_ccsd = self.ccsd_energies()
        e_ccsd_t = np.array([FLOAT(e) for e in re_pT.findall(self.OUT)])
        pT = []
        for i in range(len(e_scf)):
            if (len(e_ccsd_t) >= i and len(e_ccsd) >= i):
                pT.append(e_ccsd_t[i] - e_ccsd[i] - e_scf[i])
        return np.array(pT)

    def grad_norms(self):
        """Return a list of all gradient norms."""
        re_grd = re.compile('Molecular gradient norm\s+([-\d.E]+)')
        return np.array([FLOAT(e) for e in re_grd.findall(self.OUT)])

    def read_hessian(self, threshold=1e-10):
        """
        Return the cfour Hessian matrix read from the FCM file.

        Units: Eh / (bohr^2)
        The threshold can be used to remove translational inaccuracies, e.g. a
         threshold of 1e-5 gets rid of the remaining translational frequencies
         and brings the rotational remainders down as well.
        It's not quite clear though how it affects VPT2 (--> lower default).
        """
        if not self.has_hessian:
            return

        with open(self.HESSIAN_FILE) as hess_f:
            line = hess_f.readline()
            nAtoms = int(line.split()[0])
            lines = int(3 * nAtoms * nAtoms)
            hessian = np.zeros((3 * nAtoms, 3 * nAtoms), dtype=FLOAT)
            a, b = 0, 0
            for i in range(lines):
                line = hess_f.readline()
                split = [FLOAT(f) for f in line.split()]
                for j in range(3):
                    hessian[b, a] = split[j]
                    a += 1
                if a == 3 * nAtoms:
                    a = 0
                    b += 1
        return hessian

    def read_dipole_derivative(self, cutoff=1e-12):
        """
        Return the Cfour dipole derivative vector (3Nx3) matrix.

        The data is found in the DIPDER file.
        Units: (Eh*bohr)^1/2
        """
        nAtoms = self.geometry.nAtoms
        threeN = nAtoms * 3
        dipder_tmp = np.zeros((threeN, 3), dtype=FLOAT)
        dipder = np.zeros((threeN, 3), dtype=FLOAT)
        re_dd = re.compile(r'[\d.]+' + r'\s+([-\d.]+)' * 3)
        c = 0
        if not self.DIPDER_FILE:
            return None
        with open(self.DIPDER_FILE) as hess_f:
            for line in hess_f:
                if c == threeN:
                    break
                if re_dd.search(line):
                    for i in range(3):
                        der = FLOAT(re_dd.search(line).group(i + 1))
                        if der > cutoff:
                            dipder_tmp[c, i] = der
                    c += 1
        # Cfour sorts the DipDer in a weird way. We have to re-sort
        c = 0
        for i in range(nAtoms):
            for j in range(3):
                dipder[c] = dipder_tmp[i + j * nAtoms]
                c += 1
        return dipder
