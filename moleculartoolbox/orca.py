#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""These classes deal with the Orca input and output structure."""
import os
import re
import sys
import numpy as np
from numpy import linalg
from .geometry import Geometry
# from . import systemtools as ST

TAB = " " * 4
FLOAT = np.float128


class OrcaOutput(object):
    """This class is dealing with the orca 4.0 output structure."""

    def __init__(self, source_directory, basename=None, input_only=False):
        """
        Initiate the class with the source_directory name.

        Everything else should be detected automatically.
        If multiple jobs are present in that directory, a basename needs to be
         specified.
        """
        super(OrcaOutput, self).__init__()
        # All we need initially are the directory and the file names.
        self.directory = source_directory
        self.filenames(source_directory, basename, input_only)
        if input_only:
            self.geometry = self.get_xyz_geometry_from_input()
        else:
            self.info = self.get_Calc_info()
            self.geometries = self.get_xyz_geometries()
            self.geometry = self.geometries[-1]
            if self.has_gradient:
                gradient = self.read_orca_gradient()
                if np.any(gradient):
                    self.gradient = gradient
            if self.has_hessian:
                self.hessian = self.read_hessian()

    def filenames(self, source_directory, basename, input_only):
        """
        Define all the orca files (min: input and output file).

        Currently supported:
            .inp
            .*.out
            .engrad
            .hess
        """
        # 1. Input file
        if not basename:
            # If basename is not provided, try to find one using an input file.
            occurances = 0
            for element in os.listdir(source_directory):
                if element[-4:] == ".inp":
                    occurances += 1
                    basename = element[:-4]
            if occurances != 1:
                sys.exit("OrcaOutput.filenames(): "
                         "Orca basename not provided")
        else:
            input_file = os.path.join(source_directory, basename + '.inp')
            if os.path.exists(input_file):
                self.INPUT_FILE = input_file
            else:
                sys.exit("OrcaOutput.filenames(): "
                         "Could not find an Orca input file.")
        self.basename = basename
        # we want to terminate here if we only have an input file
        if input_only:
            return
        # Output file
        out_files = [f for f in os.listdir(source_directory) if '.out' in f]
        out_files_c = out_files[:]
        re_out_file = re.compile(basename + '[\w\d.]*' + '.out')
        for out_file in out_files_c:
            if not re_out_file.match(out_file):
                out_files.remove(out_file)
        if (not out_files or len(out_files) > 1):
            sys.exit("OrcaOutput.filenames(): "
                     "Could not find a unique Orca output file.")
        else:
            self.OUT_FILE = os.path.join(source_directory, out_files[0])
            # if (ST.get_memory(0.5, "k") >
            #         os.path.getsize(self.OUT_FILE)):
            with open(self.OUT_FILE) as out:
                self.OUT = out.read()
            # else:
            #     sys.exit("OrcaOutput.filenames(): "
            #              "{} is {} GB (too large)".format(
            #                  os.path.basename(self.OUT_FILE),
            #                  os.path.getsize(self.OUT_FILE)))
        # Gradient file
        gradient_file = os.path.join(source_directory, basename + '.engrad')
        if os.path.exists(gradient_file):
            self.GRADIENT_FILE = gradient_file
            self.has_gradient = True
        else:
            self.GRADIENT_FILE = ""
            self.has_gradient = False

        # Hessian file
        hessian_file = os.path.join(source_directory, basename + '.hess')
        if os.path.exists(hessian_file):
            self.HESSIAN_FILE = hessian_file
            self.has_hessian = True
        else:
            self.HESSIAN_FILE = ""
            self.has_hessian = False

    def get_xyz_geometry_from_input(self):
        """Obtain the xyz geometry from the Orca input file and return it."""
        re_xyzHeader = re.compile("xyz\s+([-+\d]+)\s+([\d]+)")
        re_xyz = re.compile("(\w+)" + "\s+([-.\d]+)" * 3)
        geometry = []
        has_xyz = False
        distance_units = "Angs"

        with open(self.INPUT_FILE) as inp:
            for line in inp:
                if re.search("Bohr", line, re.IGNORECASE):
                    distance_units = "Bohr"
                    continue
                if re_xyzHeader.search(line):
                    has_xyz = True
                    charge = int(re_xyzHeader.search(line).group(1))
                    mult = int(re_xyzHeader.search(line).group(2))
                    continue
                if has_xyz:
                    if re_xyz.search(line):
                        geometry.append(list(re_xyz.search(line).groups()))
                        continue
            return Geometry(geometry,
                            charge=charge,
                            mult=mult,
                            distance_units=distance_units)

    def get_Calc_info(self):
        """Summarise calculation set-up and return it as a dictionary."""
        infodict = {"program": "Orca",
                    "version": self.get_OrcaVersion(),
                    "basename": self.basename,
                    "calctype": self.get_CalcType(),
                    "method": self.get_Method(),
                    "basis": self.get_Basis(),
                    "charge": self.get_Charge(),
                    "mult": self.get_Multiplicity(),
                    "ref": self.get_Reference(),
                    "scfconv": self.get_SCF_Details()[1],
                    "geoconv": self.get_GeoOptDetails()[0],
                    }
        return infodict

    def get_OrcaVersion(self):
        """Obtain Orca version and revision."""
        re_version = re.compile("Program Version ([\d.]+)")
        re_svn = re.compile("SVN: \$(Rev: \d+)\$")
        version = ""
        with open(self.OUT_FILE) as out_file:
            for line in out_file:
                if re_version.search(line):
                    version += re_version.search(line).group(1)
                    continue
                if re_svn.search(line):
                    version += " " + re_svn.search(line).group(1)
                if "With contributions from" in line:
                    break
        return version

    def get_CalcType(self):
        """
        Check the output file for various types of calculations.

        Possible types are:
            single point calculation:   'Energy_Calc',
            geometry optimisation:      'Geo_Opt',
            gradient calculation:       'Gradient_Calc',
            frequency calculation:      'Hessian_Calc'.
        """
        with open(self.OUT_FILE) as out_file:
            sp_flag = False
            for line in out_file:
                if '* Geometry Optimization Run *' in line:
                    return 'Geo_Opt'
                elif '* Single Point Calculation *' in line:
                    sp_flag = True
                    continue
                elif "* Energy+Gradient Calculation *" in line:
                    grad_flag = True
                elif "HESSIAN" in line:
                    if sp_flag:
                        return 'Hessian_Calc'
                    if grad_flag:
                        return 'Grad+Hessian_Calc'
        if sp_flag:
            return 'Energy_Calc'
        elif grad_flag:
            return 'Gradient_Calc'
        else:
            sys.exit("OrcaOutput.get_CalcType(): "
                     "Type of calculation could not be determined")

    def get_Method(self):
        """Check the output file for the method used in the calculation."""
        # RegEx definitions:
        re_DFT = re.compile("The ([\d\w]+) functional is recognized")
        re_CCSD = re.compile("Correlation treatment\s+[.]+\s+CCSD")
        re_pT = re.compile("TRIPLES CORRECTION")
        re_MP2 = re.compile("ORCA  MP2 ")
        re_DISP = re.compile("DFT(D\d)")
        # Flags
        bDFT_flag = False
        bCCSD_flag = False

        # Let's go digging
        with open(self.OUT_FILE) as out_file:
            for line in out_file:
                # Test for DFT:
                if re_DFT.search(line):
                    bDFT_flag = True
                    dft_functional = re_DFT.search(line).group(1)
                if (bDFT_flag and re_DISP.search(line)):
                    return re_DISP.search(line).group(1) + '-' + dft_functional
                if re_MP2.search(line):
                    return "MP2"
                if re_CCSD.search(line):
                    bCCSD_flag = True
                if (bCCSD_flag and re_pT.search(line)):
                    return "CCSD(T)"
        if bCCSD_flag:
            return "CCSD"
        else:
            return "HF"

    def get_Basis(self):
        """
        Obtain the general Basis set.

        For atom specific definitions we should have a separate routine
        """
        # Here we can afford to open the file object, since we break early.
        with open(self.OUT_FILE) as out_file:
            for line in out_file:
                if "Your calculation utilizes the basis:" in line:
                    return line.strip().split(":")[-1].strip()

    def get_Reference(self):
        """Obtain the reference determinant type."""
        re_HF = re.compile("Hartree-Fock type\s+HFTyp\s+[.]+\s*(\w+)")
        # Here we can afford to open the file object, since we break early.
        with open(self.OUT_FILE) as out_file:
            for line in out_file:
                if re_HF.search(line):
                    return re_HF.search(line).group(1)

    def get_Charge(self):
        """Read the charge."""
        re_Charge = re.compile("Total Charge\s+Charge\s+[.]+\s*([-+\d]+)")
        # Here we can afford to open the file object, since we break early.
        with open(self.OUT_FILE) as out_file:
            for line in out_file:
                if re_Charge.search(line):
                    return int(re_Charge.search(line).group(1))

    def get_Multiplicity(self):
        """Read the multiplicity."""
        re_Mult = re.compile("Multiplicity\s+Mult\s+[.]+\s*(\d+)")
        # Here we can afford to open the file object, since we break early.
        with open(self.OUT_FILE) as out_file:
            for line in out_file:
                if re_Mult.search(line):
                    return int(re_Mult.search(line).group(1))

    def get_SCF_Details(self):
        """Read SCF details from the Orca output file."""
        re_SCF_Thresh = re.compile("Thresh\s+[.]+\s*([\d.e+-]+) Eh")
        re_SCF_DeltaE = re.compile("TolE\s+[.]+\s*([\d.e+-]+) Eh")
        scf_Thresh = 0.0
        scf_DeltaE = 0.0
        # Here we can afford to open the file object, since we break early.
        with open(self.OUT_FILE) as out_file:
            for line in out_file:
                if re_SCF_Thresh.search(line):
                    scf_Thresh = FLOAT(re_SCF_Thresh.search(line).group(1))
                if re_SCF_DeltaE.search(line):
                    scf_DeltaE = FLOAT(re_SCF_DeltaE.search(line).group(1))
                if (scf_Thresh and scf_DeltaE):
                    return scf_Thresh, scf_DeltaE

    def get_GeoOptDetails(self):
        """Read geometry optimisation details from the Orca output file."""
        if not self.get_CalcType() == "Geo_Opt":
            return "", ""

        re_GEO_DeltaE = re.compile("TolE\s+[.]+\s*([\d.e+-]+) Eh")
        re_GEO_RMSG = re.compile("TolRMSG\s+[.]+\s*([\d.e+-]+) Eh/bohr")
        geo_DeltaE = 0.0
        geo_RMSG = 0.0
        # Here we can afford to open the file object, since we break early.
        with open(self.OUT_FILE) as out_file:
            for line in out_file:
                if re_GEO_DeltaE.search(line):
                    geo_DeltaE = FLOAT(re_GEO_DeltaE.search(line).group(1))
                if re_GEO_RMSG.search(line):
                    geo_RMSG = FLOAT(re_GEO_RMSG.search(line).group(1))
                if (geo_DeltaE and geo_RMSG):
                    return geo_DeltaE, geo_RMSG

    def get_numberOfAtoms(self):
        """Read the number of Atoms from the orca output file."""
        # re_nAtoms = re.compile("Number of atoms\s+[.]+\s+(\d+)")
        # Here we can afford to open the file object, since we break early.
        # with open(self.OUT_FILE) as out_file:
        #     for line in out_file:
        #         if re_nAtoms.search(line):
        #             return int(re_nAtoms.search(line).group(1))
        nAtoms = -2
        count_flag = False
        with open(self.OUT_FILE) as out_file:
            for line in out_file:
                if "CARTESIAN COORDINATES (A.U.)" in line:
                    count_flag = True
                    continue
                if (count_flag and line.strip()):
                    nAtoms += 1
                    continue
                elif (count_flag and not line.strip()):
                    return nAtoms

    def final_energies(self):
        """Return a list of all final energies."""
        re_final = re.compile("FINAL SINGLE POINT ENERGY\s+([-\d.]+)")
        return np.array([FLOAT(e) for e in re_final.findall(self.OUT)])

    def scf_energies(self):
        """Return a list of all SCF energies."""
        re_scf = re.compile('Total Energy\s+:\s+([-\d.]+)\s+Eh\s+[-\d.]+\s+eV')
        return np.array([FLOAT(e) for e in re_scf.findall(self.OUT)])

    def mp2_energies(self):
        """Return a list of all MP2 correlation energies."""
        re_mp2 = re.compile('MP2 CORRELATION ENERGY\s+:\s+([-\d.]+) Eh')
        return np.array([FLOAT(e) for e in re_mp2.findall(self.OUT)])

    def ccsd_energies(self):
        """Return a list of all CCSD correlation energies."""
        re_ccsd = re.compile('E\(CORR\)\s+[.]+\s+([-\d.]+)')
        return np.array([FLOAT(e) for e in re_ccsd.findall(self.OUT)])

    def pT_energies(self):
        """Return a list of all CCSD(T) correlation energies."""
        re_pT = re.compile('Triples Correction \(T\)\s+[.]+\s+([-\d.]+)')
        return np.array([FLOAT(e) for e in re_pT.findall(self.OUT)])

    def grad_norms(self):
        """Return a list of all gradient norms."""
        re_grd = re.compile('Norm of the cartesian gradient\s+[.]+\s+([\d.]+)')
        if self.info["method"] == "MP2":
            re_grd = re.compile('NORM OF THE MP2 GRADIENT:\s+([\d.]+)')
        return np.array([FLOAT(e) for e in re_grd.findall(self.OUT)])

    def get_xyz_geometries(self):
        """
        Obtain the xyz geometries from the orca outfile.

        The coordinates are read in internal units (bohr) and then converted to
         Angstroem and stored in a Geometry object.
        An array of geometries is returned.
        """
        geometries = []
        nAtoms = self.get_numberOfAtoms()
        charge = self.get_Charge()
        mult = self.get_Multiplicity()

        search_str = "CARTESIAN COORDINATES (A.U.)"
        out = self.OUT.split('\n')
        lFind = [i + 3 for i, l in enumerate(out) if search_str in l]
        raw_coords = [out[i: i + nAtoms] for i in lFind]

        for raw_coord in raw_coords:
            geometry = []
            for line in raw_coord:
                if (line.strip() and len(line.split()) == 8):
                    v = line.split()
                    geometry.append([v[0], v[1], v[4], v[5], v[6], v[7]])
            geometries.append(Geometry(geometry,
                                       charge=charge,
                                       mult=mult,
                                       distance_units='Bohr'))

        if geometries:
            return geometries
        else:
            sys.exit("OrcaOutput.get_xyz_geometries(): "
                     "No geometries found")

    def read_orca_gradient(self):
        """Obtain the Cartesian gradients (Eh/bohr)."""
        if self.has_gradient:
            gradient = []
            no_of_atoms = 0
            with open(self.GRADIENT_FILE) as gradient_file:
                line = gradient_file.readline()
                while line:
                    if "Number of atoms" in line:
                        gradient_file.readline()
                        no_of_atoms = int(gradient_file.readline().strip())
                    elif "The current gradient in Eh/bohr" in line:
                        gradient_file.readline()
                        while line.strip() != "#":
                            tmp_grad = []
                            for i in range(3):
                                line = gradient_file.readline()
                                tmp_grad.append(FLOAT(line.strip()))
                            gradient.append(tmp_grad)
                            if len(gradient) == no_of_atoms:
                                break
                    line = gradient_file.readline()
            if (gradient and len(gradient) == no_of_atoms):
                return np.array(gradient)
            else:
                sys.exit("OrcaOutput.read_orca_gradient(): "
                         "Gradient could not be retrieved")

    def norm_of_grad(self):
        """Return the norm of the gradient."""
        if np.any(self.gradient):
            return linalg.norm(self.gradient.astype(np.float64))
        else:
            return 0

    def rms_of_grad(self):
        """Return the norm of the gradient."""
        if np.any(self.gradient):
            return self.norm_of_grad() / (3 * self.geometry.nAtoms)
        else:
            return 0.0

    def read_hessian(self, threshold=1e-10):
        """
        Return the Oca Hessian matrix read from hess_file.

        Units: Eh / (bohr^2)
        The threshold can be used to remove translational inaccuracies, e.g. a
         threshold of 1e-5 gets rid of the remaining translational frequencies
         and brings the rotational remainders down as well.
        It's not quite clear though how it affects VPT2 (--> lower default).
        """
        if not self.has_hessian:
            return

        no_colums = 5.0
        with open(self.HESSIAN_FILE) as hess_f:
            line = hess_f.readline()
            while line:
                line = hess_f.readline()
                if line.strip() == "$hessian":
                    line = hess_f.readline()
                    threeN = int(line.strip())
                    hessian = np.zeros((threeN, threeN), dtype=FLOAT)
                    col_sets = range(int(np.ceil(float(threeN) / no_colums)))
                    for col_set in col_sets:
                        line = hess_f.readline()
                        j_s = [int(j) for j in line.split()]
                        for i in range(threeN):
                            line = hess_f.readline()
                            values = [FLOAT(v) for v in line.split()[1:]]
                            for k, value in enumerate(values):
                                if np.abs(value) > threshold:
                                    hessian[i][j_s[k]] = value
        return hessian

    def read_dipole_derivative(self, cutoff=1e-12):
        """
        Return the Orca dipole derivative vector (3Nx3) matrix.

        Units: (Eh*bohr)^1/2
        """
        with open(self.HESSIAN_FILE) as hess_f:
            line = hess_f.readline()
            while line:
                if line.strip() == "$dipole_derivatives":
                    line = hess_f.readline()
                    threeN = int(line.strip())
                    dipole_derivatives = np.zeros((threeN, 3), dtype=FLOAT)
                    for i in range(threeN):
                        line = hess_f.readline()
                        values = [FLOAT(v) for v in line.split()[:]]
                        for j, value in enumerate(values):
                            if np.abs(value) > cutoff:
                                dipole_derivatives[i, j] = value
                    break
                line = hess_f.readline()
        return dipole_derivatives

    def find_displacements(self):
        """
        Find all displacement directories in self.directory.

        Return a list containing strings of the format [directory, basename].
        """
        re_dir = re.compile("Displacement_(\d+)")
        re_basename = re.compile("(([\s\w_+-]+)_D[_]*(\d+))")
        dirlist = [d for d in os.listdir(self.directory) if os.path.isdir(d)]
        displacements = []
        for d in sorted(dirlist):
            if re_dir.match(d):
                disp_num = re_dir.search(d).group(1)
                for element in os.listdir(d):
                    if re_basename.search(element):
                        if re_basename.search(element).group(3) == disp_num:
                            basename = re_basename.search(element).group(1)
                            disp_dir = os.path.join(self.directory, d)
                            displacements.append([disp_dir, basename])
                            break
        if not displacements:
            sys.exit("OrcaOutput.find_displacements(): "
                     "No displacements found")
        return displacements

    def get_displaced_geometries(self):
        """Return a list of Hessians, one for each displacement."""
        geometries = []
        disps = self.find_displacements()
        nTransRot = self.geometry.nTransRot()
        threeN = 3 * self.geometry.nAtoms
        number_of_displacements = 2 * (threeN - nTransRot)
        for disp in disps:
            orca_out = OrcaOutput(disp[0], disp[1])
            geometries.append(orca_out.geometry)
        if len(geometries) != number_of_displacements:
            sys.exit("OrcaOutput.get_displaced_geometries(): "
                     "Could only find {} of {} displacements.".format(
                         len(geometries), number_of_displacements))
        return geometries

    def get_displaced_Hessians(self):
        """Return a list of Hessians, one for each displacement."""
        hessians = []
        disps = self.find_displacements()
        nTransRot = self.geometry.nTransRot()
        threeN = 3 * self.geometry.nAtoms
        number_of_displacements = 2 * (threeN - nTransRot)
        for disp in disps:
            orca_out = OrcaOutput(disp[0], disp[1])
            hessians.append(orca_out.hessian)
        if len(hessians) != number_of_displacements:
            sys.exit("OrcaOutput.get_displaced_Hessians(): "
                     "Could only find {} of {} displacements.".format(
                         len(hessians), number_of_displacements))
        return hessians

    def get_displaced_dipole_derivatives(self):
        """Return a list of Hessians, one for each displacement."""
        dipole_derivatives = []
        disps = self.find_displacements()
        nTransRot = self.geometry.nTransRot()
        threeN = 3 * self.geometry.nAtoms
        number_of_displacements = 2 * (threeN - nTransRot)
        for disp in disps:
            orca_out = OrcaOutput(disp[0], disp[1])
            dipole_derivatives.append(orca_out.read_orca_dipole_derivative())
        if len(dipole_derivatives) != number_of_displacements:
            sys.exit("OrcaOutput.get_displaced_dipole_derivatives(): "
                     "Could only find {} of {} displacements.".format(
                         len(dipole_derivatives), number_of_displacements))
        return dipole_derivatives

    def get_internal_coordinates(self):
        """Read the internal coordinates from the output file."""
        int_coords = []
        with open(self.OUT_FILE) as orca_out:
            line = orca_out.readline()
            while line:
                if "INTERNAL COORDINATES (ANGSTROEM)" in line:
                    line = orca_out.readline()
                    line = orca_out.readline()
                    geom = []
                    while line.strip():
                        if len(line.split()) < 7:
                            break
                        v = line.split()
                        geom.append([v[0], int(v[1]), int(v[2]), int(v[3]),
                                     float(v[4]), float(v[5]), float(v[6])])
                        line = orca_out.readline()
                    int_coords.append(geom)
                line = orca_out.readline()
        if int_coords:
            self.geometry.set_zmato(int_coords[-1])
            return int_coords
        else:
            sys.exit("OrcaOutput.get_internal_coordinates(): "
                     "No geometries found")

    def get_orbital_populations(self, threshold=5.0):
        """
        Return an MO (occupied) population dictionary.

        treshold: The neccesary contribution of an AO in percent that is
                  required before it's added to the dictionary.

        This uses the Loewdin reduced orbital population analysis of orca which
         can be activated by including the following in the input file:
        %output
           Print[P_OrbPopMO_L]  1
        end
        """
        columns = 6  # Orca prints 6 columns at a time
        # MO properties: Number, Energy, Occupation, AO data
        re_mo_numb = re.compile('\s+(\d+)' * columns)
        re_mo_ener = re.compile('\s+([-.\d]+)' * columns)
        re_mo_occu = re.compile('\s+(1.00000|0.00000)' * columns)
        re_mo_line = re.compile('\s*(\d+)\s+(\w+)\s+([\w\d+-]+)' +
                                '\s+(\d+\.\d)' * columns)

        mo_dict = {}
        mo_read_flag = False
        ene_read_flag = False
        occ_read_flag = False
        line_read_flag = False
        orbital_occupied = True
        for line in self.OUT.split('\n'):
            if "LOEWDIN REDUCED ORBITAL POPULATIONS PER MO" in line:
                mo_read_flag = True
                continue
            if not mo_read_flag:
                continue
            if mo_read_flag:
                if re_mo_numb.search(line):
                    # reading mo number
                    mos = [int(mo) for mo in line.split()]
                    ene_read_flag = True
                    continue
                if (re_mo_ener.search(line) and not occ_read_flag and
                        ene_read_flag):
                    # reading mo energies
                    ene = [float(e) for e in re_mo_ener.search(line).groups()]
                    ene_read_flag = False
                    occ_read_flag = True
                    continue
                if (re_mo_occu.search(line) and occ_read_flag):
                    # reading mo occupations
                    occ = [float(o) for o in re_mo_occu.search(line).groups()]
                    occ_read_flag = False
                    line_read_flag = True
                    details, per = [], []
                    continue
                if (line_read_flag and re_mo_line.search(line)):
                    # reading mo details and percentages
                    details.append(re_mo_line.search(line).groups()[:3])
                    per.append([float(f) for f in
                                re_mo_line.search(line).groups()[3:]])
                    continue
                if (line_read_flag and not line.strip()):
                    # writing the orbital details to the mo dictionary
                    for i in range(columns):
                        mo = mos[i]
                        if occ[i] == 0.0:
                            orbital_occupied = False
                        if (mo not in mo_dict and occ[i] > 0):
                            mo_dict[mo] = {'MO': mo,
                                           'energy': ene[i],
                                           'occ': occ[i],
                                           'AOs': {}}
                            for j, detail in enumerate(details):
                                if per[j][i] >= threshold:
                                    ao = [int(detail[0]), detail[1],
                                          detail[2], per[j][i]]
                                    mo_dict[mo]['AOs'][j] = ao
                    line_read_flag = False
                if not orbital_occupied:
                    break
        if not mo_dict:
            sys.exit("OrcaOutput.get_orbital_populations(): "
                     "No Loewdin reduced orbital population analysis found")
        return mo_dict

    def filter_orbitals(self, orb_pop, **kwargs):
        """
        Return summed orbital populations for certain atom types/numbers.

        We can select specific atom(s) with the atom(s) keyword:
            e.g. atom=24 or atoms=[24, 35, 42];
         specific element(s):
            e.g. element="Pt" or atoms=["O", "H"],
         as well as orbital types:
            e.g. o_type="f"
        """
        threshold = 50.0  # Percentage of the total contrib. filtered AOs have

        if 'orbital_type' in kwargs:
            o_type = kwargs['orbital_type']
        else:
            o_type = 'f'  # Default

        if 'atom' in kwargs:
            atoms = [kwargs['atom']]
        elif 'atoms' in kwargs:
            atoms = kwargs['atoms']
        else:
            atoms = []
        if 'element' in kwargs:
            elements = [kwargs['element']]
        elif 'elements' in kwargs:
            elements = kwargs['elements']
        else:
            elements = []

        if (not elements and not atoms):
            sys.exit("OrcaOutput.filter_orbitals(): "
                     "You need to specify either some atoms or some elements")
        filter_results = []
        for mo in orb_pop.values():
            pop_total = 0.0
            for ao in mo['AOs'].values():
                if not ao[2][0] == o_type:  # Selecting the first letter.
                    continue
                if (atoms and elements):
                    if (ao[0] in atoms and ao[1] in elements):
                        pop_total += ao[3]
                elif atoms:
                    if ao[0] in atoms:
                        pop_total += ao[3]
                else:
                    if ao[1] in elements:
                        pop_total += ao[3]
            if pop_total > threshold:
                filter_results.append([mo['MO'], mo['energy'],
                                       o_type, pop_total])
        return filter_results

    def get_partial_charges_spins(self):
        """
        Return a list of position, element, partial charge and partial spin.

        These densities are also written in the self.geometry atom objects.
        """
        re_cs = re.compile("\s*(\d+) (\w+)\s*:\s+([-.\d]+)\s+([-.\d]+)")
        read_flag = False
        partial_charges_spins = []
        for line in self.OUT.split('\n'):
            if "LOEWDIN ATOMIC CHARGES AND SPIN POPULATIONS" in line:
                read_flag = True
                continue
            if read_flag:
                if re_cs.search(line):
                    print(line)
                    hit = re_cs.search(line).groups()
                    num = int(hit[0])
                    el = hit[1]
                    pc = float(hit[2])
                    ps = float(hit[3])
                    partial_charges_spins.append([num, el, pc, ps])
                    self.geometry.atoms[num].add_partial_charge_density(pc)
                    self.geometry.atoms[num].add_partial_spin_density(ps)
            if (read_flag and not line.strip()):
                break
        return partial_charges_spins

