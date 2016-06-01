# -*- coding: UTF-8 -*-
"""These classes deal with the Orca input and output structure."""
from __future__ import print_function
from __future__ import division
import os
import re
import sys
import numpy as np
from numpy import linalg
from geometry import Geometry
import systemtools as ST

TAB = " " * 4
FLOAT = np.float128


class OrcaOutput(object):
    """This class is dealing with the orca 4.0 output structure."""

    def __init__(self, source_directory, basename=None):
        """
        Initiate the class with the source_directory name.

        Everything eles should be detected automatically.
        If multiple jobs are present in that directory, a basename needs to be
         specified.
        """
        super(OrcaOutput, self).__init__()
        # All we need initially are the directory and the file names.
        self.directory = source_directory
        self.filenames(source_directory, basename)
        self.geometry = self.get_xyz_geometries()[-1]
        if self.has_gradient:
            gradient = self.read_orca_gradient()
            if np.any(gradient):
                self.gradient = gradient
        if self.has_hessian:
            self.hessian = self.read_orca_hessian()

    def filenames(self, source_directory, basename):
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
                if occurances > 1:
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
            if (ST.get_memory(0.5, "k") >
                    os.path.getsize(self.OUT_FILE)):
                with open(self.OUT_FILE) as out:
                    self.OUT = out.read()
            else:
                sys.exit("OrcaOutput.filenames(): "
                         "{} is {} GB (too large)".format(
                             os.path.basename(self.OUT_FILE),
                             os.path.getsize(self.OUT_FILE)))
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
                    "ref": self.get_Referce(),
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
        re_MP2 = re.compile("ORCA MP2 CALCULATION")
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
        with open(self.OUT_FILE) as out_file:
            for line in out_file:
                if "Your calculation utilizes the basis:" in line:
                    return line.strip().split(":")[-1].strip()

    def get_Referce(self):
        """Obtain the reference determinant type."""
        re_HF = re.compile("Hartree-Fock type\s+HFTyp\s+[.]+\s*(\w+)")
        with open(self.OUT_FILE) as out_file:
            for line in out_file:
                if re_HF.search(line):
                    return re_HF.search(line).group(1)

    def get_Charge(self):
        """Read the charge."""
        re_Charge = re.compile("Total Charge\s+Charge\s+[.]+\s*([-+\d]+)")
        with open(self.OUT_FILE) as out_file:
            for line in out_file:
                if re_Charge.search(line):
                    return int(re_Charge.search(line).group(1))

    def get_Multiplicity(self):
        """Read the multiplicity."""
        re_Mult = re.compile("Multiplicity\s+Mult\s+[.]+\s*(\d+)")
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
        with open(self.OUT_FILE) as out_file:
            for line in out_file:
                if re_GEO_DeltaE.search(line):
                    geo_DeltaE = FLOAT(re_GEO_DeltaE.search(line).group(1))
                if re_GEO_RMSG.search(line):
                    geo_RMSG = FLOAT(re_GEO_RMSG.search(line).group(1))
                if (geo_DeltaE and geo_RMSG):
                    return geo_DeltaE, geo_RMSG

    def get_energies(self):
        """
        Read the final energies as well as the energy contributions.

        Return an array of energy contributions (all units in Eh).
        If the array has more than one column, the first one is the
         SCF and the following ones are post-HF contributions
         depending on the method used;
         MP2:       [SCF, MP2],
         CCSD:      [SCF, CCSD],
         CCSD(T):   [SCF, CCSD, (T)],
         where the sum of all components equals the total energy.
        """
        method = self.get_Method()
        # re_scf = re.compile('E\(0\)\s+[.]+\s+([-\d.]+)')
        re_ccsd = re.compile('E\(CORR\)\s+[.]+\s+([-\d.]+)')
        re_pT = re.compile('Triples Correction \(T\)\s+[.]+\s+([-\d.]+)')
        re_scf = re.compile('Total Energy\s+:\s+([-\d.]+) Eh')
        re_MP2 = re.compile('MP2 CORRELATION ENERGY\s+:\s+([-\d.]+) Eh')
        if method == "MP2":
            e_scf = np.array([FLOAT(e) for e in re_scf.findall(self.OUT)])
            e_mp2 = np.array([FLOAT(e) for e in re_MP2.findall(self.OUT)])
            return e_scf, e_mp2
        elif "CCSD" in method:
            e_scf = np.array([FLOAT(e) for e in re_scf.findall(self.OUT)])
            e_ccsd = np.array([FLOAT(e) for e in re_ccsd.findall(self.OUT)])
            if method == "CCSD(T)":
                e_PT = np.array([FLOAT(e) for e in re_pT.findall(self.OUT)])
                return e_scf, e_ccsd, e_PT
            else:
                return e_scf, e_ccsd
        else:
            re_final = re.compile("FINAL SINGLE POINT ENERGY\s+([-\d.]+)")
            return np.array([FLOAT(e) for e in re_final.findall(self.OUT)])

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

        with open(self.OUT_FILE) as orca_out:
            line = orca_out.readline()
            while line:
                if "CARTESIAN COORDINATES (A.U.)" in line:
                    line = orca_out.readline()
                    line = orca_out.readline()
                    line = orca_out.readline()
                    geometry = []
                    while line.strip():
                        if len(line.split()) < 8:
                            break
                        v = line.split()
                        geometry.append([v[0], v[1], v[4], v[5], v[6], v[7]])
                        line = orca_out.readline()
                    geometries.append(Geometry(geometry, charge=charge,
                                               mult=mult,
                                               distance_units='Bohr'))
                line = orca_out.readline()
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

    def read_orca_hessian(self, threshold=1e-10):
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

    def read_orca_dipole_derivative(self, cutoff=1e-12):
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
