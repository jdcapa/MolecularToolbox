# -*- coding: UTF-8 -*-
"""This module contains a collection of printing functions for this package."""


import numpy as np
import sys

TAB = " " * 2


def print_complex(np_complex, scaling=1.0, precision=6):
    """Return a string of a (numpy) complex number."""
    freq_f = "{{0:> {},.{}f}}".format(7 + precision, precision)
    if np_complex.imag:
        # the first case should not be possible...
        if np_complex.real:
            pm = "+-"[np_complex.imag < 0.0]
            return (freq_f + pm + freq_f + "i").format(np_complex.real,
                                                       abs(np_complex.imag))
        else:
            return (freq_f + "i").format(np_complex.imag)
    else:
        return (freq_f + " ").format(np_complex.real * scaling)


def print_harmonics(harmonic, scaling=1.0,):
    """Return an output string containing harmonic frequencies in 1/cm."""
    output_string = "Harmonic vibrational properties\n"
    output_string += "  i   v [1/cm]"
    harm_frequencies = harmonic.freq_inv_cm
    nTransRot = harmonic.geometry.nTransRot()
    if np.all(harmonic.dipole_derivatives):
        output_string += " " * 4 + "I [km/mol]\n" + "-" * 28 + "\n"
        tmp_str = '{0:> 4,d} {1} {2:>10,.4f}\n'
        harm_intensities = harmonic.harmonic_intensities()
        for i, f in enumerate(harm_frequencies):
            if i == nTransRot:
                output_string += "   " + "-" * 25 + "\n"
            output_string += tmp_str.format(i + 1,
                                            print_complex(f, scaling, 4),
                                            harm_intensities[i])
        output_string += "-" * 28 + "\n"
    else:
        output_string += "\n" + "-" * 17 + "\n"
        tmp_str = '{0:> 4,d} {1}\n'
        for i, f in enumerate(harm_frequencies):
            if i == nTransRot:
                output_string += "   " + "-" * 13 + "\n"
            output_string += tmp_str.format(i + 1,
                                            print_complex(f, scaling, 4))
        output_string += "-" * 17 + "\n"
    return output_string


def csv_harmonics(harmonic, scaling=1.0,):
    """Return a comma-separated-value string of the harmonic transitions."""
    output_string = ""
    if not np.all(harmonic.dipole_derivatives):
        sys.exit("Printfunctons.csv_harmonics():" +
                 "Dipole derivatives are required to produce a 2D csv.")
    nTransRot = harmonic.geometry.nTransRot()
    frequencies = harmonic.freq_inv_cm[nTransRot:].real
    intensities = harmonic.harmonic_intensities()[nTransRot:]
    csv_str = "{:> 9.3f}, {:> 10.4f}\n"
    for i in range(harmonic.geometry.nVib()):
        output_string += csv_str.format(frequencies[i] * scaling,
                                        intensities[i])
    return output_string


def print_np_2Dmatrix(mat, precision=9):
    """Pretty print a 2D-matrix."""
    output_string = ""  # + TAB
    shape = mat.shape
    if not len(shape) == 2:
        return "Matrix not two-dimensional"
    for line in mat:
        for element in line:
            element = np.complex(element)
            output_string += print_complex(element, precision)  # + TAB
        output_string += "\n"  # + TAB
    return output_string  # [:-len(TAB)]


def print_VibRot_constants(self, alpha, geometry, precision=8):
    """Print the Vibrational-rotational constants alpha_k."""
    nVib, nTransRot = geometry.nVib(), geometry.nTransRot()
    output_string = "Vibrational-Rotational constants:\n\n"
    const_str = " {{:> {},.{}f}}".format(6 + precision, precision)

    for a in range(3):
        for i in range(nVib):
            j = i + nTransRot + 1
            v = alpha[a, i]
            output_string += ("{:> 3,d} " * 2).format(a + 1, j)
            output_string += (const_str * 4).format(*v)
            output_string += const_str.format(np.sum(v)) + '\n'
    return output_string


def print_rigid_rotational_constants(rotation, model='rigid'):
    """Print rotational constants in cm^-1 and MHz."""
    output_string = "{} rotational constants (".format(model.title())
    if model == 'rigid':
        abc_e_M = rotation.rigid_rotational_constants("MHz")
        abc_e_c = rotation.rigid_rotational_constants("1/cm")
        r_type = rotation.geometry.rot_prop.rotational_symmetry()[1]
        output_string += "{} molecule)\n".format(r_type.title())
        output_string += '    {:^12} {:^12}\n'.format('1/cm', 'MHz')
        for i, l in enumerate(['A', 'B', 'C']):
            if abc_e_c[i] > 1e-5:
                output_string += '{} '.format(l)
                output_string += '{:> 12,.6f} '.format(abc_e_c[i])
                output_string += '{:> 12,.2f} \n'.format(abc_e_M[i])
        return output_string
    else:
        sys.exit("PrintFunctions.print_rigid_rotational_constants(): "
                 "Model {} is undefined.".format(model))


def print_anharmonic_constants(chi, precision=6):
    """Print the anharmonic constants chi."""
    output_string = "Anharmonic constants:\n\n"
    const_str = " {{:> {},.{}f}}\n".format(7 + precision, precision)
    # print_str = "{:> 3,d} " + const_str * self.geometry.nVib() + '\n'
    for i, row in enumerate(chi):
        i += 7
        for j, value in enumerate(row):
            j += 7
            if j >= i:
                output_string += ("{:> 3,d} {:> 3,d}" +
                                  const_str).format(i, j, value)
        # output_string += print_str.format(i, *row)
    return output_string


def print_coriolis_zeta(coriolis_zeta, precision=6):
    """Print the Coriolis zeta matrices."""
    output_string = ""
    const_str = " {{:> {},.{}f}}".format(7 + precision, precision)
    print_str = "{:> 3,d} " * 2 + const_str * 3 + '\n'
    threeN = len(coriolis_zeta[0])

    for i in range(threeN):
        for j in range(threeN):
            if (i != j and i > j):
                output_string += print_str.format(i + 1, j + 1,
                                                  coriolis_zeta[0, i, j],
                                                  coriolis_zeta[1, i, j],
                                                  coriolis_zeta[2, i, j])
    return output_string


def print_force_constants(fc_mat, geometry,
                          mat_type="cubic", precision=4):
    """
    Return an output string of a pretty printed force constant matrix.

    mat_type: ("quadratic", "cubic", "semi-quartic")
    """
    output_string = ""
    nVib, nTransRot = geometry.nVib(), geometry.nTransRot()
    const_str = " {{:> {},.{}f}}".format(7 + precision, precision)
    corr = nTransRot + 1  # correction for printing
    print_zero = False

    if mat_type == "quadratic":
        print_str = "{:> 3,d} " + const_str + '\n'
        for i in range(nVib):
            output_string += print_str.format(i + 1, fc_mat[i + nTransRot,
                                                            i + nTransRot])
    elif mat_type == "cubic":
        print_str = "{:> 3,d} " * 3 + const_str + '\n'
        for i in range(nVib):
            for j in range(nVib):
                for k in range(nVib):
                    if (np.abs(fc_mat[i, j, k]) < 1e-4 and not print_zero):
                        continue
                    output_string += print_str.format(i + corr,
                                                      j + corr,
                                                      k + corr,
                                                      fc_mat[i, j, k])
    elif mat_type == "semi-quartic":
        print_str = "{:> 3,d} " * 4 + const_str + '\n'
        for i in range(nVib):
            for j in range(nVib):
                for k in range(nVib):
                    if np.abs(fc_mat[i, j, k]) < 1e-4:
                        continue
                    output_string += print_str.format(i + corr,
                                                      j + corr,
                                                      k + corr,
                                                      k + corr,
                                                      fc_mat[i, j, k])
    else:
        sys.exit("PrintFunctions.print_force_constants(): "
                 "mat_type \'{}\' unknown".format(mat_type))
    return output_string


def print_cartesian_diplacements(harmonic, unit="bohr",
                                 precision=9, scaling=1.0):
    """
    Return a string of back transformed normal coordinates.

    Unit: Bohr or Angstroem.
    This method should produce the same numbers as in the cfour QUADRATURE
     file (in bohr).
    """
    output_string = ""
    vec_str = " {{:> {},.{}f}}".format(5 + precision, precision)
    freq_str = "{}\nv{} = {} 1/cm\n"
    inten_str = "{}\nv_{} = {} 1/cm, I_{} = {:,.3f} km/mol\n"

    nAtoms = harmonic.geometry.nAtoms
    nTransRot = harmonic.geometry.nTransRot()
    harm_inv_cm = harmonic.freq_inv_cm[nTransRot:]
    if np.all(harmonic.dipole_derivatives):
        intensities = harmonic.harmonic_intensities()[nTransRot:]
    cart_disps = harmonic.cartesian_displacements(1.0, unit)[0]

    for i, disp in enumerate(cart_disps):
        freq = print_complex(harm_inv_cm[i], scaling, 2)
        if np.all(harmonic.dipole_derivatives):
            output_string += inten_str.format(nAtoms, i + 1, freq,
                                              i + 1, intensities[i])
        else:
            output_string += freq_str.format(nAtoms, i + 1, freq)
        for a, atom in enumerate(harmonic.geometry.atoms):
            v = np.zeros((3,))
            for j in range(3):
                v[j] = disp[3 * a + j]
            # atom.coordinates are given in Angstroem
            output_string += atom.print_cartesian(precision, unit) + "    "
            output_string += (vec_str * 3).format(*v) + "\n"
    return output_string


def print_normal_coordinates(normal_modes, nTransRot=6):
    """Return an output string containing the normal coordinates."""
    sFormat_Vib = "{:>8,.5f} " * (len(normal_modes) - nTransRot) + '\n'

    output_string = 'Translational and Rotational modes:\n'
    for normal_mode in normal_modes:
        string = ""
        for p in normal_mode[:nTransRot]:
            string += "{0:>6,.3f}{1}{2:>5,.3f}j ".format(p.real,
                                                         '+-'[p.imag < 0],
                                                         abs(p.imag))
        output_string += string + '\n'

    output_string += '\nVibrational modes:\n'
    for normal_mode in normal_modes:
        output_string += sFormat_Vib.format(
            *[f.real for f in normal_mode[nTransRot:]])
    print (output_string)


def print_harmonic_VPT2_derivatives(vpt2_derivatives, harmonic, precision=3):
    """
    Print Matthew's harmonic VPT2 derivative matrix.

    This matrix can indicate the presence of Fermi resonances.
    """
    emph = '\033[91m'  # '\033[1m' - bold, '\033[91m' - red, '\033[4m' - uline
    value_str = " {{:> {},.{}f}}".format(4 + precision, precision)
    emph_value_str = emph + value_str + '\033[0m'
    output_string = "\nHarmonic VPT2 derivative matrix:\n\n"
    nVib = harmonic.geometry.nVib()
    header = (" " * precision + "v_{:<3,d}") * nVib
    output_string += " " * 5 + header.format(*list(range(1, nVib + 1))) + '\n'
    for i in range(nVib):
        output_string += "w_{:<3,d}".format(i + 1)
        for j in range(nVib):
            value = vpt2_derivatives[i, j]
            if abs(value) >= 0.5:
                output_string += emph_value_str.format(value)
            else:
                output_string += value_str.format(value)
        output_string += "\n"
    return output_string
