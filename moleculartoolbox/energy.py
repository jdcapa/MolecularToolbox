#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""This module contains functions related to various energy components."""
import sys
import numpy as np
from collections import OrderedDict


class Energy(object):
    """docstring for Energy"""
    def __init__(self, job):
        """Initialise the energy object with an Orca/Cfour Output object."""
        super(Energy, self).__init__()
        self.job = job
        self.energies = self.get_energies()

    def get_energies(self):
        """Save all partial energies and return an energy dictionary."""
        energies = self.energies()
        energies = self.add_total_energies_and_differences(energies)
        energies = self.add_gradients(energies)
        return energies

    def total_energy(self, energies, i):
        """Return the total energy of step i in a trajectory."""
        if i < 0:
            i = len(energies) + i
        if i not in energies:
            sys.exit("OrcaOutput.total_energy(): No trj step {}".format(i))
        return np.sum([v for k, v in energies[i].items() if "E_" in k])

    def energies(self):
        """
        Read the final energies as well as the energy contributions.

        Return an array of energy contributions (all units in Eh).
        If the array has more than one column, the first one is the
         SCF and the following ones are post-HF contributions
         depending on the method used;
         SCF:       [SCF],
         DFT:       [SCF],
         MP2:       [SCF, MP2],
         CCSD:      [SCF, CCSD],
         CCSD(T):   [SCF, CCSD, (T)],
         where the sum of all components equals the total energy.
        """
        self.scf = []
        self.mp2 = []
        self.ccsd = []
        self.pT = []

        method = self.job.info["method"]
        if method in ["MP2", "CCSD", "CCSD(T)"]:
            e_scf = self.job.scf_energies()
        else:
            e_scf = self.job.final_energies()

        self.scf = e_scf
        energies = OrderedDict()
        for i in range(len(e_scf)):
            energies[i] = OrderedDict()
            energies[i]["E_SCF"] = e_scf[i]

        if method == "MP2":
            self.mp2 = self.job.mp2_energies()
            for i in range(len(self.mp2)):
                energies[i]["E_MP2"] = self.mp2[i]
        elif "CCSD" in method:
            self.ccsd = self.job.ccsd_energies()
            for i in range(len(self.ccsd)):
                energies[i]["E_CCSD"] = self.ccsd[i]
            if method == "CCSD(T)":
                self.pT = self.job.pT_energies()
                for i in range(len(self.pT)):
                    energies[i]["E_(T)"] = self.pT[i]

        return energies

    def add_total_energies_and_differences(self, energies):
        """Add the total energies (and differences) to the energy list."""
        total = []
        iterations = [k for k in energies.keys() if "E_SCF" in energies[k]]
        for i in iterations:
            if len(energies[0]) > 1:
                energies[i]["total"] = self.total_energy(energies, i)
                total.append(energies[i]["total"])
                j = i - 1
                if j in iterations:
                    if "total" in energies[j]:
                        adiff = np.abs(energies[i]["total"] -
                                       energies[j]["total"])
                        if adiff > 0:
                            energies[i]["log_diff"] = -np.log(adiff)
                elif j == -1:
                    energies[i]["log_diff"] = 0
            else:
                total.append(energies[i]["E_SCF"])
        self.total = np.array(total)
        return energies

    def add_gradients(self, energies):
        """Adds the gradient norms the the energy list."""
        self.gradients = self.job.grad_norms()
        for i in range(len(self.gradients)):
            energies[i]["grad_norm"] = self.gradients[i]
        return energies
