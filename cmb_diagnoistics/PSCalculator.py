from .Container import PSContainer, NmtFieldContainer
from .Constants import PSType
import pymaster as nmt
import numpy as np
import itertools
from .utils import knox_covar


class PSCalculator:
    """ Power spectrum calculator """
    def __init__(
        self,
        fc1 : NmtFieldContainer,
        fc2 : NmtFieldContainer | None,
        bins : nmt.NmtBin,
        fsky : float,
        pstypes : list[PSType] = [PSType.PP, PSType.TP]
    ) -> None:
        self.fc1 : NmtFieldContainer = fc1
        self.fc2 : NmtFieldContainer = fc2 if fc2 is not None else fc1
        self.auto_instru : bool = fc2 is None
        self.bins : nmt.NmtBin = bins
        self.eff_ell : np.ndarray = bins.get_effective_ells()
        self.bin_width : float = np.diff(self.eff_ell)[0]
        self.pstypes : list[PSType] = pstypes
        self.fsky : float = fsky

        if PSType.TT in pstypes:
            self.tt_container = PSContainer(self.fc1.name, self.fc2.name, PSType.TT, self.eff_ell)
        if PSType.TP in pstypes:
            self.tp_container = PSContainer(self.fc1.name, self.fc2.name, PSType.TP, self.eff_ell)
        if PSType.PP in pstypes:
            self.pp_container = PSContainer(self.fc1.name, self.fc2.name, PSType.PP, self.eff_ell)

    def calculate_spectra(self, pstype : PSType = PSType.PP):
        match pstype:
            case PSType.PP:
                self.__calculate_pp_spectra()
            case PSType.TP:
                self.__calculate_tp_spectra()


    def __calculate_pp_spectra(self):

        tr1 = self.fc1.get_spin2_field().keys()
        tr2 = self.fc1.get_spin2_field().keys()

        aux_specs1 = {}
        for key in tr1:
            f2 = self.fc1.get_spin2_field(key)
            aux_specs1[key] = nmt.compute_full_master(f2, f2, self.bins)
        
        aux_specs2 = {}
        for key in tr2:
            f2 = self.fc2.get_spin2_field(key)
            aux_specs2[key] = nmt.compute_full_master(f2, f2, self.bins)

        for t1, t2 in itertools.product(tr1, tr2):
            f2_1 = self.fc1.get_spin2_field(t1)
            f2_2 = self.fc2.get_spin2_field(t2)
            ee, eb, be, bb = nmt.compute_full_master(f2_1, f2_2, self.bins)
            dee = knox_covar(aux_specs1[t1][0], aux_specs2[t2][0], ee, ee, \
                                   self.fsky, self.eff_ell, self.bin_width)
            deb = knox_covar(aux_specs1[t1][0], aux_specs2[t2][3], eb, be, \
                                   self.fsky, self.eff_ell, self.bin_width)
            dbe = knox_covar(aux_specs1[t1][3], aux_specs2[t2][0], be, eb, \
                                   self.fsky, self.eff_ell, self.bin_width)
            dbb = knox_covar(aux_specs1[t1][3], aux_specs2[t2][3], bb, bb, \
                                   self.fsky, self.eff_ell, self.bin_width)
            self.pp_container.add_spectrum('EE', t1, t2, ee, dee)
            self.pp_container.add_spectrum('EB', t1, t2, eb, deb)
            self.pp_container.add_spectrum('BE', t1, t2, be, dbe)
            self.pp_container.add_spectrum('BB', t1, t2, bb, dbb)

    def __calculate_tp_spectra(self):
        pass