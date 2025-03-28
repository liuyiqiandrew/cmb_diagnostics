import pymaster as nmt
import numpy as np
from typing import Hashable, Tuple, overload
from .Constants import PSType


class NmtFieldContainer:
    """ Container holding NaMaster spin fields """
    def __init__(
        self,
        name: str=""
    ) -> None:
        self.name: str = name
        self.f0s: dict = {} # spin-0 fields
        self.f2s: dict = {} # spin-2 fields

    def add_spin2_field(
        self, 
        f2:nmt.NmtField, 
        key: Hashable
    ) -> None:
        """ Add or update spin-2 field """
        if f2.get_maps().shape[0] != 2:
            raise ValueError("Input field is not spin-2.")
        self.f2s[key] = f2

    def add_spin0_field(
        self,
        f0:nmt.NmtField,
        key:Hashable
    ) -> None:
        """ Add or update spin-0 field"""
        if f0.get_maps().shape[0] != 1:
            raise ValueError("Input field is not spin-0.")
        self.f0s[key] = f0

    def get_spin2_field(
        self,
        key:Hashable=None
    ):
        """ 
        Get the spin-2 field with given key. If none is given,
        return the dictionary of fields.
        """
        if key is None:
            return self.f2s
        else:
            return self.f2s[key]
    
    def get_spin0_field(
        self,
        key:Hashable=None
    ):
        """ 
        Get the spin-0 field with given key. If none is given,
        return the dictionary of fields.
        """
        if key is None:
            return self.f0s
        else:
            return self.f0s[key]
    

class PSContainer:
    """ Container for power spectra """
    def __init__(
        self,
        name1 : str="",
        name2 : str|None=None,
        pstype : PSType=PSType.PP,
        eff_ell : np.ndarray|None=None
    ) -> None:
        self.name1 : str = name1
        self.name2 : str = name1 if name2 is None else name2
        self.pstype : PSType = pstype
        self.eff_ell : np.ndarray|None = eff_ell
        self.ps : dict = {} # power spectra
        self.dps : dict = {} # error bar on power spectra
        self.tracers : dict = {}

        # initializing containers
        match pstype:
            case PSType.PP:
                self.ps['BB'], self.ps['EE'], self.ps['EB'], self.ps['BE'] = {}, {}, {}, {}
                self.dps['BB'], self.dps['EE'], self.dps['EB'], self.dps['BE'] = {}, {}, {}, {}
                self.tracers['BB'], self.tracers['EE'], self.tracers['EB'], self.tracers['BE'] \
                    = set(), set(), set(), set()
            case PSType.TP:
                self.ps['TE'], self.ps['TB'] = {}, {}
                self.dps['TE'], self.dps['TB'] = {}, {}
                self.tracers['TE'], self.tracers['TB'] = set(), set()
            case PSType.TT:
                self.ps['TT'] = {}
                self.dps['TT'] = {}
                self.tracers['TT'] = set()

    def add_spectrum(
            self, 
            comp : str, 
            freq1 : float, 
            freq2 : float,
            cl : np.ndarray,
            dcl : np.ndarray | None = None
        ) -> None:
        """Add or update power spectra to container

        Parameters
        ----------
        comp : str
            Power spectral component
        freq1 : float
            Frequency 1 as key
        freq2 : float
            Frequency 2 as key
        cl : np.ndarray
            Power spectrum
        dcl : np.ndarray | None
            Error on power spectrum, by default None

        Raises
        ------
        KeyError
            When incorrect spectral component is provided for the power spectral type for
            component
        """
        if comp not in self.ps:
            raise KeyError(f"Invalid component for {self.pstype.value} pstype.")
        tracer = (freq1, freq2)
        self.ps[comp][tracer] = cl
        self.dps[comp][tracer] = dcl
        self.tracers[comp].add[tracer]

    def get_spectrum(
        self,
        comp : str,
        freq1 : float,
        freq2 : float,
        return_ps_delta : bool=False
    ):
        """Get power spectra from component

        Parameters
        ----------
        comp : str
            Power spectral component
        freq1 : float
            Frequency 1 as key
        freq2 : float
            Frequency 2 as key
        return_ps_delta : bool, optional
            return error on power spectrum as well, by default False

        Returns
        -------
        np.ndarray | Tuple[np.ndarray, np.ndarray]
            power spectrum, or tuple of power spectrum, error pair.

        Raises
        ------
        KeyError
            If invalid power spectral component is provided
        """
        if comp not in self.ps:
            raise KeyError(f"Invalid component combination for {self.pstype.value} pstype.")
        tracer = (freq1, freq2)
        if not return_ps_delta:
            return self.ps[comp][tracer]
        else:
            return self.ps[comp][tracer], self.ps[comp][tracer]
        
    