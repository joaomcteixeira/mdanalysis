# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# MDAnalysis --- https://www.mdanalysis.org
# Copyright (c) 2006-2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
#
# Please cite your use of MDAnalysis in published work:
#
# R. J. Gowers, M. Linke, J. Barnoud, T. J. E. Reddy, M. N. Melo, S. L. Seyler,
# D. L. Dotson, J. Domanski, S. Buchoux, I. M. Kenney, and O. Beckstein.
# MDAnalysis: A Python package for the rapid analysis of molecular dynamics
# simulations. In S. Benthall and S. Rostrup editors, Proceedings of the 15th
# Python in Science Conference, pages 102-109, Austin, TX, 2016. SciPy.
# doi: 10.25080/majora-629e541a-00e
#
# N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein.
# MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations.
# J. Comput. Chem. 32 (2011), 2319--2327, doi:10.1002/jcc.21787
#

"""
PDBx/mmCIF Topology Parser
=========================================================================

This topology parser uses a standard PDBx/mmCIF file to build an internal 
structure representation (list of atoms). Optionally parses bonds, box vectors,
and others if these are present in the file. Useful when dealing with very 
large files that break the PDB format (>9,999 residues or >99,999 atoms).

.. Note::

   The parser processes atoms and their names. Masses are guessed and set to 0
   if unknown. Partial charges are not set.

See Also
--------
* :class:`MDAnalysis.core.universe.Universe`


Classes
-------

.. autoclass:: PDBxParser
   :members:
   :inherited-members:

"""
from __future__ import absolute_import, print_function

import numpy as np
import warnings

from six.moves import range
from .guessers import guess_masses, guess_types
from ..lib import util
from .base import TopologyReaderBase, change_squash
from ..core.topology import Topology
from ..core.topologyattrs import (
    Atomnames,
    Atomids,
    AltLocs,
    Bonds,
    ChainIDs,
    Atomtypes,
    ICodes,
    Masses,
    Elements,
    Occupancies,
    RecordTypes,
    Resids,
    Resnames,
    Resnums,
    Segids,
    Tempfactors,
)

def float_or_default(val, default):
    try:
        return float(val)
    except ValueError:
        return default


class PDBxParser(TopologyReaderBase):
    """Parser that obtains a list of atoms from a standard PDBx/mmCIF file.



    Creates the following Attributes:
        Atoms:
         - altLoc
         - atom ID
         - bfactor
         - masses (guessed)
         - name
         - occupancy
         - type (guessed)
         - bonds (from _struct_conn)

        Residues:
         - icode
         - resname
         - resid
         - resnum

        Segments:
         - segid
         - model

    Guesses the following Attributes:
     - masses
     - types

    .. versionadded:: 0.x
    """
    format = ['CIF`', 'MMCIF']

    def parse(self, **kwargs):
        """Parse atom information from PDBx/mmCIF file

        Returns
        -------
        MDAnalysis Topology object
        """
        top = self._parseatoms()

        try:
            bonds = self._parsebonds(top.ids.values)
        except AttributeError:
            warnings.warn("Invalid atom serials were present, "
                          "bonds will not be parsed")
        else:
            top.add_TopologyAttr(bonds)

        return top

    def _parseatoms(self):
        """Create the initial Topology object"""
        resid_prev = 0  # resid looping hack

        # Field Indexes
        # To be populated on reading _atom_site.xxx entries
        i_rectype = None
        i_serial = None
        i_atmtype = None
        i_atmname_label = None
        i_atmname_auth = None
        i_altloc = None
        i_resname_label = None
        i_resname_auth = None
        i_chainid_label = None
        i_chainid_auth = None
        i_resid_label = None
        i_resid_auth = None
        i_icode = None
        i_coordx = None
        i_coordy = None
        i_coordz = None
        i_occupancy = None
        i_tempfactor = None
        i_charge = None
        i_modelid = None

        # Storage
        record_types = []

        models = []
        serials = []
        names = []
        altlocs = []
        chainids = []
        icodes = []
        tempfactors = []
        occupancies = []
        atomtypes = []

        resids = []
        resnames = []

        segids = []

        with util.openany(self.filename) as f:

            # Ensure first non-comment line is 'data_'
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                else:
                    break

            if line != 'data_':
                raise ValueError("Bad format in file '{}':"
                                 "First non-comment line should start with "
                                 "'_data'. Found '{}' instead."
                                 "".format(self.filename line))

            # Continue reading the file until we find the coord loop
            for line in f:
                line = line.strip()  # Remove extra spaces
                if not line or line.startswith('#'):  # Skip if empty/comment
                    continue



                record_types.append(line[:6].strip())
                try:
                    serial = int(line[6:11])
                except:
                    try:
                        serial = hy36decode(5, line[6:11])
                    except ValueError:
                        # serial can become '***' when they get too high
                        self._wrapped_serials = True
                        serial = last_wrapped_serial
                        last_wrapped_serial += 1
                finally:
                    serials.append(serial)

                names.append(line[12:16].strip())
                altlocs.append(line[16:17].strip())
                resnames.append(line[17:21].strip())
                chainids.append(line[21:22].strip())

                # Resids are optional
                try:
                    if self.format == "XPDB":  # fugly but keeps code DRY
                        # extended non-standard format used by VMD
                        resid = int(line[22:27])
                    else:
                        resid = int(line[22:26])
                        # Wrapping
                        while resid - resid_prev < -5000:
                            resid += 10000
                        resid_prev = resid
                except ValueError:
                    warnings.warn("PDB file is missing resid information.  "
                                  "Defaulted to '1'")
                    resid = 1
                    icode = ''
                finally:
                    resids.append(resid)
                    icodes.append(line[26:27].strip())

                occupancies.append(float_or_default(line[54:60], 0.0))
                tempfactors.append(float_or_default(line[60:66], 1.0))  # AKA bfactor

                segids.append(line[66:76].strip())
                atomtypes.append(line[76:78].strip())

        # Warn about wrapped serials
        if self._wrapped_serials:
            warnings.warn("Serial numbers went over 100,000.  "
                          "Higher serials have been guessed")

        # If segids not present, try to use chainids
        if not any(segids):
            segids, chainids = chainids, None

        n_atoms = len(serials)

        attrs = []
        # Make Atom TopologyAttrs
        for vals, Attr, dtype in (
                (names, Atomnames, object),
                (altlocs, AltLocs, object),
                (chainids, ChainIDs, object),
                (record_types, RecordTypes, object),
                (serials, Atomids, np.int32),
                (tempfactors, Tempfactors, np.float32),
                (occupancies, Occupancies, np.float32),
        ):
            if not vals is None:
                attrs.append(Attr(np.array(vals, dtype=dtype)))
        # Guessed attributes
        # masses from types if they exist
        # OPT: We do this check twice, maybe could refactor to avoid this
        if not any(atomtypes):
            atomtypes = guess_types(names)
            attrs.append(Atomtypes(atomtypes, guessed=True))
        else:
            attrs.append(Atomtypes(np.array(atomtypes, dtype=object)))

        masses = guess_masses(atomtypes)
        attrs.append(Masses(masses, guessed=True))

        # Residue level stuff from here
        resids = np.array(resids, dtype=np.int32)
        resnames = np.array(resnames, dtype=object)
        if self.format == 'XPDB':  # XPDB doesn't have icodes
            icodes = [''] * n_atoms
        icodes = np.array(icodes, dtype=object)
        resnums = resids.copy()
        segids = np.array(segids, dtype=object)

        residx, (resids, resnames, icodes, resnums, segids) = change_squash(
            (resids, resnames, icodes, segids), (resids, resnames, icodes, resnums, segids))
        n_residues = len(resids)
        attrs.append(Resnums(resnums))
        attrs.append(Resids(resids))
        attrs.append(Resnums(resids.copy()))
        attrs.append(ICodes(icodes))
        attrs.append(Resnames(resnames))

        if any(segids) and not any(val == None for val in segids):
            segidx, (segids,) = change_squash((segids,), (segids,))
            n_segments = len(segids)
            attrs.append(Segids(segids))
        else:
            n_segments = 1
            attrs.append(Segids(np.array(['SYSTEM'], dtype=object)))
            segidx = None

        top = Topology(n_atoms, n_residues, n_segments,
                       attrs=attrs,
                       atom_resindex=residx,
                       residue_segindex=segidx)

        return top

    def _parsebonds(self, serials):
        # Could optimise this by saving lines in the main loop
        # then doing post processing after all Atoms have been read
        # ie do one pass through the file only
        # Problem is that in multiframe PDB, the CONECT is at end of file,
        # so the "break" call happens before bonds are reached.

        # If the serials wrapped, this won't work
        if self._wrapped_serials:
            warnings.warn("Invalid atom serials were present, bonds will not"
                          " be parsed")
            raise AttributeError  # gets caught in parse

        # Mapping between the atom array indicies a.index and atom ids
        # (serial) in the original PDB file
        mapping = dict((s, i) for i, s in enumerate(serials))

        bonds = set()
        with util.openany(self.filename) as f:
            lines = (line for line in f if line[:6] == "CONECT")
            for line in lines:
                atom, atoms = _parse_conect(line.strip())
                for a in atoms:
                    try:
                        bond = tuple([mapping[atom], mapping[a]])
                    except KeyError:
                        # Bonds to TER records have no mapping
                        # Ignore these as they are not real atoms
                        warnings.warn(
                            "PDB file contained CONECT record to TER entry. "
                            "These are not included in bonds.")
                    else:
                        bonds.add(bond)

        bonds = tuple(bonds)

        return Bonds(bonds)


def _parse_conect(conect):
    """parse a CONECT record from pdbs

    Parameters
    ----------
    conect : str
        white space striped CONECT record

    Returns
    -------
    atom_id : int
        atom index of bond
    bonds : set
        atom ids of bonded atoms

    Raises
    ------
    RuntimeError
        Raised if ``conect`` is not a valid CONECT record
    """
    atom_id = np.int(conect[6:11])
    n_bond_atoms = len(conect[11:]) // 5

    try:
        if len(conect[11:]) % n_bond_atoms != 0:
            raise RuntimeError("Bond atoms aren't aligned proberly for CONECT "
                               "record: {}".format(conect))
    except ZeroDivisionError:
        # Conect record with only one entry (CONECT A\n)
        warnings.warn("Found CONECT record with single entry, ignoring this")
        return atom_id, []  # return empty list to allow iteration over nothing

    bond_atoms = (int(conect[11 + i * 5: 16 + i * 5]) for i in
                  range(n_bond_atoms))
    return atom_id, bond_atoms
