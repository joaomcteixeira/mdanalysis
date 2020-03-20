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

import collections
import re

import numpy as np
import warnings

from .guessers import guess_masses, guess_types
from ..lib import util
from .base import TopologyReaderBase, change_squash
from ..core.topology import Topology
from ..core.topologyattrs import (
    Atomnames,
    Atomids,
    AltLocs,
    Bonds,
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
         - type
         - elements (guess)
         - bonds (from _struct_conn)

        Residues:
         - icode
         - resname
         - resid
         - resnum

        Segments:
         - segid  # taken from chain ids

    Guesses the following Attributes:
     - masses
     - types, if missing
     - elements

    .. versionadded:: 0.x
    """
    format = ['CIF', 'MMCIF']

    def parse(self, **kwargs):
        """Parse atom and bond information from PDBx/mmCIF file

        Returns
        -------
        MDAnalysis Topology object
        """

        return self._parse_atom_n_bonding_data()

    def _tokenize_pdbx_line(self, line):
        """Separates a PDBx/mmCIF line by whitespace respecting quoted fields.
        """

        # Split by unquoted whitespace
        # Use grouping to remove quotes from final output
        # Join groups: two will be empty, one won't.
        regex = re.compile(r'''
                            '(.*?)' | # single quoted substring OR
                            "(.*?)" | # double quoted substring OR
                            (\S+)     # any consec. non-whitespace character(s)
                            ''', re.VERBOSE)
        raw_tokens = [''.join(t) for t in regex.findall(line)]
        # Convert '.' and '?' to empty strings
        return ['' if t in set(('.', '?')) else t for t in raw_tokens]

    def _read_datatables(self, fhandle, table=None, start_line_idx=0):
        """Parses data from PDBx/mmCIF tables.

        Parameters
        ----------
        fhandle : file
            open file handle.
        table: str
            name of table to read (e.g. '_atom_site' or '_cell').
            None (default value) reads all tables.
        start_line_idx : int
            line number increment to add to error reporting messages.

        Returns
        -------
        dictionary with labels as keys and data as values.
        """

        _tokenizer_func = self._tokenize_pdbx_line  # local scope

        # ['atom_site']['group_PDB']: ['ATOM', 'ATOM' ...]'
        pdbx_tbl = collections.defaultdict(dict)
        # 0: 'group_PDB'
        idx_to_item = {}
        _i = 0

        read_data = False
        open_table = False
        for line_idx, line in enumerate(fhandle, start=start_line_idx + 1):
            line = line.strip()
            if line == 'loop_':
                open_table = True
            elif not line or line == '#':
                open_table = False
                read_data = False
                # Reset indexers
                idx_to_item.clear()
                _i = 0
            elif line[0] == '_' and open_table:  # read labels
                if table is None or line.startswith(table):
                    cat, item = line.split('.')
                    pdbx_tbl[cat][item] = []
                    idx_to_item[_i] = item
                    _i += 1
                    read_data = True
            elif read_data:
                tokens = _tokenizer_func(line)
                if len(tokens) != len(idx_to_item):
                    raise ValueError("Number of tokens on line {} in file '{}'"
                                     " does not match expected value: {}, {}"
                                     "".format(line_idx, self.filename,
                                               len(tokens), len(idx_to_item)))
                for idx_t, t in enumerate(tokens):
                    pdbx_tbl[cat][idx_to_item[idx_t]].append(t)

        if not pdbx_tbl:
            table = 'any' if table is None else table
            raise AttributeError("Could not find '{}' table in file '{}'"
                                 "".format(table, self.filename))

        return pdbx_tbl

    def _auth_or_label(self, vlist, atom_label):
        """Returns _label_X if any value in _auth_X is empty/False."""
        if all(vlist[atom_label]):  # lazy
            return vlist[atom_label]
        return vlist[atom_label.replace('_auth', '_label')]

    def _squash_models(self, atom_tbl):
        """Returns a new PDBx _atom_site table containing only entries
        belonging to the first model"""

        models = list(map(int, atom_tbl['pdbx_PDB_model_num']))
        model_idx = models[0]

        # Models should be contiguous, but just in case
        at_idx_model = set((i for i, m in enumerate(models) if m == model_idx))

        # Filter _atom_site entries
        for key, data in atom_tbl.items():
            atom_tbl[key] = [d for i, d in enumerate(data) if i in at_idx_model]
        return atom_tbl

    def _parse_atom_n_bonding_data(self):
        """Create and populate a Topology object with atom and bonding data."""

        # Storage
        record_types = []

        models = []
        serials = []
        names = []
        altlocs = []
        icodes = []
        tempfactors = []
        occupancies = []
        atomtypes = []
        charges = []

        resids = []
        resnames = []
        segids = []

        # To map bonds to atoms we need to use the _label data, which may
        # or may not be used to populate the topology. As such, it's simpler
        # to just read the file twice in a row to get the two datablocks
        # we need. Unsure if util.openany objects always have a .seek() method
        # otherwise we'd just rewind...
        #
        # Round 1: read atoms
        with util.openany(self.filename) as f:

            # Skip initial comments
            for idx_line, line in enumerate(f, start=1):
                line = line.strip()
                if line.startswith('#'):
                    continue
                else:
                    break

            # Ensure we open a data block
            if not line.startswith('data_'):
                raise ValueError("Unexpected field in file '{}' on line {}: {}."
                                 " First non-comment line must open a data"
                                 " block ('data_xxx')."
                                 "".format(self.filename, idx_line, line))

            # Continue reading the file and read atom table
            tbl = self._read_datatables(f, '_atom_site', idx_line)
            tbl_atom_data = tbl['_atom_site']
        #
        # Round 2: read bonds; no need to do checks
        # May or may not have bonds!
        with util.openany(self.filename) as f:
            try:
                tbl = self._read_datatables(f, '_struct_conn', 0)
            except AttributeError:
                tbl_bond_data = {}
            else:
                tbl_bond_data = tbl['_struct_conn']

        #
        # Populate Topology
        #

        # PDBx/mmCIF can have multiple models, all sharing exactly the same
        # atoms and residues and chains. For topology purposes, we can use
        # only the first one.
        models = list(map(int, tbl_atom_data['pdbx_PDB_model_num']))

        if len(set(models)) > 1:
            warnings.warn("File '{}' contains multiple models, will only"
                          " consider the first when building Topology."
                          "".format(self.filename))
            tbl_atom_data = self._squash_models(tbl_atom_data)

        # Iterate over table data and convert fields where needed
        # e.g. resid -> int; occupancy -> float
        # When possible, prefer _auth. If None, default to _label.
        record_types = tbl_atom_data['group_PDB']
        serials = list(map(int, tbl_atom_data['id']))
        atomtypes = tbl_atom_data['type_symbol']
        names = self._auth_or_label(tbl_atom_data, 'auth_atom_id')
        altlocs = tbl_atom_data['label_alt_id']
        resnames = self._auth_or_label(tbl_atom_data, 'auth_comp_id')
        segids = self._auth_or_label(tbl_atom_data, 'auth_asym_id')
        _resids = self._auth_or_label(tbl_atom_data, 'auth_seq_id')
        resids = list(map(int, _resids))
        icodes = tbl_atom_data['pdbx_PDB_ins_code']
        occupancies = list(map(float, tbl_atom_data['occupancy']))
        tempfactors = list(map(float, tbl_atom_data['B_iso_or_equiv']))

        # If charges are all present, add them.
        charges = tbl_atom_data['pdbx_formal_charge']
        if all(charges):  # lazy
            charges = list(map(float, charges))

        n_atoms = len(serials)

        attrs = []
        # Make Atom TopologyAttrs
        for vals, Attr, dtype in (
                (names, Atomnames, object),
                (altlocs, AltLocs, object),
                (record_types, RecordTypes, object),
                (serials, Atomids, np.int32),
                (tempfactors, Tempfactors, np.float32),
                (occupancies, Occupancies, np.float32),
        ):
            attrs.append(Attr(np.array(vals, dtype=dtype)))

        # Guessed attributes:
        #   - types from names if any is missing
        #   - masses from types
        #   - elements from types
        if all(atomtypes):
            attrs.append(Atomtypes(np.array(atomtypes, dtype=object)))
        else:
            atomtypes = guess_types(names)
            attrs.append(Atomtypes(atomtypes, guessed=True))

        masses = guess_masses(atomtypes)
        attrs.append(Masses(masses, guessed=True))

        elements = guess_types(atomtypes)
        attrs.append(Elements(elements, guessed=True))

        # Residue level stuff from here
        resids = np.array(resids, dtype=np.int32)
        resnames = np.array(resnames, dtype=object)
        icodes = np.array(icodes, dtype=object)
        resnums = resids.copy()
        segids = np.array(segids, dtype=object)

        residx, (resids, resnames, icodes, resnums, segids) = change_squash(
            (resids, resnames, icodes, segids),
            (resids, resnames, icodes, resnums, segids))

        n_residues = len(resids)

        attrs.append(Resnums(resnums))
        attrs.append(Resids(resids))
        attrs.append(Resnums(resids.copy()))
        attrs.append(ICodes(icodes))
        attrs.append(Resnames(resnames))

        # Segments
        segidx, (segids,) = change_squash((segids,), (segids,))
        n_segments = len(segids)
        attrs.append(Segids(segids))

        # Bonds
        if tbl_bond_data:
            # Make mapping of (asym, comp, seq, atom) to atom index
            # simplify and use label_X only as this is guaranteed to exist.
            atom_mapping = {}
            for idx, ziptup in enumerate(zip(tbl_atom_data['label_asym_id'],
                                             tbl_atom_data['label_comp_id'],
                                             tbl_atom_data['label_seq_id'],
                                             tbl_atom_data['label_atom_id'])):
                atom_mapping[ziptup] = idx

            bonds = []
            for idx, b in enumerate(zip(tbl_bond_data['ptnr1_label_asym_id'],
                                        tbl_bond_data['ptnr1_label_comp_id'],
                                        tbl_bond_data['ptnr1_label_seq_id'],
                                        tbl_bond_data['ptnr1_label_atom_id'],
                                        tbl_bond_data['ptnr2_label_asym_id'],
                                        tbl_bond_data['ptnr2_label_comp_id'],
                                        tbl_bond_data['ptnr2_label_seq_id'],
                                        tbl_bond_data['ptnr2_label_atom_id'])):

                atomA, atomB = atom_mapping.get(b[:4]), atom_mapping.get(b[4:])
                if atomA is None or atomB is None:
                    warnings.warn("Could not map bond #{} to any known atom."
                                  "".format(idx))
                    continue
                bonds.append((atomA, atomB))
            attrs.append(Bonds(set(bonds)))

        return Topology(n_atoms, n_residues, n_segments,
                        attrs=attrs,
                        atom_resindex=residx,
                        residue_segindex=segidx)
