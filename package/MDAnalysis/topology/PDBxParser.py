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
         - segid  # same as chainid if present
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
        top = self._parse_atoms()

        # try:
        #     bonds = self._parsebonds(top.ids.values)
        # except AttributeError:
        #     warnings.warn("Invalid atom serials were present, "
        #                   "bonds will not be parsed")
        # else:
        #     top.add_TopologyAttr(bonds)

        return top

    def _tokenize_pdbx_line(self, line):
        """Separates a PDBx/mmCIF line by whitespace respecting quoted fields.
        """
        # TODO: Move compiled regex to global scope

        # Split by unquoted whitespace
        regex = re.compile(r'''
                            '.*?' | # single quoted substring OR
                            ".*?" | # double quoted substring OR
                            \S+     # any consec. non-whitespace character(s)
                            ''', re.VERBOSE)
        raw_tokens = regex.findall(line)

        # Remove quotes from tokens
        re_dequote = re.compile(r'''
                                 ^' | '$ |
                                 ^" | "$
                                ''', re.VERBOSE)
        dequoted_tokens = [re_dequote.sub('', t) for t in raw_tokens]

        # Convert '.' and '?' to None
        return [None if t in set(('.', '?')) else t for t in dequoted_tokens]

    def _auth_or_label(self, vlist, atom_label):
        """Returns _label_X if any value in _auth_X is None."""
        if any(e is None for e in vlist[atom_label]):  # lazy
            return vlist[atom_label.replace('_auth', '_label')]
        return vlist[atom_label]

    def _read_datatables(self, fhandle, table=None, start_line_idx=0):
        """Parses data from PDBx/mmCIF tables.

        Parameters
        ----------
        fhandle : file
            open file handle.
        table: str
            names of table to read (e.g. '_atom_site' or '_cell').
            None (default value) reads all tables.
        start_line_idx : int
            line number increment to add to error reporting messages.

        Returns
        -------
        dictionary with labels as keys and data as values.
        """

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
                tokens = self._tokenize_pdbx_line(line)
                if len(tokens) != len(idx_to_item):
                    raise ValueError("Number of tokens on line {} in file '{}'"
                                     " does not match expected value: {}, {}"
                                     "".format(line_idx, self.filename,
                                               len(tokens), len(idx_to_item)))
                for idx_t, t in enumerate(tokens):
                    pdbx_tbl[cat][idx_to_item[idx_t]].append(t)

        return pdbx_tbl

    def _parse_atoms(self):
        """Create the initial Topology object"""

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
        charges = []

        resids = []
        resnames = []
        segids = []

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

            # Continue reading the file and read _atom_site table
            tbl = self._read_datatables(f, '_atom_site', idx_line)

        # Iterate over table data and convert fields where needed
        # e.g. resid -> int; occupancy -> float
        # When possible, prefer _auth. If None, default to _label.
        record_types = tbl['_atom_site']['group_PDB']
        serials = list(map(int, tbl['_atom_site']['id']))
        atomtypes = tbl['_atom_site']['type_symbol']
        names = self._auth_or_label(tbl['_atom_site'], 'auth_atom_id')
        altlocs = tbl['_atom_site']['label_alt_id']
        resnames = self._auth_or_label(tbl['_atom_site'], 'auth_comp_id')
        chainids = self._auth_or_label(tbl['_atom_site'], 'auth_asym_id')
        _resids = self._auth_or_label(tbl['_atom_site'], 'auth_seq_id')
        resids = list(map(int, _resids))
        icodes = tbl['_atom_site']['pdbx_PDB_ins_code']
        occupancies = list(map(float, tbl['_atom_site']['occupancy']))
        tempfactors = list(map(float, tbl['_atom_site']['B_iso_or_equiv']))

        # If charges are all present, add them.
        _charges = tbl['_atom_site']['pdbx_formal_charge']
        if all(_charges):  # lazy
            charges = list(map(float, _charges))

        # If chains are present, use also as segids
        # Should always be. Either _auth or _label
        segids = chainids

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
            attrs.append(Attr(np.array(vals, dtype=dtype)))

        # Guessed attributes:
        #   - types from names if any is missing
        #   - masses from types
        if any(e is None for e in atomtypes):
            atomtypes = guess_types(names)
            attrs.append(Atomtypes(atomtypes, guessed=True))
        else:
            attrs.append(Atomtypes(np.array(atomtypes, dtype=object)))

        masses = guess_masses(atomtypes)
        attrs.append(Masses(masses, guessed=True))

        # Residue level stuff from here
        resids = np.array(resids, dtype=np.int32)
        resnames = np.array(resnames, dtype=object)
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

        # _atom_site.label_asym_id should *always* be there
        # so we can safely avoid the check.
        segidx, (segids,) = change_squash((segids,), (segids,))
        n_segments = len(segids)
        attrs.append(Segids(segids))

        top = Topology(n_atoms, n_residues, n_segments,
                       attrs=attrs,
                       atom_resindex=residx,
                       residue_segindex=segidx)

        return top

    # def _parsebonds(self, serials):
    #     # Could optimise this by saving lines in the main loop
    #     # then doing post processing after all Atoms have been read
    #     # ie do one pass through the file only
    #     # Problem is that in multiframe PDB, the CONECT is at end of file,
    #     # so the "break" call happens before bonds are reached.

    #     # Mapping between the atom array indicies a.index and atom ids
    #     # (serial) in the original PDB file
    #     mapping = dict((s, i) for i, s in enumerate(serials))

    #     bonds = set()
    #     with util.openany(self.filename) as f:
    #         lines = (line for line in f if line[:6] == "CONECT")
    #         for line in lines:
    #             atom, atoms = _parse_conect(line.strip())
    #             for a in atoms:
    #                 try:
    #                     bond = tuple([mapping[atom], mapping[a]])
    #                 except KeyError:
    #                     # Bonds to TER records have no mapping
    #                     # Ignore these as they are not real atoms
    #                     warnings.warn(
    #                         "PDB file contained CONECT record to TER entry. "
    #                         "These are not included in bonds.")
    #                 else:
    #                     bonds.add(bond)

    #     bonds = tuple(bonds)

    #     return Bonds(bonds)
