# GNN Classification 

## Node Features

Each atom in a molecule is represented by a set of features:

| **Feature**           | **Description**                                                | **Categories/Values**                             |
|------------------------|----------------------------------------------------------------|---------------------------------------------------|
| **Atom type**          | Represents all currently known chemical elements.             | 118 types                                         |
| **Degree**             | Number of heavy atom neighbors (non-hydrogen atoms).          | 6 categories                                      |
| **Formal charge**      | Charge assigned to an atom.                                   | -2, -1, 0, 1, 2                                   |
| **Hybridization**      | Orbital hybridization of the atom.                            | sp, sp², sp³, sp³d, sp³d²                        |
| **Aromaticity**        | Indicates whether the atom is aromatic or not.                | Binary (1 = aromatic, 0 = not aromatic)          |


## Edge Features

Each bond in a molecule is described using the following features:

| **Feature**            | **Description**                                                | **Categories/Values**                             |
|------------------------|----------------------------------------------------------------|---------------------------------------------------|
| **Bond type**          | Type of bond between two atoms.                               | Single, double, triple, or aromatic              |
| **Ring**               | Indicates if the bond is part of a ring structure.            | Binary (1 = in a ring, 0 = not in a ring)        |
