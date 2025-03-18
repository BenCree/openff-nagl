NAGL Dihedral Energy Prediction
==============================

The goal of this project is to predict dihedral bond energies using NAGL. An example notebook based on work by Lily Wang can be found in examples/train-gnn-dihedrals/train-gnn-central-bond.ipynb.

## Current status
- [x] Verified that c_ij coefficients are calculated via pooling (see https://github.com/openforcefield/openff-nagl/pull/163)
- [x] Test geometry functions (see alpha_test.py)
- [ ] Verify alpha for a test set (biaryl?)
- [ ] Modify NAGL output to a scalar
- [ ] Train with a QCarchive dataset
- [ ] Train with a larger and more diverse dataset




A playground for applying graph convolutional networks to molecules, with a focus on learning continuous "atom-type" embeddings and from these classical molecule force field parameters.

NAGL is mostly based upon the [*End-to-End Differentiable Molecular Mechanics Force Field Construction*](https://arxiv.org/abs/2010.01196) 
preprint by Wang, Fass and Chodera.

NAGL is bound by a [Code of Conduct](https://github.com/openforcefield/openff-nagl/blob/main/CODE_OF_CONDUCT.md).

### [Documentation](https://docs.openforcefield.org/projects/nagl/en/latest/?badge=latest)

See our documentation for notes on [installation](https://docs.openforcefield.org/projects/nagl/en/latest/installation.html), basic usage, theory, and examples!

### Copyright

The NAGL source code is hosted at https://github.com/openforcefield/openff-nagl
and is available under the MIT license (see the file [LICENSE](https://github.com/openforcefield/openff-nagl/blob/main/LICENSE)). Some parts inherit from code distributed under other licenses, as detailed in [LICENSE-3RD-PARTY](https://github.com/openforcefield/openff-nagl/blob/main/LICENSE-3RD-PARTY)).

NAGL inherits from Simon Boothroyd's NAGL library at https://github.com/SimonBoothroyd/nagl.
