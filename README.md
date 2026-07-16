# ![](doc/crisp_logo_dark.svg#gh-light-mode-only)
# ![](doc/crisp_logo_light.svg#gh-dark-mode-only)<br> Cosmic Ray Stochastic Interactions for Propagation

A convenience package to compute quantities related to the propagation of ultrahigh-energy based on closed form probability distribution functions. Applicable for both in-source and extragalactic propagation scenarios.

[API documentation](https://leonel-morejon.org/crisp/crisp.html)

## Usage

Try the examples directly in your browser — no installation needed:

| Notebook | Binder | Colab |
|----------|--------|-------|
| Cross Sections | [![Binder](https://gesis.mybinder.org/badge_logo.svg)](https://gesis.mybinder.org/v2/gh/mohller/crisp/HEAD?filepath=examples/Cross_Sections.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohller/crisp/blob/main/examples/Cross_Sections.ipynb) |
| Injection Composition | [![Binder](https://gesis.mybinder.org/badge_logo.svg)](https://gesis.mybinder.org/v2/gh/mohller/crisp/HEAD?filepath=examples/Injection_composition.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohller/crisp/blob/main/examples/Injection_composition.ipynb) |
| Nuclear Decays | [![Binder](https://gesis.mybinder.org/badge_logo.svg)](https://gesis.mybinder.org/v2/gh/mohller/crisp/HEAD?filepath=examples/Nuclear_Decays.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohller/crisp/blob/main/examples/Nuclear_Decays.ipynb) |

## Installation

Install directly from PyPI:

```bash
pip install crisp-pypi
```

Or clone the repository and install in editable mode for development:

```bash
git clone https://github.com/mohller/crisp.git
cd crisp
pip install -e .
```

### Cross-section tables (optional, downloaded post-installation)

The code employs photodisintegration and photopion cross-section tables from the
[CRPropa3-data](https://github.com/CRPropa/CRPropa3-data) repository.
These tables are **not bundled** with the package, but **are downloaded automatically post installation the first time it is used**. The user can, however, use some of the cross section models available in CRISP as alternative.

The user can also fetch the tables any time once with either:

```python
import crisp
crisp.fetch_crpropa_tables()          # saves to ~/.cache/crisp/CRPropa3-data/tables/
```

```bash
crisp-fetch-data                      # on the terminal, equivalent
```

## Citation

Use of the code should be aknowledged citing

*Stochastic analysis of ultrahigh-energy cosmic ray interactions*\
L. Morejon, KH. Kampert\
A&A, 708, A21 (2026)\
doi: [10.1051/0004-6361/202557405](https://doi.org/10.1051/0004-6361/202557405)

## Author

Leonel Morejon\
Wuppertal University

## Contact

For questions or collaborations, feel free to open an issue or reach out.
