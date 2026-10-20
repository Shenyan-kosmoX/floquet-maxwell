# Dependencies, source credits and reference material

Original Floquet–Maxwell code is available under the [MIT License](LICENSE).
Upstream components retain their own copyright and license notices. This file
helps you identify the software and reference material used by each part of the
project.

## Legume and the numerical core

The spatial solver is [Legume](https://github.com/fancompute/legume), used here at
version 1.0.3 under the MIT License, copyright (c) 2019 Fan group. Please cite
the relevant Legume papers when using its methods in research; the upstream
[citation guide](https://github.com/fancompute/legume/blob/v1.0.3/README.md#citing)
lists them.

The radiation overlap loop in
[frequency_radiation.py](sptc_toolbox/frequency_radiation.py) is adapted from
Legume's `GuidedModeExp._compute_rad_components`. The module records the source
version, changes and full upstream MIT notice. Keep that notice with the adapted
code.

| Component | Role | License |
| --- | --- | --- |
| Legume-GME | Geometry, static modes and fields | MIT |
| NumPy and SciPy | Arrays, linear algebra and integration | BSD-3-Clause for the main projects; distributed binaries include additional notices |
| Matplotlib | Figures | Matplotlib license, based on the PSF license |
| PyYAML | Configuration files | MIT |
| pytest | Development checks | MIT |

The tensor-selection example uses these components. Its package versions are
recorded alongside the calculation. When distributing a complete environment,
keep the license and notice files shipped with its actual binary packages,
including numerical runtimes and bundled components.

## Additional tools and comparisons

| Component | Use in this workspace | License or access |
| --- | --- | --- |
| Meep / PyMeep | Independent time-domain and Green-response comparisons | GPL-2.0-or-later |
| treams | Sphere and periodic-array comparisons | MIT |
| MPh | Python interface to COMSOL | MIT |
| COMSOL | Selected full-wave comparisons | Separate COMSOL installation and applicable license |
| PyMuPDF | PDF extraction and reference-figure preparation | AGPLv3 or an Artifex license |
| YUCHEN010/Floquet-modulation | External MATLAB reference calculations | GPLv3; retain the upstream MAN and contributor credits |

These tools have their own installation and distribution terms. Directly
integrated GPL/AGPL components and scripts should be shared with the applicable
terms. The project's MIT license covers its original code; copied upstream
software and its notices keep their original scope.

The downloaded pytmod snapshot has conflicting license declarations: its
`LICENSE.txt` contains GPLv3 while package metadata names MIT. Use a version with
clarified licensing before redistributing that snapshot. The Legume tensor
example runs independently of pytmod.

The public source snapshot does not bundle the downloaded pytmod or MATLAB
repositories, PDF extraction utilities, or installed third-party binaries.
The optional `native_lorentz_sigma.cpp` adapter is original project code that
compiles against a separately installed Meep 1.35.0. Meep's terms continue to
apply to the resulting linked component and its distribution.

## Papers, figures and data

The Science Advances Figure 4 source panels in
`sptc_toolbox/results/paper_refresh_20261016/sources/science_fig4*.png`, the
page-6 text excerpt, and the source/digitization images in
`ptc_fig4/results/qualitative_paper_reference/` are from Puneet Garg et al.,
*Photonic time crystals assisted by quasi-bound states in the continuum*,
Science Advances 12, eaed4055 (2026),
[doi:10.1126/sciadv.aed4055](https://doi.org/10.1126/sciadv.aed4055),
copyright © 2026 the authors, distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Source panels were cropped and rasterized from the published figure;
digitization plots are marked separately and retain their extraction limits.
These source materials retain CC BY 4.0 rather than the software MIT license.

The two rendered supplement pages `SI_page15.png` and `SI_page16.png` in
`sptc_toolbox/results/zhang2018_original_alignment_20261019/` come from Lei Zhang
et al., *Space-time-coding digital metasurfaces*, Nature Communications 9, 4334
(2018), [doi:10.1038/s41467-018-06802-0](https://doi.org/10.1038/s41467-018-06802-0),
copyright © 2018 the authors, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The pages were rasterized for parameter inspection; their scientific content
was not redrawn. They retain the original publication's license and credits.

Each study identifies its source paper and the parameters used. Original paper
figures, supplements and author repositories follow their own terms. Computed
arrays and original project figures are kept separately from source images and
composite comparisons, so you can reuse the calculation while citing its
scientific basis.

Use the DOI and source notes in a case to obtain reference material. When
sharing a figure containing an original panel, retain the attribution and check
the image's redistribution conditions. A source repository without a license
provides no general permission to redistribute its code.
