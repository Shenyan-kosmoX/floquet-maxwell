# Published Fig. 4 crop and layout

The unchanged five-panel published figure is in `paper_figure4.png`; provenance, page rectangle and SHA-256 hashes are in `paper_figure4_manifest.json`. Rebuild with `extract_paper_figure.py`. This crop includes the original legend and excludes the article prose and caption. Source: Garg et al., Optics Express 30, 45832–45847 (2022), Fig. 4, DOI 10.1364/OE.476035, PDF page 11 / printed page 45842.

`layout_contract.json` supplies all five data-frame rectangles, label/title extents, legend rectangle, Matplotlib axes fractions, approximate source value ranges, exact dominant line colors, and observable definitions. The low x endpoint is only raster-inferred: a is about 0.016; b/c about 0.02. Use 0.02–1.0 consistently for comparison and retain this uncertainty rather than claim an exact original minimum.

The plot quantity in c is 1−R−T, including exchange with the external modulation. It must not be described as pure material absorption. Panels d/e use the carrier magnetic/electric dipole coefficients with l=1,m=−1. Their relative phase convention is testable at the first static Huygens point: magnitude ratio approximately 1, phase difference approximately −π.

For comparison, retain the full five-panel block for both paper and calculation at equal visual scale. Do not hide uncomputed ranges by stretching a local reproduction. Source curves are raster evidence, not numerical truth or a substitute for convergence tests.

The auxiliary 2017 Fig. 3 reproduction retains the correct a–f ordering. Existing e/f add negative-sideband curves absent from the paper; existing independent-reference markers are validation overlays. Those additions need explicit labels in any paper comparison, or separate validation figures.
