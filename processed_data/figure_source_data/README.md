# Figure source data

`fig_manifest.csv` maps each figure to the script that draws it and to the result
tables that script reads. The granularity is the *script*, not the panel: several
panels in one figure are drawn from several tables, and a script that draws three
figures (e.g. `24_figures_robust.py` for fig12a-c) necessarily lists the same
tables three times. Rows marked `(computed in-script)` are figures whose inputs are
objects built earlier in the same script rather than a file on disk.

Every path in the manifest is also present in the freeze, so the list can be walked
programmatically without hitting a missing file.
