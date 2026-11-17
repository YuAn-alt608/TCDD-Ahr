# Script index

Scripts are kept flat and numbered in execution order. Every file starts with a
docstring saying *why* it exists, not only what it does; the first line of that
docstring is reproduced here.

| Module | Range | Purpose | Scripts |
|---|---|---|---|
| `01_data_acquisition` | 00-04 | GEO retrieval, annotation build, per-library processing | see below |
| `02_smc_heterochromatin` | 05-11 | lineage ATAC metrics, heterochromatin domains, motif dissection | see below |
| `03_cascade_and_te` | 12-17 | cell-cell communication, transposable elements, in-silico validation, trajectory | see below |
| `04_pseudobulk_and_null` | 18-21 | depth downsampling, per-animal pseudobulk, domain permutation | see below |
| `05_external_ahr_mapping` | 22-28 | external AHR ChIP, liftOver, human coronary projection, primer design | see below |
| `06_confounding_audit` | 29-34 | composition decomposition, depth matching, cell subsampling, Tier C gates | see below |
| `07_occupancy_and_replication` | 35-45 | occupancy specificity, claim audit, donor-stratified replication, Table S1 | see below |


## 01_data_acquisition — GEO retrieval, annotation build, per-library processing

- `00_geo_search.py` — GEO dataset discovery via NCBI eutils (no API key -> keep <=3 req/s).
- `01_geo_inspect.py` — Inspect GEO series: download family SOFT, parse supplementary files + sample titles, HEAD sizes.
- `02_download.py` — Download all raw inputs with resume + manifest. Run with --only <tag>.
- `03_build_annotation.py` — Build mm10 heterochromatin / repeat annotation resources.
- `04_process.py` — Process one dataset: QC -> cluster -> annotate lineage -> compute 4 heterochromatin fingerprints.
- `04b_recompute.py` — Recompute per-cell fingerprints on already clustered datasets (no re-clustering).
- `04c_refine_D2.py` — Refine D2 cell-type labels using the genetic lineage label (Myh11-CreERT2;Rosa-tdT).

## 02_smc_heterochromatin — lineage ATAC metrics, heterochromatin domains, motif dissection

- `05_atac.py` — scATAC-seq layer (GSE276870, paired with GSE276872).
- `05a_atac_activity.py` — Cache the peak->gene activity matrix and check marker behaviour before labelling.
- `05b_atac_annot.py` — Re-annotate the scATAC object from the cached gene-activity matrix.
- `06_matrix.py` — Build the exposure x lineage interaction matrix with effect sizes, timing (T50) and meta-comparison.
- `07_perturb.py` — In silico perturbation of epigenetic switches on TCDD-exposed coronary cells.
- `07b_switch_evidence.py` — Epigenetic-switch reversibility evidence without a deep generative model.
- `08_genetics.py` — Layer 3: human genetic evidence for the epigenetic switch panel (GWAS Catalog v2 + GTEx eQTL).
- `09_figures.py` — Figures for the Age3 report.
- `10_report.py` — Assemble the restructured manuscript: core discovery (D2 + ATAC), boundary
- `11_motif.py` — Motif-level dissection of the two arms (AhR-dependent vs AhR-independent).
- `11b_het_promoter_motif.py` — Which motifs sit on the heterochromatin-gene promoters that actually open?

## 03_cascade_and_te — cell-cell communication, transposable elements, in-silico validation, trajectory

- `12_ccc.py` — Cell-cell communication: does SMC-specific Ahr deletion pull the niche back via secreted factors?
- `13_te.py` — Transposable-element layer.
- `14_insilico_validation.py` — Three lightweight *in silico* replications of the wet-lab experiments that would
- `15_trajectory.py` — Per-lineage diffusion pseudotime of barrier collapse in D2 (0 / 8 / 16 weeks).
- `16_te_kmers.py` — Family-level transposable-element quantification directly from FASTQ, without an aligner.
- `17_figures_story.py` — Story figures for the restructured manuscript: model cascade, validation panels, TE.

## 04_pseudobulk_and_null — depth downsampling, per-animal pseudobulk, domain permutation

- `18_atac_downsample.py` — Depth-sensitivity of the scATAC claims (Supplementary Fig. S1a-S1c).
- `18b_downsample_summary.py` — Turn the downsampling tables into the numbers quoted in the report.
- `19_pseudobulk.py` — Pseudobulk (animal-level) re-analysis of the two headline claims.
- `20_permutation.py` — Is the heterochromatin-domain effect an artefact of how domains are defined?
- `21_lr_permutation.py` — Does the ligand-receptor transfer model need the real network topology?

## 05_external_ahr_mapping — external AHR ChIP, liftOver, human coronary projection, primer design

- `22_ahr_chip.py` — Is the SMC opening at real AHR binding sites, or only at AHR motifs?
- `22b_ahr_liftover.py` — Base-pair overlap between the ATAC bins and real AHR ChIP-seq peaks.
- `23_human_coronary.py` — Do the mouse-defined programmes exist as coherent subsets in human coronary?
- `24_figures_robust.py` — Figures for the pre-submission verification block.
- `25_power_n2.py` — How much does one more animal actually buy?
- `25b_n2_analysis.py` — The frozen decision procedure for the "one more animal" question.
- `26_ahr_gene_level.py` — Item 2 of the pre-submission list, done at gene resolution instead of 5 kb bins.
- `27_qpcr_primers.py` — Turn the candidate list into something a bench can actually order.
- `28_figures_plan.py` — Figures for the two follow-ups: power of the n=2 upgrade, and what the public ChIP

## 06_confounding_audit — composition decomposition, depth matching, cell subsampling, Tier C gates

- `29_composition_audit.py` — Composition audit: is the shift a change of cell state or a change of census?
- `30_depth_matched.py` — Matched-depth downsampling: do the four frozen endpoints survive when every
- `31_cell_subsampling.py` — Cell-subsampling and outlier robustness of the four frozen endpoints.
- `32_alt_chip.py` — A replacement AHR ChIP-seq / CUT&Tag set, admitted only if it passes Tier C.
- `32b_bigwig_tierC.py` — Tier C gate applied to datasets that deposit only bigWig.
- `32c_summit_tierC.py` — Tier C with a summit test instead of a window mean.
- `33_primer_specificity.py` — Genome-wide specificity of the 42 ChIP-qPCR primer pairs (local in-silico PCR).
- `33b_primer_redesign.py` — Replace the primer pairs that came out multi-hit.
- `34_figures_confound.py` — Figures for the confounding audit (section 10.7 / 10.8).

## 07_occupancy_and_replication — occupancy specificity, claim audit, donor-stratified replication, Table S1

- `35_occupancy_specificity.py` — Close ring 5 with a dry experiment: is AHR occupancy ligand- and
- `36_figures_occupancy.py` — Figure for the ring-5 dry experiment (report section 10.11).
- `37_claim_audit.py` — Claim audit: every number the story line quotes, recomputed from the
- `39_external_replication.py` — External replication in a dataset that is not ours.
- `40_human_qc_pass.py` — One streaming pass over GSE131778 for the things R3 needs but 23 threw away.
- `41_human_branch_stats.py` — Does the branch point survive donor stratification and technical covariates?
- `42_stromal_continuum.py` — The stromal continuum: why the Matrix panel "loses" its SMC label in human.
- `43_power_floor.py` — What a p-value can and cannot mean at 3 vs 3.
- `44_figures_strengthen.py` — Figures for the three strengthening analyses (report section 16).
- `45_table_s1.py` — Supplementary Table 1: the 21-claim audit matrix, bilingual.

## Execution order

```bash
python scripts/09_figures.py            # main figures from processed results
python scripts/37_claim_audit.py        # the 21-claim audit (Table S1 source)
python scripts/39_external_replication.py
python scripts/41_human_branch_stats.py
python scripts/42_stromal_continuum.py
python scripts/43_power_floor.py
python scripts/45_table_s1.py           # Supplementary Table 1 (.xlsx)
python scripts/10_report.py             # rebuild report/Age3_report.html
```
