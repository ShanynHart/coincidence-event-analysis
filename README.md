# Coincidence event analysis

Event building, classification, and cross-channel time alignment for multi-detector gamma-ray systems, from my PhD (LaBr3:Ce + POLARIS CZT prompt-gamma imaging) and postdoc (LaBr3:Ce Compton camera) work at the University of Cape Town and iThemba LABS.

Multi-detector experiments produce independent hit streams per channel. The analysis task is to align the channels in time, pair hits that belong to the same physical event, and classify pairs by their interaction physics, at millions of events per run.

## What is here

**`cpp_sorters/`** (ROOT/C++): coincidence sorters for an 8-channel Compton camera. `CCsort.C` pairs scatterer and absorber hits inside a timing window and applies a kinematic test (whether the energy pair is consistent with a physical Compton scattering angle, i.e. |cos(theta)| < 1) to reject unphysical orderings. `sort_comptonEvents.C` and `sort_CCspecific.C` build the event classes used downstream; `plot_event_classes.C` shows their separation.

**`event_classification/`** (Python + Cython): `CLF_LaBr3POLARIS.py` classifies coincidence events between a LaBr3:Ce detector and a position-sensitive CZT detector, working on merged parquet dataframes with pandas. The hot inner loops live in `CLF_LaBr3POLARIS_utils.pyx`, compiled with Cython (`setup.py build_ext --inplace`) after profiling showed pure pandas was the bottleneck.

**`time_sync/`** (Python): `timeSyncDetectors_beam.py` aligns detector timestamps against the accelerator RF reference and each other, fitting per-channel time offsets and drift so that a single coincidence window works across all channels. `timeDelayLaBrPOLARIS.py` measures the fixed cable and electronics delays.

## Methods, in general terms

Stream alignment and windowed joins on timestamped data, physics-constrained event filtering, performance work (profiling, Cython for hot loops), and validation against known source geometries.

## Author

Shanyn Hart. All code in this repository is my own.
