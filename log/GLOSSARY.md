# Glossary

- **2b**: proportions stage; MediaPipe bone directions, standard adult lengths (H = trunk / 0.288).
- **2c / fit2d**: per-clip solve so the skeleton projects onto MediaPipe 2D, with Depth Pro relative depth.
- **captured / tracked**: joint was seen (visible or occluded-in-frame) in at least one frame; never guessed.
- **occluded**: low visibility, presence ≥ 0.5, inside the frame (3 % margin): hidden in frame, MediaPipe's guess kept.
- **inferred**: filled by rules (`core/infer_hidden.py`), never treated as capture; excluded from metrics.
- **H**: body height in metres.
- **floor_R**: rotation camera → Z-up world (X right, Y forward).
- **MPJPE**: mean per-joint position error after aligning pelvises.
- **soft-L1**: robust cost 2(√(1+r²) − 1); quadratic near 0, linear for outliers.
- **Depth Pro**: Apple's single-image metric depth + focal length model.
- **static / moving camera**: background moves < / ≥ 0.5 px per frame.
