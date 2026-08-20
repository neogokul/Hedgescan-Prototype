# Pixel annotations

This branch holds hand-corrected hedge/not-hedge masks produced by the
"Pixel annotation" tab in `hedgescan_prototype.py` (on the
`claude/hedgescan-phase-0-prototype-lyfmww` branch).

## Workflow

1. On the code branch, run the app locally and open the **Pixel annotation**
   tab. Point "Dataset folder" at wherever you've checked out the `Dataset`
   branch's photos (`1.jpg` .. `19.jpg`).
2. The system shows its current hedge/not-hedge call: green = hedge
   material, red = everything else (gaps, other trees, flowers, people,
   background).
3. Paint over anything it got wrong, then click "Save annotation for this
   image". This writes `annotation_data/<n>.txt` locally.
4. Push the resulting `annotation_data/` folder to this branch (or hand the
   files back for it to be committed here).

## File format

Each `<n>.txt` is plain text:

```
image: 1.jpg
width: 850
height: 567
# 1=hedge material, 0=not hedge; each line below is one row, run-length encoded
1:63,0:79,1:8,...
1:63,0:79,1:8,...
...
```

- One header block (`image`, `width`, `height`, a comment line), then
  exactly `height` more lines — one per image row.
- Each row line is run-length encoded as `label:count,label:count,...` in
  reading order (left to right): `1` = hedge material, `0` = not hedge.
- `hedgescan_prototype.py` has a matching reader,
  `rle_lines_to_label_grid(text) -> np.ndarray[bool]`, for loading these
  back for evaluation or future model training.
