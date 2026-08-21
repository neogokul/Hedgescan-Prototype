# Pixel annotations

This branch holds hand-corrected hedge/not-hedge masks produced by the
"Pixel annotation" tab in `hedgescan_prototype.py` (on the
`claude/hedgescan-phase-0-prototype-lyfmww` branch).

## Workflow

1. Open the app (either the hosted URL, or run locally from the code
   branch) — it opens on the **Annotate** tab. Point "Dataset folder" at
   `Dataset` (or wherever you've checked out that branch's photos).
2. The system shows its current hedge/not-hedge call: green = anything it
   thinks looks like foliage (a rough guess, not specifically the surveyed
   hedgerow — see the in-app caption), red = everything else.
3. Paint over ALL of it with your own judgement: green = the specific
   hedgerow being surveyed, red = everything else (other trees, other
   plants, grass, flowers, people, gaps, background), then click "Save
   annotation for this image". If the app is hosted with GitHub configured,
   this pushes straight to this branch; otherwise (or if it fails) it saves
   locally / offers a "Download annotation .txt" button as a fallback.
4. Check the **Progress** tab any time to see how many photos are
   done/remaining across everyone.
5. To add more source photos, use the "Upload new photos" section on the
   Annotate tab — they're auto-numbered continuing from whatever's already
   in the dataset.

The "Show developer tools" checkbox in the sidebar reveals earlier
single-image/testing tools used while building the detection algorithm —
not part of the annotation workflow, safe to ignore.

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

## Where to source new photos from

**Do not use Google Street View screenshots.** Street View imagery is
Google's copyrighted content, and Google's terms of service prohibit
downloading/storing/redistributing it outside their own viewer — this
repo's `Dataset` branch is public, so adding Street View screenshots there
would be republishing copyrighted material without a license. It also has
real quality downsides for this project specifically: it's shot through a
360° camera rig on a car, so even "normal-looking" crops carry projection
distortion, motion blur, and inconsistent exposure, and there's no control
over distance/angle to the hedge — all of which would especially hurt the
height-estimation logic (crop_to_hedge_band / estimate_height_from_reference
both assume a normal camera perspective).

Recommended sources instead, roughly best to worst:

1. **Real photos from someone on the ground** (a friend, a contact) — best
   quality, and no licensing question at all. This is how the first 3
   photos in this dataset were sourced.
2. **[Mapillary](https://www.mapillary.com/)** — a crowd-sourced,
   street-level imagery platform similar to Street View but released under
   an open license (CC BY-SA) that's actually meant to be reused and
   redistributed. Has usable UK coverage; the practical fallback for
   annotators not near the UK.
3. **Wikimedia Commons** — free-licensed photos; slower to find enough
   hedge-specific images, but occasionally turns up good ones.
