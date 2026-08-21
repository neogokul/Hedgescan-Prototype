"""
HedgeScan Phase 0 vision prototype.

Validates the computer-vision approach for automating hedgerow condition
surveys against the UK Defra Biodiversity Net Gain Metric 4.0 structural
attributes, before any mobile development starts. This is a Streamlit
research tool, not the mobile app.

Automated attributes (structural group only):
- A1 Height:   pass if average height > 1.5 m (base of stem to top of
               shoots), excluding gaps and isolated trees
- A2 Width:    pass if average canopy width > 1.5 m at the widest point,
               excluding gaps
- B1 Basal gap: pass if the gap between ground and canopy base is < 0.5 m
               for > 90% of hedge length
- B2 Gappiness: pass if gaps make up < 10% of total hedge length AND no
               single canopy gap exceeds 5 m
- D2 Damage:   pass if > 90% of hedge/margin is free of human-caused damage

B1 and B2 must be aggregated across a full 30 m survey section, not scored
per image. A1/A2 must exclude gap regions from the calculation. This
prototype currently automates optical porosity (feeding B2) and a hedge
height estimate (feeding A1) from HSV color thresholding.
Width, basal gap and damage are not yet implemented.
"""

import io
import json
from dataclasses import dataclass, field
from math import radians, tan
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import streamlit.elements.image as _st_image

# streamlit-drawable-canvas 0.9.3 (last released 2022) calls
# streamlit.elements.image.image_to_url, which newer Streamlit (this app
# targets current Streamlit, not a pinned old one) moved to
# streamlit.elements.lib.image_utils, and its signature changed from a
# plain pixel-width int to a LayoutConfig object. Shim the old call shape
# back in, rather than pin an old Streamlit just for this one dependency.
if not hasattr(_st_image, "image_to_url"):
    from streamlit.elements.lib.image_utils import image_to_url as _modern_image_to_url
    from streamlit.elements.lib.layout_utils import LayoutConfig as _LayoutConfig

    def _image_to_url_shim(image, width, clamp, channels, output_format, image_id):
        return _modern_image_to_url(
            image, _LayoutConfig(width=width), clamp, channels, output_format, image_id
        )

    _st_image.image_to_url = _image_to_url_shim

from streamlit_drawable_canvas import st_canvas

# ---------------------------------------------------------------------------
# Defra Metric 4.0 structural thresholds
# ---------------------------------------------------------------------------
A1_HEIGHT_THRESHOLD_M = 1.5
B2_GAPPINESS_THRESHOLD_PCT = 10.0
B2_MAX_SINGLE_GAP_M = 5.0
SECTION_LENGTH_M = 30.0

# ---------------------------------------------------------------------------
# Gap segmentation (classical HSV thresholding, no ML model)
# ---------------------------------------------------------------------------
# A "gap" is anything that ISN'T hedge material — foliage (any shade of
# green) or woody branch/stem (brown/grey, low-mid brightness). Whatever
# shows through a gap in the hedge — sky, a wall, a building, another plant
# — is out of the classifier's control, so gap detection works by
# recognising hedge material and calling everything else a gap, rather than
# trying to enumerate every possible background.
#
# Tuned against a real head-on hedgerow photo (hedgerow-trim-16896461364.jpg)
# whose background is a grey wall, not sky: sampled foliage pixels were
# H 48-95 (green) with S >= 25 when lit, or very dark (V < 30, deep interior
# shadow — treated as hedge, not a gap, since it's occlusion by the hedge's
# own canopy rather than an opening). The wall itself sampled at H~26 S~12
# V~166 — outside both the green-hue and dark-shadow ranges, so it correctly
# falls out as gap. Earlier sky-only thresholds (see git history) would have
# missed this wall entirely.
HSV_FOLIAGE_HUE_MIN = 35
HSV_FOLIAGE_HUE_MAX = 100
HSV_FOLIAGE_SAT_MIN = 25
HSV_BRANCH_HUE_MIN = 5
HSV_BRANCH_HUE_MAX = 30
HSV_BRANCH_SAT_MIN = 25
HSV_BRANCH_SAT_MAX = 160
HSV_BRANCH_VAL_MAX = 130
HSV_SHADOW_VAL_MAX = 30

# Non-hedge vegetation and people are not gaps either — a flower, a rose
# bush, or a person standing in front of the hedge blocks the view of
# whatever is behind them, same as foliage does; they just aren't hedge
# material. Excluding them keeps them out of the porosity count instead of
# miscounting them as an opening.
# Tuned against hedgerow-yard-1689572455874.jpg and
# planting-a-hedge-row-1689706733107.jpg: rose petals sampled at H~164-165,
# S 53-163, V 191-210 (pink/magenta); an orange-leaved shrub sampled at
# H~33-38, S~229-233, V~128-184 (vivid warm color, much more saturated than
# the brown_branch range above, which is deliberately desaturated to avoid
# swallowing exactly this kind of non-hedge shrub). Skin tone is approximated
# with the standard H 0-20, S 30-150, V 80-235 heuristic range used broadly
# in CV skin-detection work — an approximation, not verified against this
# specific photo, since we don't sample identifiable people pixel-by-pixel.
HSV_PINK_MAGENTA_HUE_MIN = 140
HSV_PINK_MAGENTA_SAT_MIN = 40
HSV_PINK_MAGENTA_VAL_MIN = 140
HSV_VIVID_WARM_HUE_MAX = 40
HSV_VIVID_WARM_SAT_MIN = 165
HSV_SKIN_HUE_MAX = 20
HSV_SKIN_SAT_MIN = 30
HSV_SKIN_SAT_MAX = 150
HSV_SKIN_VAL_MIN = 80
HSV_SKIN_VAL_MAX = 235


def segment_gap_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask, True where the pixel is classified as a gap
    (anything that is not hedge material, non-hedge vegetation, or a
    person/skin-toned obstruction).
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    green_foliage = (
        (h >= HSV_FOLIAGE_HUE_MIN) & (h <= HSV_FOLIAGE_HUE_MAX) & (s >= HSV_FOLIAGE_SAT_MIN)
    )
    brown_branch = (
        (h >= HSV_BRANCH_HUE_MIN)
        & (h <= HSV_BRANCH_HUE_MAX)
        & (s >= HSV_BRANCH_SAT_MIN)
        & (s <= HSV_BRANCH_SAT_MAX)
        & (v <= HSV_BRANCH_VAL_MAX)
    )
    deep_shadow = v <= HSV_SHADOW_VAL_MAX
    pink_magenta_flower = (
        (h >= HSV_PINK_MAGENTA_HUE_MIN) & (s >= HSV_PINK_MAGENTA_SAT_MIN) & (v >= HSV_PINK_MAGENTA_VAL_MIN)
    )
    vivid_warm_flower = (h <= HSV_VIVID_WARM_HUE_MAX) & (s >= HSV_VIVID_WARM_SAT_MIN)
    skin_tone = (
        (h <= HSV_SKIN_HUE_MAX)
        & (s >= HSV_SKIN_SAT_MIN)
        & (s <= HSV_SKIN_SAT_MAX)
        & (v >= HSV_SKIN_VAL_MIN)
        & (v <= HSV_SKIN_VAL_MAX)
    )

    is_hedge_or_excluded = (
        green_foliage | brown_branch | deep_shadow | pink_magenta_flower | vivid_warm_flower | skin_tone
    )
    return ~is_hedge_or_excluded


def skin_tone_fraction_pct(image_bgr: np.ndarray) -> float:
    """
    Percentage of the frame that is skin-toned.

    Skin tone is excluded from the gap mask (a person standing in front of
    the hedge isn't an opening), but that means a frame dominated by
    people rather than hedge would score close to 0% porosity — a
    misleadingly "perfect" hedge, not a real measurement. Surfacing this
    fraction lets the UI warn when that's likely what happened, rather
    than silently reporting a clean pass.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    skin_tone = (
        (h <= HSV_SKIN_HUE_MAX)
        & (s >= HSV_SKIN_SAT_MIN)
        & (s <= HSV_SKIN_SAT_MAX)
        & (v >= HSV_SKIN_VAL_MIN)
        & (v <= HSV_SKIN_VAL_MAX)
    )
    return 100.0 * float(np.count_nonzero(skin_tone)) / skin_tone.size


SKIN_DOMINANT_WARNING_PCT = 15.0


def optical_porosity_pct(mask: np.ndarray) -> float:
    """Percentage of the frame classified as gap."""
    return 100.0 * float(np.count_nonzero(mask)) / mask.size


def hedge_silhouette_porosity_pct(mask: np.ndarray) -> float:
    """
    Percentage of gap pixels, counted only within each column's hedge
    silhouette (from its topmost to bottommost non-gap pixel) rather than
    across the whole frame.

    Open sky above the hedge crown, or grass/pavement below its base, is a
    gap by color but isn't part of the hedge at all — it shouldn't inflate
    porosity just because it's visible in the same photo. Restricting the
    count to each column's own hedge extent excludes both automatically,
    without needing a manual hedge-band crop for a hedge with a clean,
    roughly horizontal top edge (crop_to_hedge_band is still the tool for a
    hedge sharing columns with taller background trees/hedges).
    """
    foliage = ~mask
    total_in_silhouette = 0
    gap_in_silhouette = 0
    for col in range(mask.shape[1]):
        rows = np.flatnonzero(foliage[:, col])
        if rows.size == 0:
            continue
        top, bottom = rows[0], rows[-1]
        total_in_silhouette += bottom - top + 1
        gap_in_silhouette += int(np.count_nonzero(mask[top : bottom + 1, col]))
    if total_in_silhouette == 0:
        return 0.0
    return 100.0 * gap_in_silhouette / total_in_silhouette


# ---------------------------------------------------------------------------
# Hedge band selection (separating the hedgerow from background trees)
# ---------------------------------------------------------------------------
def crop_to_hedge_band(image_bgr: np.ndarray, top_frac: float, bottom_frac: float) -> np.ndarray:
    """
    Crop the frame to the row band that contains the hedgerow itself.

    A photo can show background trees/hedges rising above the hedgerow
    being surveyed — same green color, so color thresholding alone cannot
    tell them apart. There is no depth cue in a single 2D photo to separate
    "this hedge" from "that hedge/tree behind it" automatically; the
    surveyor marks the row band where the hedgerow being assessed actually
    sits (e.g. top_frac=0.25 to exclude the top quarter of the frame if
    that's background trees), and only that band feeds porosity/height.
    """
    height_px = image_bgr.shape[0]
    top_row = max(0, min(height_px, int(round(top_frac * height_px))))
    bottom_row = max(top_row, min(height_px, int(round(bottom_frac * height_px))))
    return image_bgr[top_row:bottom_row, :]


# ---------------------------------------------------------------------------
# Height estimation
# ---------------------------------------------------------------------------
def find_hedge_top_row(mask: np.ndarray) -> int:
    """
    Median row index of the topmost non-gap pixel in each column, within
    whatever band of the frame the mask covers.

    Using the median (rather than the single highest point) avoids letting
    a spike in the mask dominate the height estimate. Run this on a mask
    from crop_to_hedge_band, not the full frame, when background trees are
    present — otherwise the topmost non-gap pixel is background canopy, not
    the hedgerow's own top edge.
    """
    foliage = ~mask
    top_rows = []
    for col in range(foliage.shape[1]):
        rows = np.flatnonzero(foliage[:, col])
        if rows.size:
            top_rows.append(rows[0])
    if not top_rows:
        return foliage.shape[0]  # no foliage found; treat as ground level
    return int(np.median(top_rows))


def estimate_height_m(
    top_row: int,
    frame_height_px: int,
    distance_m: float,
    camera_height_m: float,
    tilt_deg: float,
    vertical_fov_deg: float,
) -> float:
    """
    Estimate hedge height from the top-edge pixel row using pinhole-camera
    trigonometry: the top edge subtends an angle above the camera's optical
    axis proportional to its offset from the frame's vertical center, offset
    by the camera's tilt, and converted to a vertical rise over the known
    horizontal distance.

    Requires an accurate distance/FOV estimate, which is often just guessed
    in the field — see estimate_height_from_reference for a more reliable
    alternative when a person or other known-height object is in frame.
    """
    if not 0 <= top_row <= frame_height_px:
        raise ValueError(
            f"top_row ({top_row}) is outside the frame (0-{frame_height_px})"
        )
    frac_from_center = (frame_height_px / 2.0 - top_row) / (frame_height_px / 2.0)
    angle_from_axis_deg = frac_from_center * (vertical_fov_deg / 2.0)
    angle_above_horizontal_deg = tilt_deg + angle_from_axis_deg
    rise_m = distance_m * tan(radians(angle_above_horizontal_deg))
    return camera_height_m + rise_m


def estimate_height_from_reference(
    hedge_top_row_px: float,
    hedge_base_row_px: float,
    reference_top_row_px: float,
    reference_base_row_px: float,
    reference_height_m: float,
) -> float:
    """
    Estimate hedge height by scaling against a reference object of known
    height (typically a person standing at roughly the same distance from
    the camera as the hedge). This avoids needing to guess distance/FOV
    values in the field: pixel height of the reference object gives a
    pixels-per-metre scale, which converts the hedge's own pixel height
    directly to metres.

    Assumes the reference object and the hedge face are at approximately
    the same distance from the camera — if the person is standing well in
    front of or behind the hedge, perspective will bias the result.
    """
    reference_height_px = abs(reference_base_row_px - reference_top_row_px)
    if reference_height_px <= 0:
        raise ValueError("Reference object must have positive pixel height")
    px_per_metre = reference_height_px / reference_height_m
    hedge_height_px = abs(hedge_base_row_px - hedge_top_row_px)
    return hedge_height_px / px_per_metre


# ---------------------------------------------------------------------------
# Section-level aggregation (B2 gappiness across a 30 m survey section)
# ---------------------------------------------------------------------------
@dataclass
class ImagePorosityResult:
    """Per-image porosity result plus the gap runs used for B2's single-gap check."""

    filename: str
    porosity_pct: float
    gap_run_lengths_m: list = field(default_factory=list)


@dataclass
class SectionGappinessResult:
    """Aggregated B2 gappiness result for a full survey section."""

    section_length_m: float
    total_gap_length_m: float
    gappiness_pct: float
    largest_single_gap_m: float
    image_results: list
    gappiness_pass: bool
    single_gap_pass: bool

    @property
    def b2_pass(self) -> bool:
        return self.gappiness_pass and self.single_gap_pass


def _gap_run_lengths_m(mask: np.ndarray, image_length_m: float) -> list:
    """
    Column-wise gap run lengths (in metres) along the hedge, derived from a
    gap mask. A column counts as "gap" if the majority of pixels within
    that column's own hedge silhouette (its topmost-to-bottommost non-gap
    pixel) are gap, or if the column has no hedge material at all (a
    genuine full-height opening straight through the hedge). Restricting to
    the silhouette keeps open sky above the hedge crown, or ground below
    its base, from padding out a column's gap fraction. Consecutive gap
    columns form a single run, converted to metres by the image's
    along-hedge ground length.
    """
    width_px = mask.shape[1]
    if width_px == 0:
        return []

    foliage = ~mask
    column_is_gap = np.zeros(width_px, dtype=bool)
    for col in range(width_px):
        rows = np.flatnonzero(foliage[:, col])
        if rows.size == 0:
            column_is_gap[col] = True  # no hedge material anywhere in this column
        else:
            top, bottom = rows[0], rows[-1]
            column_is_gap[col] = mask[top : bottom + 1, col].mean() > 0.5
    m_per_px = image_length_m / width_px

    runs = []
    run_len_px = 0
    for is_gap in column_is_gap:
        if is_gap:
            run_len_px += 1
        elif run_len_px:
            runs.append(run_len_px * m_per_px)
            run_len_px = 0
    if run_len_px:
        runs.append(run_len_px * m_per_px)
    return runs


def analyze_image_for_section(
    image_bgr: np.ndarray, filename: str, image_length_m: float
) -> ImagePorosityResult:
    """Run porosity/gap-run detection on a single image within a section."""
    mask = segment_gap_mask(image_bgr)
    porosity = hedge_silhouette_porosity_pct(mask)
    gap_runs = _gap_run_lengths_m(mask, image_length_m)
    return ImagePorosityResult(
        filename=filename, porosity_pct=porosity, gap_run_lengths_m=gap_runs
    )


def aggregate_section_gappiness(
    image_results: list, section_length_m: float = SECTION_LENGTH_M
) -> SectionGappinessResult:
    """
    Aggregate per-image porosity/gap-run results across a full survey
    section to score the true B2 criterion:
      - total gap length must be < 10% of section length, AND
      - no single canopy gap run may exceed 5 m.

    Gap runs are tracked per image (not just a percentage) because a run
    that is split across the boundary between two consecutive images would
    otherwise be undercounted; each image's runs are pooled as independent
    runs, which is a conservative approximation until image overlap/stitch
    metadata is available to merge cross-boundary runs.
    """
    all_gap_runs = [
        run for result in image_results for run in result.gap_run_lengths_m
    ]
    total_gap_length_m = sum(all_gap_runs)
    gappiness_pct = (
        100.0 * total_gap_length_m / section_length_m if section_length_m else 0.0
    )
    largest_single_gap_m = max(all_gap_runs, default=0.0)

    gappiness_pass = gappiness_pct < B2_GAPPINESS_THRESHOLD_PCT
    single_gap_pass = largest_single_gap_m <= B2_MAX_SINGLE_GAP_M

    return SectionGappinessResult(
        section_length_m=section_length_m,
        total_gap_length_m=total_gap_length_m,
        gappiness_pct=gappiness_pct,
        largest_single_gap_m=largest_single_gap_m,
        image_results=image_results,
        gappiness_pass=gappiness_pass,
        single_gap_pass=single_gap_pass,
    )


def analyze_section_folder(
    filenames_and_images: list, section_length_m: float = SECTION_LENGTH_M
) -> SectionGappinessResult:
    """
    Run B2 gappiness aggregation over a folder of images representing one
    30 m survey section.

    filenames_and_images: list of (filename, image_bgr) tuples, ordered
    along the hedge. Each image is assumed to cover an equal share of the
    section length.
    """
    n = len(filenames_and_images)
    if n == 0:
        return aggregate_section_gappiness([], section_length_m)

    image_length_m = section_length_m / n
    image_results = [
        analyze_image_for_section(image_bgr, filename, image_length_m)
        for filename, image_bgr in filenames_and_images
    ]
    return aggregate_section_gappiness(image_results, section_length_m)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def _load_image_bgr(uploaded_file) -> np.ndarray:
    file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def render_single_image_tab():
    st.subheader("Single image assessment")
    st.caption(
        "Scores one photo against A1 height and B2 optical porosity. "
        "For a real B2 verdict, use the Section batch tab to aggregate "
        "across a full 30 m section."
    )

    uploaded_file = st.file_uploader(
        "Hedge photo", type=["jpg", "jpeg", "png"], key="single_upload"
    )

    if uploaded_file is None:
        return

    image_bgr = _load_image_bgr(uploaded_file)
    frame_height_px = image_bgr.shape[0]

    st.markdown("**Hedge band** (exclude background trees/hedges rising above the frame)")
    top_frac, bottom_frac = st.slider(
        "Row band containing the hedgerow (% of frame height, top to bottom)",
        min_value=0,
        max_value=100,
        value=(0, 100),
        step=1,
        key="hedge_band",
    )
    hedge_band_bgr = crop_to_hedge_band(image_bgr, top_frac / 100.0, bottom_frac / 100.0)
    band_top_row_px = int(round(top_frac / 100.0 * frame_height_px))

    if hedge_band_bgr.shape[0] == 0:
        st.error("Hedge band is zero height — drag the two slider handles apart.")
        return

    mask_full = segment_gap_mask(image_bgr)
    mask_band = segment_gap_mask(hedge_band_bgr)
    porosity = hedge_silhouette_porosity_pct(mask_band)
    top_row_in_band = find_hedge_top_row(mask_band)
    hedge_top_row_px = band_top_row_px + top_row_in_band
    hedge_base_row_px = band_top_row_px + hedge_band_bgr.shape[0]

    overlay = image_bgr.copy()
    overlay[mask_full] = (255, 0, 255)
    blended = cv2.addWeighted(image_bgr, 0.6, overlay, 0.4, 0)
    cv2.rectangle(
        blended, (0, band_top_row_px), (image_bgr.shape[1] - 1, hedge_base_row_px - 1), (0, 255, 255), 2
    )

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Original")
    with img_col2:
        st.image(
            cv2.cvtColor(blended, cv2.COLOR_BGR2RGB),
            caption="Gap mask overlay (magenta) with hedge band (yellow box)",
        )

    st.markdown("**Height estimation method**")
    height_method = st.radio(
        "Method",
        ["Reference object (recommended)", "Trigonometry (distance/FOV guess)"],
        key="height_method",
        label_visibility="collapsed",
    )

    if height_method == "Reference object (recommended)":
        st.caption(
            "Read pixel row numbers off the image above (top of image = row 0). "
            "Mark the reference object's top and base, and the hedge's top and base "
            "within the yellow hedge band."
        )
        ref_col1, ref_col2, ref_col3 = st.columns(3)
        with ref_col1:
            reference_height_m = st.number_input(
                "Reference object height (m)", value=1.83, min_value=0.1,
                help="e.g. a 6 ft (1.83 m) person standing at roughly the hedge's distance",
            )
        with ref_col2:
            reference_top_row_px = st.number_input(
                "Reference top row (px)", value=0, min_value=0, max_value=frame_height_px
            )
            reference_base_row_px = st.number_input(
                "Reference base row (px)", value=frame_height_px, min_value=0, max_value=frame_height_px
            )
        with ref_col3:
            hedge_top_row_input = st.number_input(
                "Hedge top row (px)", value=hedge_top_row_px, min_value=0, max_value=frame_height_px
            )
            hedge_base_row_input = st.number_input(
                "Hedge base row (px)", value=hedge_base_row_px, min_value=0, max_value=frame_height_px
            )
        try:
            height_m = estimate_height_from_reference(
                hedge_top_row_input, hedge_base_row_input,
                reference_top_row_px, reference_base_row_px,
                reference_height_m,
            )
        except ValueError as exc:
            st.error(str(exc))
            return
    else:
        col1, col2 = st.columns(2)
        with col1:
            distance_m = st.number_input("Distance to hedge (m)", value=5.0, min_value=0.1)
            camera_height_m = st.number_input("Camera height (m)", value=1.5, min_value=0.0)
        with col2:
            tilt_deg = st.number_input("Camera tilt angle (deg, up +)", value=0.0)
            vertical_fov_deg = st.number_input("Vertical field of view (deg)", value=55.0, min_value=1.0)
        height_m = estimate_height_m(
            top_row_in_band, hedge_band_bgr.shape[0], distance_m, camera_height_m, tilt_deg, vertical_fov_deg
        )

    height_pass = height_m > A1_HEIGHT_THRESHOLD_M
    porosity_pass = porosity < B2_GAPPINESS_THRESHOLD_PCT

    st.metric("A1 — Estimated height (m)", f"{height_m:.2f}", delta="PASS" if height_pass else "FAIL")
    st.metric("B2 — Optical porosity (%, single image, within hedge silhouette)", f"{porosity:.1f}", delta="PASS" if porosity_pass else "FAIL")
    st.caption(
        "This single-image porosity figure is indicative only — B2 requires "
        "aggregation across the full 30 m section (see Section batch tab)."
    )

    skin_pct = skin_tone_fraction_pct(hedge_band_bgr)
    if skin_pct >= SKIN_DOMINANT_WARNING_PCT:
        st.warning(
            f"{skin_pct:.0f}% of the hedge band is skin-toned. Skin is excluded from "
            "the gap count (a person isn't an opening), but a frame this dominated by "
            "a person means little hedge is actually visible — porosity here may be "
            "understated rather than genuinely low. Narrow the hedge band or use a "
            "photo with less of the person in frame."
        )


def render_section_batch_tab():
    st.subheader("Section batch (B2 gappiness aggregation)")
    st.caption(
        "Upload every photo covering one 30 m survey section, in order "
        "along the hedge. Porosity and gap-run detection run per image, "
        "then are aggregated to score the real B2 criterion: gaps < 10% "
        "of section length AND no single gap > 5 m."
    )

    section_length_m = st.number_input(
        "Section length (m)", value=SECTION_LENGTH_M, min_value=1.0
    )
    uploaded_files = st.file_uploader(
        "Section photos (in order along the hedge)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="section_upload",
    )

    if not uploaded_files:
        return

    filenames_and_images = [
        (f.name, _load_image_bgr(f)) for f in uploaded_files
    ]
    result = analyze_section_folder(filenames_and_images, section_length_m)

    st.metric(
        "B2 — Section gappiness (%)",
        f"{result.gappiness_pct:.1f}",
        delta="PASS" if result.gappiness_pass else "FAIL",
    )
    st.metric(
        "B2 — Largest single gap (m)",
        f"{result.largest_single_gap_m:.1f}",
        delta="PASS" if result.single_gap_pass else "FAIL",
    )
    st.metric("B2 — Overall", "PASS" if result.b2_pass else "FAIL")

    st.write("Per-image porosity:")
    st.table(
        {
            "filename": [r.filename for r in result.image_results],
            "porosity_pct": [round(r.porosity_pct, 1) for r in result.image_results],
            "gap_runs_m": [
                [round(g, 1) for g in r.gap_run_lengths_m] for r in result.image_results
            ],
        }
    )


# ---------------------------------------------------------------------------
# Labeling (build a corrected data pool from a folder of dataset images)
# ---------------------------------------------------------------------------
LABEL_DIR = Path("labels")
LABEL_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
LABEL_CANVAS_DISPLAY_WIDTH = 650


def _list_dataset_images(data_dir: Path) -> list:
    """Images in data_dir, sorted numerically by stem when the stem is a number
    (so 1.jpg, 2.jpg, ..., 100.jpg sort in the expected order), else by name."""
    if not data_dir.is_dir():
        return []

    def sort_key(path):
        try:
            return (0, int(path.stem))
        except ValueError:
            return (1, path.stem)

    return sorted(
        (p for p in data_dir.iterdir() if p.suffix.lower() in LABEL_IMAGE_EXTENSIONS),
        key=sort_key,
    )


def _circles_from_canvas_json(json_data, scale: float) -> list:
    """Extract drawn circles from a streamlit-drawable-canvas result, converted
    from displayed-canvas pixel coordinates back to original-image pixel
    coordinates (canvas is shown scaled down to fit the page)."""
    if not json_data:
        return []
    circles = []
    for obj in json_data.get("objects", []):
        if obj.get("type") != "circle":
            continue
        radius_px = obj.get("radius", 0) * obj.get("scaleX", 1)
        cx = (obj.get("left", 0) + radius_px) / scale
        cy = (obj.get("top", 0) + radius_px) / scale
        circles.append({"cx": round(cx, 1), "cy": round(cy, 1), "r": round(radius_px / scale, 1)})
    return circles


def render_labeling_tab():
    st.subheader("Labeling: correct the model's porosity calls")
    st.caption(
        "Step through a folder of dataset photos, see the current gap-mask "
        "overlay, then circle where the model missed a real gap or wrongly "
        "flagged one. Saved labels build a corrected data pool for evaluating "
        "or tuning the classifier later — they don't change the live thresholds."
    )

    data_dir = Path(st.text_input("Dataset folder (relative to the app)", value="data"))
    images = _list_dataset_images(data_dir)

    image_bgr = None
    image_name = None

    if images:
        LABEL_DIR.mkdir(exist_ok=True)
        labeled_stems = {p.stem for p in LABEL_DIR.glob("*.json")}
        st.caption(f"{len(labeled_stems)} / {len(images)} images labeled so far, saved under `{LABEL_DIR}/`.")

        if "label_index" not in st.session_state:
            st.session_state.label_index = 0
        st.session_state.label_index = max(0, min(len(images) - 1, st.session_state.label_index))

        nav1, nav2, nav3 = st.columns([1, 2, 1])
        with nav1:
            if st.button("< Previous", disabled=st.session_state.label_index == 0):
                st.session_state.label_index -= 1
        with nav3:
            if st.button("Next >", disabled=st.session_state.label_index >= len(images) - 1):
                st.session_state.label_index += 1
        with nav2:
            st.session_state.label_index = st.number_input(
                "Image index",
                min_value=0,
                max_value=len(images) - 1,
                value=st.session_state.label_index,
                step=1,
                label_visibility="collapsed",
            )

        image_path = images[st.session_state.label_index]
        image_name = image_path.name
        status = "already labeled" if image_path.stem in labeled_stems else "not yet labeled"
        st.markdown(f"**{image_name}** ({st.session_state.label_index + 1} / {len(images)}) — {status}")
        image_bgr = cv2.imread(str(image_path))
    else:
        st.info(
            f"No images found in `{data_dir}/`. Point this at a folder of photos "
            "(e.g. named 1.jpg .. 100.jpg), or upload one below to label it "
            "standalone."
        )
        uploaded = st.file_uploader("Or upload a single photo", type=["jpg", "jpeg", "png"], key="label_upload")
        if uploaded is not None:
            image_bgr = _load_image_bgr(uploaded)
            image_name = uploaded.name

    if image_bgr is None:
        return

    mask = segment_gap_mask(image_bgr)
    overlay = image_bgr.copy()
    overlay[mask] = (255, 0, 255)
    blended_rgb = cv2.cvtColor(cv2.addWeighted(image_bgr, 0.6, overlay, 0.4, 0), cv2.COLOR_BGR2RGB)

    orig_h, orig_w = blended_rgb.shape[:2]
    scale = LABEL_CANVAS_DISPLAY_WIDTH / orig_w
    display_h = int(round(orig_h * scale))
    pil_bg = Image.fromarray(blended_rgb).resize((LABEL_CANVAS_DISPLAY_WIDTH, display_h))

    st.markdown(
        "Magenta = model says gap. Draw a circle over each mistake: **red** "
        "where a real gap was missed, **cyan** where a gap was wrongly flagged."
    )

    canvas_col1, canvas_col2 = st.columns(2)
    with canvas_col1:
        st.markdown("**Missed porosity** — should be a gap, wasn't marked")
        missed_canvas = st_canvas(
            fill_color="rgba(255,0,0,0.25)",
            stroke_width=3,
            stroke_color="red",
            background_image=pil_bg,
            height=display_h,
            width=LABEL_CANVAS_DISPLAY_WIDTH,
            drawing_mode="circle",
            key=f"missed_canvas_{image_name}",
        )
    with canvas_col2:
        st.markdown("**Mistaken porosity** — wrongly marked as a gap")
        mistaken_canvas = st_canvas(
            fill_color="rgba(0,255,255,0.25)",
            stroke_width=3,
            stroke_color="cyan",
            background_image=pil_bg,
            height=display_h,
            width=LABEL_CANVAS_DISPLAY_WIDTH,
            drawing_mode="circle",
            key=f"mistaken_canvas_{image_name}",
        )

    if st.button("Save labels for this image", type="primary"):
        LABEL_DIR.mkdir(exist_ok=True)
        record = {
            "image": image_name,
            "model_porosity_pct": round(hedge_silhouette_porosity_pct(mask), 2),
            "missed_porosity": _circles_from_canvas_json(missed_canvas.json_data, scale),
            "mistaken_porosity": _circles_from_canvas_json(mistaken_canvas.json_data, scale),
        }
        out_path = LABEL_DIR / f"{Path(image_name).stem}.json"
        out_path.write_text(json.dumps(record, indent=2))
        st.success(
            f"Saved {len(record['missed_porosity'])} missed + "
            f"{len(record['mistaken_porosity'])} mistaken marks to {out_path}"
        )


# ---------------------------------------------------------------------------
# Pixel annotation (full hedge/not-hedge mask correction -> annotation_data/)
# ---------------------------------------------------------------------------
ANNOTATION_DIR = Path("annotation_data")
ANNOTATION_CANVAS_DISPLAY_WIDTH = 650
ANNOTATION_HEDGE_COLOR = (0, 200, 0)      # green, displayed RGB
ANNOTATION_NOT_HEDGE_COLOR = (220, 30, 30)  # red, displayed RGB


def _hedge_material_mask(image_bgr: np.ndarray) -> np.ndarray:
    """True where the current classifier calls a pixel hedge material (not a gap)."""
    return ~segment_gap_mask(image_bgr)


def _classification_rgb(hedge_mask: np.ndarray) -> np.ndarray:
    """Flat green/red visualization of a hedge/not-hedge label grid."""
    rgb = np.empty((*hedge_mask.shape, 3), dtype=np.uint8)
    rgb[hedge_mask] = ANNOTATION_HEDGE_COLOR
    rgb[~hedge_mask] = ANNOTATION_NOT_HEDGE_COLOR
    return rgb


def _classification_overlay_rgb(image_rgb: np.ndarray, hedge_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Green/red classification blended over the photo at `alpha` opacity, so
    the photo underneath stays visible while painting corrections — a flat
    color fill hides exactly the detail you need to see to know where to
    paint.
    """
    flat = _classification_rgb(hedge_mask)
    return cv2.addWeighted(image_rgb, 1 - alpha, flat, alpha, 0)


def _apply_canvas_corrections(base_label_grid: np.ndarray, image_data) -> np.ndarray:
    """
    Overlay user-painted corrections onto the model's label grid.

    image_data is the canvas's RGBA drawing layer at canvas (display)
    resolution — transparent (alpha=0) wherever nothing was painted, so
    those pixels keep the model's original call; painted pixels (alpha>0)
    are relabeled hedge/not-hedge by whichever of red/green is stronger.
    """
    label_grid = base_label_grid.copy()
    if image_data is None:
        return label_grid
    alpha = image_data[..., 3]
    painted = alpha > 0
    if not np.any(painted):
        return label_grid
    red = image_data[..., 0].astype(int)
    green = image_data[..., 1].astype(int)
    painted_hedge = painted & (green > red)
    painted_not_hedge = painted & (red >= green)
    label_grid[painted_hedge] = True
    label_grid[painted_not_hedge] = False
    return label_grid


def _label_grid_to_rle_lines(label_grid: np.ndarray) -> list:
    """Row-major run-length encoding: one line per row, 'label:count,label:count,...'."""
    lines = []
    for row in label_grid:
        runs = []
        current = bool(row[0])
        count = 0
        for value in row:
            value = bool(value)
            if value == current:
                count += 1
            else:
                runs.append(f"{int(current)}:{count}")
                current = value
                count = 1
        runs.append(f"{int(current)}:{count}")
        lines.append(",".join(runs))
    return lines


def annotation_text(image_name: str, label_grid: np.ndarray) -> str:
    """
    Render a full-resolution hedge/not-hedge annotation as run-length-encoded
    text: a small header (image name, width, height) followed by one RLE
    line per image row. 1 = the specific hedgerow being surveyed, 0 =
    anything else (gap, other trees/vegetation, grass, flowers, person,
    background, etc) — a human judgement call, not the raw classifier
    output. See rle_lines_to_label_grid for the matching reader.
    """
    height, width = label_grid.shape
    lines = [
        f"image: {image_name}",
        f"width: {width}",
        f"height: {height}",
        "# 1=the surveyed hedgerow, 0=everything else; each line below is one row, run-length encoded",
    ]
    lines.extend(_label_grid_to_rle_lines(label_grid))
    return "\n".join(lines) + "\n"


def save_annotation_text(image_name: str, label_grid: np.ndarray) -> Path:
    """Write annotation_text's output to annotation_data/<name>.txt on local disk."""
    ANNOTATION_DIR.mkdir(exist_ok=True)
    out_path = ANNOTATION_DIR / f"{Path(image_name).stem}.txt"
    out_path.write_text(annotation_text(image_name, label_grid))
    return out_path


# ---------------------------------------------------------------------------
# Push annotations straight to GitHub (for a hosted, multi-annotator setup)
# ---------------------------------------------------------------------------
# A hosted instance (e.g. Render) has no reliably persistent disk and no git
# credentials by default, so "write to local disk" doesn't get an annotator's
# work back to us. Writing directly through the GitHub Contents API sidesteps
# both problems: no local git state to keep consistent, just one authenticated
# HTTP call per save. Configured via three env vars (unset = feature off,
# falls back to the local-disk save above):
#   GITHUB_TOKEN            - fine-grained PAT scoped to this repo, contents:write
#   GITHUB_REPO             - "owner/repo", e.g. "neogokul/Hedgescan-Prototype"
#   GITHUB_ANNOTATION_BRANCH - defaults to "annotation-data"
import os


def github_push_configured() -> bool:
    return bool(os.environ.get("GITHUB_TOKEN") and os.environ.get("GITHUB_REPO"))


def push_annotation_to_github(image_name: str, label_grid: np.ndarray) -> str:
    """
    Create or update annotation_data/<name>.txt on the configured GitHub
    branch via the Contents API, and return the commit's HTML URL.
    Raises RuntimeError with a readable message on any failure (missing
    config, bad token, network error, API error) — callers should catch
    this and fall back to save_annotation_text rather than lose the work.
    """
    import base64

    import requests

    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPO")
    branch = os.environ.get("GITHUB_ANNOTATION_BRANCH", "annotation-data")
    if not token or not repo:
        raise RuntimeError("GITHUB_TOKEN / GITHUB_REPO not configured")

    path = f"annotation_data/{Path(image_name).stem}.txt"
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

    existing = requests.get(api_url, headers=headers, params={"ref": branch}, timeout=15)
    if existing.status_code not in (200, 404):
        raise RuntimeError(f"GitHub GET failed ({existing.status_code}): {existing.text[:200]}")
    sha = existing.json().get("sha") if existing.status_code == 200 else None

    content = annotation_text(image_name, label_grid)
    payload = {
        "message": f"Annotate {image_name}",
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    response = requests.put(api_url, headers=headers, json=payload, timeout=15)
    if response.status_code not in (200, 201):
        raise RuntimeError(f"GitHub PUT failed ({response.status_code}): {response.text[:200]}")
    return response.json().get("commit", {}).get("html_url", "")


@st.cache_data(ttl=3600, show_spinner="Fetching dataset photo list from GitHub...")
def _list_github_dataset_images(repo: str, branch: str) -> list:
    """Filenames of images on the given branch's root, GitHub-sorted (same
    numeric-aware order as _list_dataset_images), via the Contents API."""
    import requests

    headers = {"Accept": "application/vnd.github+json"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    response = requests.get(
        f"https://api.github.com/repos/{repo}/contents/", headers=headers, params={"ref": branch}, timeout=15
    )
    if response.status_code != 200:
        raise RuntimeError(f"GitHub listing failed ({response.status_code}): {response.text[:200]}")
    names = [
        item["name"] for item in response.json()
        if Path(item["name"]).suffix.lower() in LABEL_IMAGE_EXTENSIONS
    ]

    def sort_key(name):
        try:
            return (0, int(Path(name).stem))
        except ValueError:
            return (1, name)

    return sorted(names, key=sort_key)


@st.cache_data(ttl=3600, show_spinner="Downloading photo from GitHub...")
def _load_github_dataset_image(repo: str, branch: str, name: str) -> np.ndarray:
    """Download one image by name from the dataset branch and decode it."""
    import requests

    headers = {"Accept": "application/vnd.github.raw+json"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    response = requests.get(
        f"https://api.github.com/repos/{repo}/contents/{name}", headers=headers, params={"ref": branch}, timeout=30
    )
    if response.status_code != 200:
        raise RuntimeError(f"GitHub download of {name} failed ({response.status_code}): {response.text[:200]}")
    file_bytes = np.frombuffer(response.content, np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def _dataset_image_source(data_dir: Path):
    """
    Return (image_names, load_fn) for the dataset, preferring a local folder
    (data_dir) when it has images, falling back to the GITHUB_DATASET_BRANCH
    branch of GITHUB_REPO via the API when it's empty — the local folder
    won't exist on a hosted instance whose deploy branch doesn't include the
    dataset photos (they live on a separate branch, e.g. `Dataset`).
    """
    local_images = _list_dataset_images(data_dir)
    if local_images:
        return [p.name for p in local_images], (lambda name: cv2.imread(str(data_dir / name)))

    repo = os.environ.get("GITHUB_REPO")
    branch = os.environ.get("GITHUB_DATASET_BRANCH", "Dataset")
    if not repo:
        return [], None
    names = _list_github_dataset_images(repo, branch)
    return names, (lambda name: _load_github_dataset_image(repo, branch, name))


def rle_lines_to_label_grid(text: str) -> np.ndarray:
    """Inverse of save_annotation_text's RLE body, for reading annotations back."""
    lines = text.splitlines()
    meta = {}
    body_start = 0
    for i, line in enumerate(lines):
        if line.startswith("width:"):
            meta["width"] = int(line.split(":")[1])
        elif line.startswith("height:"):
            meta["height"] = int(line.split(":")[1])
        elif not line.startswith("image:") and not line.startswith("#"):
            body_start = i
            break
    grid = np.zeros((meta["height"], meta["width"]), dtype=bool)
    for row_index, line in enumerate(lines[body_start : body_start + meta["height"]]):
        col = 0
        for run in line.split(","):
            label, count = run.split(":")
            count = int(count)
            if label == "1":
                grid[row_index, col : col + count] = True
            col += count
    return grid


def render_pixel_annotation_tab():
    st.subheader("Pixel annotation: repaint hedgerow vs. everything else")
    st.caption(
        "The green/red you see below is only the classifier's naive "
        "foliage-color guess — it flags anything green as 'hedge', so it "
        "routinely mislabels background trees, other plants, and grass as "
        "green too. It is a rough starting point, not a claim about which "
        "green is the surveyed hedgerow. Paint over ALL of it so the "
        "result reflects only your own judgement: green = the specific "
        "hedgerow being surveyed, red = everything else (gaps, other "
        "trees, other plants, grass, flowers, people, background) — the "
        "same rule regardless of what the system originally guessed. "
        "Saving writes the corrected full mask to `annotation_data/` as a "
        "text file, one per photo."
    )

    data_dir = Path(st.text_input("Dataset folder (relative to the app)", value="Dataset", key="anno_data_dir"))
    try:
        image_names, load_image = _dataset_image_source(data_dir)
    except RuntimeError as exc:
        st.error(f"Could not list dataset photos from GitHub: {exc}")
        return
    if not image_names:
        st.info(
            f"No images found in `{data_dir}/`, and no GITHUB_REPO configured to fall "
            "back to. Point the folder at 1.jpg .. 19.jpg, or set GITHUB_REPO / "
            "GITHUB_DATASET_BRANCH so this can fetch them from GitHub instead."
        )
        return

    ANNOTATION_DIR.mkdir(exist_ok=True)
    annotated_stems = {p.stem for p in ANNOTATION_DIR.glob("*.txt")}
    st.caption(f"{len(annotated_stems)} / {len(image_names)} images annotated so far, saved under `{ANNOTATION_DIR}/`.")

    if "anno_index" not in st.session_state:
        st.session_state.anno_index = 0
    st.session_state.anno_index = max(0, min(len(image_names) - 1, st.session_state.anno_index))

    nav1, nav2, nav3 = st.columns([1, 2, 1])
    with nav1:
        if st.button("< Previous", disabled=st.session_state.anno_index == 0, key="anno_prev"):
            st.session_state.anno_index -= 1
    with nav3:
        if st.button("Next >", disabled=st.session_state.anno_index >= len(image_names) - 1, key="anno_next"):
            st.session_state.anno_index += 1
    with nav2:
        st.session_state.anno_index = st.number_input(
            "Image index", min_value=0, max_value=len(image_names) - 1,
            value=st.session_state.anno_index, step=1, label_visibility="collapsed", key="anno_index_input",
        )

    image_name = image_names[st.session_state.anno_index]
    status = "already annotated" if Path(image_name).stem in annotated_stems else "not yet annotated"
    st.markdown(f"**{image_name}** ({st.session_state.anno_index + 1} / {len(image_names)}) — {status}")

    image_bgr = load_image(image_name)
    orig_h, orig_w = image_bgr.shape[:2]
    hedge_mask_full = _hedge_material_mask(image_bgr)

    scale = ANNOTATION_CANVAS_DISPLAY_WIDTH / orig_w
    display_h = int(round(orig_h * scale))
    hedge_mask_display = cv2.resize(
        hedge_mask_full.astype(np.uint8), (ANNOTATION_CANVAS_DISPLAY_WIDTH, display_h), interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    image_rgb_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb_display = cv2.resize(image_rgb_full, (ANNOTATION_CANVAS_DISPLAY_WIDTH, display_h))

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image(image_rgb_full, caption="Original")
    with img_col2:
        st.image(
            _classification_overlay_rgb(image_rgb_full, hedge_mask_full),
            caption="System's raw foliage-color guess (green = any green pixel, not specifically the surveyed hedgerow) — 50% opacity, photo underneath",
        )

    st.markdown(
        "Paint corrections directly on the canvas below: pick a color, set a "
        "brush size, and repaint any area the system got wrong."
    )
    paint_col1, paint_col2 = st.columns(2)
    with paint_col1:
        paint_label = st.radio("Brush", ["Hedge (green)", "Not hedge (red)"], key="anno_brush", horizontal=True)
    with paint_col2:
        brush_size = st.slider("Brush size", min_value=2, max_value=60, value=15, key="anno_brush_size")
    stroke_color = "rgb(0,200,0)" if paint_label.startswith("Hedge") else "rgb(220,30,30)"

    pil_bg = Image.fromarray(_classification_overlay_rgb(image_rgb_display, hedge_mask_display))
    canvas_result = st_canvas(
        fill_color=stroke_color,
        stroke_width=brush_size,
        stroke_color=stroke_color,
        background_image=pil_bg,
        height=display_h,
        width=ANNOTATION_CANVAS_DISPLAY_WIDTH,
        drawing_mode="freedraw",
        key=f"anno_canvas_{image_name}",
    )

    corrected_display = _apply_canvas_corrections(hedge_mask_display, canvas_result.image_data)
    corrected_full = cv2.resize(
        corrected_display.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    corrected_pct = 100.0 * np.count_nonzero(corrected_display != hedge_mask_display) / hedge_mask_display.size

    save_col, download_col = st.columns(2)
    with save_col:
        if st.button("Save annotation for this image", type="primary"):
            if github_push_configured():
                try:
                    commit_url = push_annotation_to_github(image_name, corrected_full)
                    st.success(
                        f"Pushed to GitHub ({corrected_pct:.1f}% of the canvas repainted). "
                        + (f"[View commit]({commit_url})" if commit_url else "")
                    )
                except RuntimeError as exc:
                    st.error(f"GitHub push failed, falling back to local save: {exc}")
                    out_path = save_annotation_text(image_name, corrected_full)
                    st.warning(f"Saved locally to {out_path} instead — download it below and send it back.")
            else:
                out_path = save_annotation_text(image_name, corrected_full)
                st.success(
                    f"Saved to {out_path} ({corrected_pct:.1f}% of the canvas repainted). "
                    "Running remotely without GITHUB_TOKEN configured? Use the download "
                    "button instead — local disk on a hosted server may not persist."
                )
    with download_col:
        st.download_button(
            "Download annotation .txt",
            data=annotation_text(image_name, corrected_full),
            file_name=f"{Path(image_name).stem}.txt",
            mime="text/plain",
        )


def main():
    st.set_page_config(page_title="HedgeScan Phase 0 Prototype", layout="wide")
    st.title("HedgeScan — Phase 0 Vision Prototype")
    st.caption(
        "Research prototype validating computer-vision approaches against the "
        "Defra Biodiversity Net Gain Metric 4.0 structural attributes. "
        "Not the mobile app."
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Single image", "Section batch", "Labeling", "Pixel annotation"])
    with tab1:
        render_single_image_tab()
    with tab2:
        render_section_batch_tab()
    with tab3:
        render_labeling_tab()
    with tab4:
        render_pixel_annotation_tab()


if __name__ == "__main__":
    main()
