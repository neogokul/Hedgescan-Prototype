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
from dataclasses import dataclass, field
from math import radians, tan

import cv2
import numpy as np
import streamlit as st

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


def segment_gap_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask, True where the pixel is classified as a gap
    (anything that is not hedge foliage or branch/stem material).
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

    is_hedge_material = green_foliage | brown_branch | deep_shadow
    return ~is_hedge_material


def optical_porosity_pct(mask: np.ndarray) -> float:
    """Percentage of the frame classified as gap."""
    return 100.0 * float(np.count_nonzero(mask)) / mask.size


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
    gap mask. A column counts as "gap" if the majority of pixels in a
    vertical foliage band are gap. Consecutive gap columns form a single
    run, converted to metres by the image's along-hedge ground length.
    """
    width_px = mask.shape[1]
    if width_px == 0:
        return []

    column_is_gap = mask.mean(axis=0) > 0.5
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
    porosity = optical_porosity_pct(mask)
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

    mask_full = segment_gap_mask(image_bgr)
    mask_band = segment_gap_mask(hedge_band_bgr)
    porosity = optical_porosity_pct(mask_band)
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
    st.metric("B2 — Optical porosity (%, single image, hedge band only)", f"{porosity:.1f}", delta="PASS" if porosity_pass else "FAIL")
    st.caption(
        "This single-image porosity figure is indicative only — B2 requires "
        "aggregation across the full 30 m section (see Section batch tab)."
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


def main():
    st.set_page_config(page_title="HedgeScan Phase 0 Prototype", layout="wide")
    st.title("HedgeScan — Phase 0 Vision Prototype")
    st.caption(
        "Research prototype validating computer-vision approaches against the "
        "Defra Biodiversity Net Gain Metric 4.0 structural attributes. "
        "Not the mobile app."
    )

    tab1, tab2 = st.tabs(["Single image", "Section batch"])
    with tab1:
        render_single_image_tab()
    with tab2:
        render_section_batch_tab()


if __name__ == "__main__":
    main()
