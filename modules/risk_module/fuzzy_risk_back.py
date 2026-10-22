import numpy as np

ARR_DEPTH_LIMITS = {"H1": 0.30, "H2": 0.50, "H3": 1.20, "H4": 2.00, "H5": 4.0}

MAX_DEPTH_LIMITS = dict(ARR_DEPTH_LIMITS)

DEPTH_UNIVERSE_MAX_M = 12.0

# Slack allowed when a depth is compared to a class boundary, in metres.
#
# Exposure stores depth_max rounded to 2 dp, so depths land EXACTLY on 0.30 and
# 0.50 routinely rather than rarely. Averaging such values over ensemble members
# does not return them: the mean of twelve copies of 0.30 is 0.29999999999999993,
# one ulp below the cutoff, which digitises to H1 where the same depth read from
# a single member digitises to H2. The error is one-directional (always the
# class BELOW) and depends on the member count -- it bites at M = 10, 12, 20 and
# not at M = 5 or 100 -- so an ensemble-size sweep would carry a spurious
# M-dependent step in every column derived from an averaged depth
# (ensemble_mean_hazard_*, final_risk_ensemble_mean, and any skill score
# referenced to them).
#
# A nanometre is far below the precision of any depth this model produces or any
# depth the ARR limits are quoted to, so nothing physical is at stake; it only
# decides which side of a boundary a value that IS the boundary falls on, and
# hazard_class documents that side as closed below.
#
# Applied in both hazard_class and fuzzify, so the crisp and fuzzy readings of
# the same rule base stay identical (see crisp_rule_strengths). Nudging only one
# of them would make them disagree inside this band.
BOUNDARY_TOLERANCE_M = 1e-9

# The three SEVERITY POSITIONS the inference works in. The FIS is positional:
# index 0 is the safe term, 1 the middle, 2 the severe one, and every rule,
# aggregation and argmax below indexes by position rather than by name. What
# differs between object types is which ARR depth band each position covers,
# and therefore what the position is CALLED -- see OBJECT_FUZZIFICATION.
#
# ALL THREE OBJECT TYPES NOW SIT ON THE SAME BANDS, cut at the ARR H1 and H2
# limits: safe below 0.30 m, middle 0.30-0.50 m, severe above 0.50 m. The
# machinery for per-type bands is kept -- OBJECT_FUZZIFICATION still carries its
# own cutoffs and names per type -- because it is what makes the choice visible
# and reversible, not because the types currently differ.
#
# The building and property scale was previously cut at H1 and H3, giving a
# middle term spanning 0.30-1.20 m. That is most of the depths a flood at a
# building actually takes, so nearly every wet structure landed in one tier and
# the reported class carried little information. Splitting at 0.50 m instead
# resolves the low end, where the placard evidence actually separates.
#
# What distinguishes a flooded street from a flooded house is therefore the RULE
# BASE, not the fuzzification: R4 caps the street's own contribution at the
# middle tier however deep the carriageway gets (see COMBINED_RULES).
#
# TERMS is the BUILDING scale, because the combined FIS emits a BUILDING hazard:
# its output tier, its probability columns and the warning thresholds are all
# named on this scale. An object reported on its own -- a road parcel, say --
# uses its own names, via object_terms().
TERMS = ("H1", "H2", "H3_above")
HAZARD_LEVELS = (1, 2, 3)

# Display forms for figures, over every scale in use. "_above" is spelled out
# rather than "+" so a term is a plain identifier that can sit inside \mathrm{}
# without the underscore becoming a subscript.
TERM_LABELS = {"H1": "H1", "H2": "H2", "H3_above": "H3+"}

OBJECT_FUZZIFICATION = {
    # cutoffs are the ARR limits the bands are cut at; terms name the three
    # positions on that object's own scale, ordered safe -> severe.
    "building": {"cutoffs": (MAX_DEPTH_LIMITS["H1"], MAX_DEPTH_LIMITS["H2"]),
                 "transition": 0.0,
                 "terms": ("H1", "H2", "H3_above")},
    "property": {"cutoffs": (MAX_DEPTH_LIMITS["H1"], MAX_DEPTH_LIMITS["H2"]),
                 "transition": 0.,
                 "terms": ("H1", "H2", "H3_above")},
    "road":     {"cutoffs": (MAX_DEPTH_LIMITS["H1"], MAX_DEPTH_LIMITS["H2"]),
                 "transition": 0.0,
                 "terms": ("H1", "H2", "H3_above")},
}

# Positions, for code that must index a term without knowing the scale.
SAFE, MIDDLE, SEVERE = 0, 1, 2


def object_terms(object_type):
    """The three term names for one object type, ordered safe -> severe."""
    return OBJECT_FUZZIFICATION[object_type]["terms"]


WARN_AT_LEAST_H2 = 0.5
WARN_H3_ABOVE = 0.5


# ---------------------------------------------------------------------------
# Fuzzification
# ---------------------------------------------------------------------------

def trapezoid_breakpoints(object_type):
    """The (a, b, c, d) breakpoints of one object type's three sets, keyed by
    that type's own term names (see object_terms).

    Built from the ARR tier boundaries so the sets cannot drift from the
    standard. The three form a partition of unity, so a depth on a boundary
    splits its membership evenly between the adjacent terms.

    The sets are cut for h_max, which is a depth AT A PLACE, so an ARR limit
    applies to it directly. That is the reason h_max is the REPORTED indicator.

    THE h_mean PASS REUSES THESE SAME BREAKPOINTS, and that is a modelling
    choice rather than a reading of the guideline: no ARR class attaches to a
    spatial average. It is made deliberately, so the two passes differ in the
    indicator ALONE and their contingency tables are comparable -- fitting
    separate cutoffs to h_mean would confound the indicator with a recalibrated
    rule base. It also makes the h_mean pass systematically the more
    conservative of the two, because an object's h_mean is diluted by the dry
    cells of its ring (impact_module divides by every in-zone cell, not by the
    wet ones), so a partly-wet object reads shallow BECAUSE it is partly dry.

    ``transition`` is either one width applied to both boundaries, or a
    ``(lower, upper)`` pair to widen them independently. The two boundaries
    carry different amounts of doubt -- the H1/H2 one decides whether
    water has reached the object at all, the upper one whether it
    threatens the structure -- so tying them to a single width is a constraint,
    not a property of the guideline.
    """
    config = OBJECT_FUZZIFICATION[object_type]
    lo_cut, hi_cut = config["cutoffs"]
    transition = config["transition"]
    lo_width, hi_width = (
        transition if isinstance(transition, (tuple, list)) else (transition, transition)
    )

    lo_ramp = round(max(lo_cut - lo_width, 0.0), 6)
    hi_ramp = round(hi_cut - hi_width, 6)
    if not 0.0 <= lo_ramp <= lo_cut <= hi_ramp <= hi_cut:
        raise ValueError(
            f"{object_type}: transition widths {(lo_width, hi_width)} do not leave "
            f"ordered breakpoints for cutoffs {(lo_cut, hi_cut)}"
        )
    # Zipped against this object type's own terms rather than spelled out, so
    # the keys cannot drift from the names it is reported under.
    return dict(zip(object_terms(object_type), (
        (0.0, 0.0, lo_ramp, lo_cut),
        (lo_ramp, lo_cut, hi_ramp, hi_cut),
        (hi_ramp, hi_cut, DEPTH_UNIVERSE_MAX_M, DEPTH_UNIVERSE_MAX_M),
    )))


def _trapezoid(x, a, b, c, d):
    """Trapezoidal membership mu_A(x), elementwise. a == b gives a left
    shoulder and c == d a right one, which is how the safe and severe terms are
    built.
    """
    x = np.asarray(x, dtype=float)
    mu = np.zeros(x.shape, dtype=float)

    if b > a:
        rising = (x > a) & (x < b)
        mu[rising] = (x[rising] - a) / (b - a)
    np.copyto(mu, 1.0, where=(x >= b) & (x <= c))
    if d > c:
        falling = (x > c) & (x < d)
        mu[falling] = (d - x[falling]) / (d - c)
    return mu


def fuzzify(depth, object_type):
    """Fuzzify a depth array into its three memberships, on the object's scale.

    Parameters
    ----------
    depth : array-like
        The pass's depth indicator in metres, any shape -- h_max for the
        reported pass, h_mean for the alternative one, on the same breakpoints.
    object_type : {"building", "property", "road"}
        Selects the ARR-derived breakpoints (see OBJECT_FUZZIFICATION).

    Returns
    -------
    ndarray, shape depth.shape + (3,)
        Memberships stacked on a trailing axis ordered as TERMS, summing to 1.

    Depths are clamped into [0, DEPTH_UNIVERSE_MAX_M] first. Without the clamp
    a depth past the universe ceiling falls off the flat top of the severe
    set and fuzzifies to zero membership everywhere, which elects it from an
    empty argmax -- the right answer for the wrong reason, and the wrong answer
    for a negative depth. NaN is left to propagate so it cannot be mistaken for
    a real classification; see elected_tier.
    """
    points = trapezoid_breakpoints(object_type)
    depth = np.clip(
        np.asarray(depth, dtype=float) + BOUNDARY_TOLERANCE_M,
        0.0, DEPTH_UNIVERSE_MAX_M,
    )
    return np.stack(
        [_trapezoid(depth, *points[term]) for term in object_terms(object_type)],
        axis=-1,
    )


# ---------------------------------------------------------------------------
# Inference primitives -- shared by the combined FIS
# ---------------------------------------------------------------------------

def aggregate(strengths, rules):
    """Aggregate rule firing strengths into one activation per linguistic term.

    Rules sharing a consequent are combined by their strongest activation,
    w_A = max over rules r with C_r = A. Returns shape (..., 3) ordered as
    TERMS; a term claimed by no rule aggregates to 0.
    """
    consequents = np.array([TERMS.index(consequent) for _, consequent in rules])
    out = np.zeros(strengths.shape[:-1] + (len(TERMS),), dtype=float)
    for term_index in range(len(TERMS)):
        members = strengths[..., consequents == term_index]
        if members.shape[-1]:
            out[..., term_index] = members.max(axis=-1)
    return out


def elected_tier(activations):
    """Tier elected by a set of aggregated activations: argmax_A w_A.

    Ties resolve to the MORE SEVERE tier, so an object activating medium and
    high equally is reported high rather than rounded down into a weaker
    warning. Returns an int array of 1/2/3.

    A row where nothing fired (all-zero or NaN activations -- only reachable
    from a NaN depth, since fuzzify clamps everything else into the universe)
    also elects the most severe tier. That is deliberate rather than an
    artefact of the argmax: a feature whose depth could not be evaluated must
    not be quietly reported as safe.
    """
    activations = np.nan_to_num(np.asarray(activations, dtype=float), nan=0.0)
    # argmax over reversed terms picks the last maximum, i.e. the most severe
    # among ties; map that index back to a 1/2/3 level.
    return (len(TERMS) - np.argmax(activations[..., ::-1], axis=-1)).astype(int)


# ---------------------------------------------------------------------------
# Object level -- a crisp reading, outside the inference
# ---------------------------------------------------------------------------

def hazard_class(depth_max, object_type):
    """Crisp ARR hazard class of a depth: h_max compared straight to the cutoffs.

    Cut at this object type's own cutoffs (see OBJECT_FUZZIFICATION). No
    fuzzy sets are involved -- with h_max the only indicator there is nothing
    to infer at object level, and a comparison against the guideline cutoff is
    the whole of what the object-level reading claims to be.

    THIS TIER IS REPORTED, NOT INFERRED FROM. The combined FIS takes the three
    objects' crisp h_max and fuzzifies them itself; it never sees this integer.
    Crisping each object first and combining the tiers would discard exactly
    the partial membership the FIS exists to carry, so the two paths are kept
    apart on purpose.

    Boundaries are closed below, matching ARR: a depth of exactly 0.30 m is H2.
    A NaN depth takes the MOST SEVERE class, as elected_tier does for the
    inference -- a feature whose depth could not be evaluated must not be
    reported as safe. No clamping is needed: the comparison is unbounded above,
    and a negative depth falls into H1 as a zero would.

    Parameters
    ----------
    depth_max : array-like
        The pass's depth indicator in metres (features, or members x
        features). Named for h_max, the reported one; the h_mean pass hands
        its own depths to the same comparison.
    object_type : {"building", "property", "road"}
        Selects the cutoffs (see OBJECT_FUZZIFICATION).

    Returns
    -------
    ndarray
        int tier 1/2/3, same shape as ``depth_max``.
    """
    cutoffs = OBJECT_FUZZIFICATION[object_type]["cutoffs"]
    depth = np.asarray(depth_max, dtype=float)
    # np.digitize already sends NaN to the top bin; spelled out so that is a
    # decision the reader can see rather than a side effect they must know.
    # The tolerance keeps a depth that IS the cutoff on the closed-below side of
    # it after floating-point averaging -- see BOUNDARY_TOLERANCE_M.
    return np.where(
        np.isnan(depth), len(TERMS),
        np.digitize(depth + BOUNDARY_TOLERANCE_M, cutoffs) + 1,
    ).astype(int)


def class_lower_bounds(object_type="building", wet_floor=0.0):
    """The depth at which each hazard class STARTS, on one object's scale.

    The cutoffs in OBJECT_FUZZIFICATION are the boundaries BETWEEN the three
    classes, so read as lower bounds they are the middle and severe terms';
    the safe term starts at ``wet_floor``. Closed below, as hazard_class is:
    exactly 0.30 m is H2, not H1.

    ``wet_floor`` is where "flooded at all" begins -- the safe class has no
    lower cutoff of its own, and a bound of 0.0 would make every dry building
    exceed it. Callers working on exposure depths pass
    impact_module.WET_DEPTH_MIN, below which a cell is floored to zero anyway,
    so the safe bound and the wet definition agree by construction rather than
    by two thresholds happening to be set the same.

    Returns an (term -> depth) dict in the object's own term order.
    """
    cutoffs = OBJECT_FUZZIFICATION[object_type]["cutoffs"]
    return dict(zip(object_terms(object_type), (wet_floor,) + tuple(cutoffs)))


def depth_exceedance(depth, object_type="building", wet_floor=0.0):
    """``P(depth >= class lower bound)`` per class, over the member axis.

    THE DEPTH-SPACE COUNTERPART OF exceedance(), and a different quantity from
    it. exceedance() works in TIER space: it reverse-accumulates the ensemble's
    votes over tiers the FIS elected, so P(>= H1) is 1 by construction and the
    fuzzy combination of several objects stands between the depths and the
    number. This one asks the ensemble the crisp question directly -- what
    share of members put the water at or above the depth this class begins at
    -- so a dry building scores 0 on every class including the safe one, and
    nothing but the depth and the cutoff enters.

    Members whose depth could not be evaluated count as EXCEEDING every bound,
    the same refusal to report an unassessed feature as safe that hazard_class
    and elected_tier make. In practice the exposure path floors NaN cells to
    zero, so this is a guarantee about the arithmetic rather than a case that
    arises.

    Parameters
    ----------
    depth : array-like, shape (n_members, n_features)
        One crisp depth indicator per member per feature, in metres.
    object_type : {"building", "property", "road"}
        Selects the cutoffs (see OBJECT_FUZZIFICATION).
    wet_floor : float
        Lower bound of the safe class -- see class_lower_bounds.

    Returns
    -------
    ndarray, shape (n_features, 3)
        Ordered as this object type's terms, safe -> severe, so column A is
        P(depth >= start of class A). Non-increasing across the row, since the
        bounds increase. All-NaN if the ensemble is empty.
    """
    depth = np.asarray(depth, dtype=float)
    bounds = np.asarray(list(class_lower_bounds(object_type, wet_floor).values()))
    n_members, n_features = depth.shape
    if n_members == 0:
        return np.full((n_features, len(bounds)), np.nan)
    exceeds = np.isnan(depth)[..., None] | (depth[..., None] >= bounds)
    return exceeds.mean(axis=0)


def object_hazard_columns(depth_max, object_type, prefix="overall_hazard",
                          mean_map_depth_max=None):
    """A road parcel or property assessed on its own: crisp tier, likelihood,
    final risk level.

    There is no second object to combine with -- the FIS exists to weigh a
    building against its section and its street, and asking it about a road
    alone would just hand back the road -- so no inference runs here at all.
    Each ensemble member's h_max is classified against the cutoffs, the members
    are pooled into a tier and its likelihood, and the same de-escalation
    matrix the building layer uses turns that pair into the final risk level.

    ``{prefix}_level`` is the object's hazard tier BEFORE de-escalation, which
    is what makes it mean the same thing as the building layer's
    ``overall_building_hazard_level``; the de-escalated reading is
    ``final_risk_level`` on every layer alike.

    Parameters
    ----------
    depth_max : array-like, shape (n_members, n_features)
        Per-member h_max in metres for this object type.
    object_type : {"building", "property", "road"}
    prefix : str
        Column-name prefix for the hazard outputs.
    mean_map_depth_max : array-like, shape (n_features,), optional
        h_max read off the ensemble-mean flood map. Supplied by the caller
        because the averaging happens before exposure and so cannot be
        recovered from ``depth_max``; omitted, the column is not emitted.

    Returns
    -------
    dict
        the ``{prefix}_*`` hazard columns, the averaged tiers and the final
        risk level.
    """
    # Members are pooled only after each has been classified, as everywhere
    # else here, so the likelihood is agreement over readings, not over depths.
    terms = object_terms(object_type)
    levels, likelihood, probs = hazard_likelihoods(hazard_class(depth_max, object_type))

    columns = {} if mean_map_depth_max is None else {
        # Read against the cutoffs, not through the FIS: no inference runs on
        # this layer, so the averaged reading is the same crisp comparison the
        # rest of the layer is (see the building layer for the FIS version).
        "final_risk_ensemble_mean": hazard_class(mean_map_depth_max, object_type),
    }
    return {
        **columns,
        # The other averaging order -- the members' own h_max averaged, rather
        # than one h_max read off the averaged map.
        "ensemble_mean_hazard_level": hazard_class(
            ensemble_mean_depth(depth_max), object_type
        ),
        f"{prefix}_level": levels,
        f"{prefix}_likelihood": likelihood,
        # The warning axis: better-than-even odds of reaching the tier. Kept
        # beside the likelihood rather than replacing it, so the modal share is
        # still readable for the earlier results.
        # Indexed by position and named on this object's own scale, so a road
        # would report prob_at_least_H2 on a scale whose middle term were
        # named differently. All three types share one scale at present, so the
        # names coincide -- the indirection is what keeps that a fact about the
        # configuration rather than an assumption baked into the column name.
        f"{prefix}_prob_at_least_{terms[MIDDLE]}": exceedance(probs)[:, MIDDLE],
        f"{prefix}_prob_{terms[SEVERE]}": probs[:, SEVERE],
        "final_risk_level": final_risk(probs),
    }


# ---------------------------------------------------------------------------
# The combined FIS -- the only inference in the system
# ---------------------------------------------------------------------------

# The rule base is the decision table below. The building and its property
# enter as peers -- the outcome depends on the more severe of the two, which is
# what "building X OR property X" is for the severe terms and "building low AND
# property low" for the safe one. Written out for that pair against the street
# (rows for property medium/high follow by the same symmetry):
#
#   building/property   street    risk
#   low                 low       low
#   low                 medium    medium
#   low                 high      medium
#   medium              low       medium
#   medium              medium    medium
#   medium              high      high
#   high                any       high
#
# So the hazard is the most severe of the three objects, with one cap: the
# street cannot elect HIGH on its own. A deep road beside a dry building and a
# dry section stops at medium (R4). That cap is the rule base's own doing, not
# a consequence of the fuzzification -- all three object types currently sit on
# the same cutoffs (see OBJECT_FUZZIFICATION), so nothing but R4 distinguishes
# a deep street from a deep house. An impassable street is a real warning, but
# it is not the same finding as a flooded house. Below the cap the street
# speaks at its own tier: medium on
# its own is a medium risk (the s_med disjunct in R3), and high lifts a medium
# pair to high (R2).
#
# Staying low consequently requires all three to be low (R5). A merely wet
# street is enough to leave the low tier on its own, which is the deliberate
# cost of treating road access as a component of the risk rather than as a
# modifier of the structure's.

COMBINED_RULES = (
    ("R1", "H3_above"),  # building severe OR property severe
    ("R3", "H2"),        # building middle OR property middle
    ("R4", "H2"),        # building safe AND property safe AND street severe
    ("R5", "H1"),        # building safe AND property safe AND street not severe
)

# THE STREET CANNOT ELECT THE TOP TIER. The old R2 -- (pair middle AND street
# severe) -> top -- is gone, and dropping it rather than re-pointing it at the
# middle tier is not a judgement call but an identity: R2 = min(pair_med,
# s_high) <= pair_med = R3, and with both consequents equal to H2 the max in
# aggregate absorbs R2 into R3 entirely. Keeping it would have been a rule that
# can never be the strongest for its own consequent.
#
# That argument is purely about the rule base and holds whatever the bands are.
# It is worth saying because the OTHER argument for deleting R2 no longer
# applies: R2 was removed when the object scales had diverged, so that a street
# severe at 0.50 m could not escalate a building whose own severe term began at
# 1.20 m. The scales have since been brought back together
# (see OBJECT_FUZZIFICATION), and the deletion stands on the identity alone.
#
# What the street still does is R4: it lifts a dry structure to the middle tier.
# A road over 0.50 m is a real access finding even where the house is dry, and
# the middle tier is where that belongs.
#
# The labels R1, R3, R4, R5 are kept at their original meanings rather than
# renumbered, so a rule name in an old output still denotes the same rule.

COMBINED_RULE_DRIVER = {
    "R1": "structure", "R3": "structure",   # the building or its own section
    "R4": "street",                         # the road lifted it off the floor
    "R5": "no driver",                      # nothing lifted it above low
}


def combined_rule_strengths(building, property_, street):
    """Firing strength of each combined-FIS rule R1-R5, from the three
    objects' (..., 3) activations. Returns shape (..., 5), aligned to
    COMBINED_RULES.

    Combining in the fuzzy domain means an object that only weakly activated
    its severe term contributes only weakly, rather than being crisped first. This
    is the whole reason the object tiers from hazard_class stay out of here:
    they are that same evidence rounded to an integer, and a rule base fed on
    them could not tell a building at 0.29 m from one at 0.01 m.

    The building and its property enter as peers: severe terms disjunctive,
    since water at either is evidence the building was reached, and the safe
    term conjunctive, since staying low needs both.

    The street enters only where it ESCALATES: a high street lifts a medium
    pair to high (R2) and a dry pair to medium (R4). A medium street does
    nothing on its own -- R3 tests the structure alone -- because water in the
    road is not a finding about the building, and R5 keeps a dry structure
    beside a medium street at low. The street also cannot elect high by
    itself: R4 caps a deep street beside a dry building and a dry section at
    medium, because a flooded street is not the same finding as a flooded
    house. The cap lives in R4 alone -- the road is fuzzified on the same
    cutoffs as the building, so nothing upstream of the rules softens it.

    Staying low is therefore R5, which now accepts any street term short of
    high rather than requiring the road to be dry.
    """
    b_low, b_med, b_high = building[..., 0], building[..., 1], building[..., 2]
    p_low, p_med, p_high = property_[..., 0], property_[..., 1], property_[..., 2]
    s_low, s_med, s_high = street[..., 0], street[..., 1], street[..., 2]

    pair_high = np.maximum(b_high, p_high)
    pair_med = np.maximum(b_med, p_med)
    pair_low = np.minimum(b_low, p_low)

    return np.stack(
        [
            pair_high,                           # R1
            pair_med,                            # R3
            np.minimum(pair_low, s_high),        # R4
            np.minimum(pair_low, np.maximum(s_low, s_med)),   # R5
        ],
        axis=-1,
    )


def combined_risk(building, property_, street):
    """Run the combined FIS: three objects' activations in, one overall
    building hazard level out.

    Returns
    -------
    (ndarray, ndarray, ndarray)
        activations of shape (..., 3), the elected hazard tier (int 1/2/3), and
        the raw rule strengths of shape (..., 5) for explanation.
    """
    strengths = combined_rule_strengths(building, property_, street)
    activations = aggregate(strengths, COMBINED_RULES)
    return activations, elected_tier(activations), strengths


def combined_risk_without_street(building, property_):
    """Run the combined FIS with the street antecedent marginalised out.

    Street takes the building's activations, which is the same device
    overall_building_hazard uses for an object a building has no link to: the
    street term
    in COMBINED_RULES then collapses and the rules read as if that antecedent
    had never been asked. So this is the same rule base with one input
    withheld, not a second rule base.

    With s = b every street term collapses into a term the pair already has:
    R2 becomes min(pair_med, b_high) <= b_high <= pair_high, so it is absorbed
    by R1; R3 never read the street to begin with; R4 becomes
    min(pair_low, b_high) = 0, because pair_low <= b_low and an object's "H1"
    and severe activations come from disjoint depth sets; R5's s_low
    becomes min(pair_low, max(b_low, b_med)) = pair_low, since pair_low <= b_low
    <= max(b_low, b_med). What is left is the building and its property as
    peers with no street term at all.

    Same return signature as combined_risk.
    """
    return combined_risk(building, property_, building)


# ---------------------------------------------------------------------------
# The same rule base, evaluated crisply
# ---------------------------------------------------------------------------
#
# COMBINED_RULES read on each object's CLASS instead of its memberships. With
# OBJECT_FUZZIFICATION's transition widths at 0 the trapezoids are rectangles,
# the memberships are indicator functions, and the two paths are the SAME
# FUNCTION -- verified exhaustively over a depth grid, not argued. Widen a
# transition and they part company again, which is why this is a separate
# implementation to switch to rather than a rewrite of the fuzzy one.
#
# What it buys is not speed: the FIS is elementwise numpy and was never the
# cost, exposure is. It buys a path whose output can be read off the cutoffs
# by hand, which is what a reader checking the method against ARR needs.


def crisp_rule_strengths(building, property_, street):
    """Firing of R1/R3/R4/R5 from the three objects' CRISP CLASSES, as 0/1.

    Same shape and rule order as combined_rule_strengths, so everything
    downstream -- aggregate, elected_tier, dominant_rule -- is indifferent to
    which path produced it.

        R1  b severe  OR p severe                    -> H3_above
        R3  b middle  OR p middle                    -> H2
        R4  b safe AND p safe AND street severe      -> H2
        R5  b safe AND p safe AND street not severe  -> H1

    R4 is the one rule that is not a max over the three: a deep carriageway
    lifts a dry structure to the MIDDLE tier and no further, so the street can
    never elect the top class. That is why the crisp form of this rule base is
    not simply hazard_class(max(h_b, h_p, h_s)).

    Each argument is an integer class array (1/2/3) as hazard_class returns,
    all three the same shape.
    """
    safe, severe = HAZARD_LEVELS[SAFE], HAZARD_LEVELS[SEVERE]
    middle = HAZARD_LEVELS[MIDDLE]
    building, property_, street = (np.asarray(a) for a in (building, property_, street))

    pair_safe = (building == safe) & (property_ == safe)
    street_severe = street == severe
    return np.stack([
        (building == severe) | (property_ == severe),      # R1 -> severe
        (building == middle) | (property_ == middle),      # R3 -> middle
        pair_safe & street_severe,                         # R4 -> middle
        pair_safe & ~street_severe,                        # R5 -> safe
    ], axis=-1).astype(float)


def crisp_combined_risk(building, property_, street):
    """The combined rule base on crisp classes. Drop-in for combined_risk.

    Returns the same ``(activations, tier, strengths)`` triple, with the
    activations one-hot rather than graded -- which is what the fuzzy path also
    produces once the transitions are 0.
    """
    strengths = crisp_rule_strengths(building, property_, street)
    activations = aggregate(strengths, COMBINED_RULES)
    return activations, elected_tier(activations), strengths


def crisp_combined_risk_without_street(building, property_):
    """As combined_risk_without_street, crisply: the street takes the
    building's own class, which collapses R4 to nothing (a class cannot be both
    safe and severe) and leaves R5 as "both safe" -- the same absorption the
    fuzzy version documents, reached the same way.
    """
    return crisp_combined_risk(building, property_, building)


def dominant_rule(strengths, rules):
    """Label of the strongest-firing rule per feature, for explanation.

    Answers "why did this object get this tier" with the rule that actually
    drove it. Ties go to the earliest rule. Empty label where nothing fired.
    """
    labels = np.array([label for label, _ in rules], dtype=object)
    strengths = np.asarray(strengths, dtype=float)
    winner = labels[np.argmax(strengths, axis=-1)]
    return np.where(strengths.max(axis=-1) > 0, winner, "")


# ---------------------------------------------------------------------------
# Ensemble likelihood and warning tier
# ---------------------------------------------------------------------------

def ensemble_risk_probabilities(member_levels):
    """Likelihood of each tier from member agreement, p_A(b, t).

    Every member goes independently through exposure and the full inference, so
    member m assigns its own tier. The likelihood of tier A is the fraction of
    members assigning it::

        p_A = (1 / M) * sum_m 1[H^(m) == A],    A in TERMS

    Each p_A is in [0, 1] and the three sum to 1. This estimates P(tier)
    directly rather than proxying forecast uncertainty with lead time, and
    quantifies the epistemic uncertainty the surrogate contributes.

    Takes (n_members, n_features) tiers, returns (n_features, 3) ordered as
    TERMS; all-NaN if the ensemble is empty.
    """
    member_levels = np.asarray(member_levels)
    n_members, n_features = member_levels.shape
    if n_members == 0:
        return np.full((n_features, len(TERMS)), np.nan)
    return np.stack(
        [(member_levels == level).mean(axis=0) for level in HAZARD_LEVELS], axis=1
    )


def ensemble_risk_level(probs):
    """Reported risk tier and its likelihood from the ensemble distribution.

    The feature sits on the risk axis of the matrix at the tier supported by
    the largest share of the ensemble, and the likelihood axis reports that
    share::

        H = argmax_A p_A,    L = max_A p_A

    Ties resolve to the MORE SEVERE tier: a 50/50 split between medium and
    high reports high, so an even ensemble is never rounded down.

    Note that with three tiers the shares sum to 1, so L >= 1/3 always: the
    likelihood lives on [1/3, 1], not [0, 1]. LIKELIHOOD_BANDS is calibrated
    to that range -- L < 0.5 means no tier held a majority, which is the
    weakest statement this formulation can make.

    Parameters
    ----------
    probs : ndarray, shape (n_features, 3)
        Per-tier ensemble probabilities from ensemble_risk_probabilities.
        Must come from a non-empty ensemble (no NaN rows).

    Returns
    -------
    (ndarray, ndarray)
        risk_level (int 1/2/3) and likelihood (float in [0, 1]) per feature.
    """
    probs = np.asarray(probs, dtype=float)
    if probs.shape[0] == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)
    levels = elected_tier(probs)
    return levels, probs[np.arange(probs.shape[0]), levels - 1]



def exceedance(probs):
    """``P(tier >= A)`` for each tier, from the per-tier probabilities.

    The reverse-cumulative sum of ``probs``, so column A holds the probability
    that the hazard reaches tier A OR ANYTHING WORSE. Column 0 is P(tier >= H1),
    which is 1 by construction and carried only to keep the axis aligned with
    TERMS.

    Takes (n_features, 3) as returned by ensemble_risk_probabilities and
    returns the same shape. All-NaN rows (an empty ensemble) stay NaN.
    """
    probs = np.asarray(probs, dtype=float)
    return np.cumsum(probs[..., ::-1], axis=-1)[..., ::-1]


def final_risk(probs):
    """Final risk level from the ensemble's exceedance probabilities.

    The impact-based-forecasting reading, taken as a threshold decision rather
    than a classification: a feature is warned at a tier when the ensemble puts
    better-than-even odds on the hazard REACHING that tier.

        level 3   if P(severe)      >= WARN_H3_ABOVE
        level 2   if P(tier >= mid)  >= WARN_AT_LEAST_H2
        level 1   otherwise

    This replaces the hazard-tier x likelihood-band matrix, whose low band was
    unreachable and whose surviving cut sat on the modal share -- see the note
    on _FINAL_RISK_MATRIX. It deliberately does NOT require the elected tier to
    agree: an ensemble at (0.50, 0.25, 0.25) elects H1 but puts even odds on
    medium or worse, and that is a building the old axis reported as safe.

    Note the two clauses cannot contradict each other. P(severe) >= 0.5 forces
    the elected tier to the severe term as well, since no other tier can then hold a
    larger share and ties resolve upward, so a level-3 warning never lands on a
    feature the inference called low.

    Parameters
    ----------
    probs : ndarray, shape (n_features, 3)
        Per-tier ensemble probabilities from ensemble_risk_probabilities.
        NaN rows (no ensemble members) elect the most severe level, so a
        missing ensemble is never quietly reported as safe -- the same choice
        _likelihood_bands made by mapping NaN to the high band.

    Returns
    -------
    ndarray
        final_risk_level (int 1/2/3) per feature.
    """
    probs = np.asarray(probs, dtype=float)
    if probs.shape[0] == 0:
        return np.empty(0, dtype=int)

    at_least = exceedance(probs)
    unknown = np.isnan(probs).any(axis=-1)
    prob_severe = np.where(unknown, 1.0, probs[:, SEVERE])
    prob_middle_up = np.where(unknown, 1.0, at_least[:, MIDDLE])

    return np.where(
        prob_severe >= WARN_H3_ABOVE, 3,
        np.where(prob_middle_up >= WARN_AT_LEAST_H2, 2, 1),
    ).astype(int)


def hazard_likelihoods(member_levels):
    """Elected level, its likelihood, and p_A for every level, from member
    agreement over a (n_members, n_features) stack of per-member levels.
    """
    probs = ensemble_risk_probabilities(member_levels)
    levels, likelihood = ensemble_risk_level(probs)
    return levels, likelihood, probs


# ---------------------------------------------------------------------------
# Entry point -- inference per member, then pooled
# ---------------------------------------------------------------------------

def _combined_columns(member_levels, strengths, prefix):
    """Pool a combined-FIS pass over the ensemble into its reported columns.

    ``member_levels`` is the (n_members, n_features) tier each member's
    inference elected and ``strengths`` its (n_members, n_features, 5) rule
    firings. Parameterised on ``prefix`` so a pass run with a different set of
    antecedents reports the same way under its own column names.
    """
    levels, likelihood, probs = hazard_likelihoods(member_levels)
    rule = dominant_rule(strengths.mean(axis=0), COMBINED_RULES)
    columns = {
        f"{prefix}_level": levels,
        f"{prefix}_likelihood": likelihood,
        f"{prefix}_rule": rule,
        # What put the building at that tier: its own footprint or section
        # ("structure"), or a deep road beside an otherwise dry one ("street").
        # Blank where the winning rule was the low one, which has no driver.
        f"{prefix}_driver": np.array(
            [COMBINED_RULE_DRIVER.get(label, "") for label in rule], dtype=object
        ),
    }
    for i, term in enumerate(TERMS):
        columns[f"{prefix}_prob_{term}"] = probs[:, i]
    return columns


def ensemble_mean_depth(depth_max):
    """Mean h_max over the ensemble member axis, NaN for an empty ensemble.

    The mean is taken over MEMBERS. It is not h_mean, the spatial mean depth
    over an object's cells, which left the inference entirely (see the module
    docstring); each object still enters on an h_max, just an averaged one.

    A plain mean, so a member whose depth could not be evaluated carries its
    NaN into the average rather than being dropped -- downstream that elects
    the most severe tier, which is the same refusal to report an unevaluated
    feature as safe that elected_tier and hazard_class make.
    """
    depth_max = np.asarray(depth_max, dtype=float)
    if depth_max.shape[0] == 0:
        return np.full(depth_max.shape[1:], np.nan)
    return depth_max.mean(axis=0)


def _marginalise_unknown(activations, unknown):
    """Point an unlinked object's activations at the building's own, in place.

    Indexed on the trailing axes, so the per-member stack and the single
    ensemble-mean pass go through this unchanged.
    """
    for name, mask in unknown.items():
        if name == "building" or not mask.any():
            continue
        activations[name][..., mask, :] = activations["building"][..., mask, :]


def _marginalise_unknown_levels(levels, unknown):
    """_marginalise_unknown for the crisp path: an unlinked object takes the
    building's CLASS instead of its activations, in place.

    The same device and the same effect -- with s = b or p = b the rule terms
    that read that object collapse into ones the building already supplies --
    only one representation earlier.
    """
    for name, mask in unknown.items():
        if name == "building" or not mask.any():
            continue
        levels[name][..., mask] = levels["building"][..., mask]


def _single_realisation_columns(depths, unknown, level_column, prefix,
                                without_street=False, crisp=False):
    """The combined FIS run once, on ONE h_max per object rather than a stack.

    Shared by the two averaged readings below. Both collapse the ensemble
    before the inference instead of after it, so both hand the rules a single
    depth field and get one tier out, with no ensemble left to measure
    agreement over -- hence no likelihood and no de-escalation (the risk matrix
    needs a likelihood band). The rule and driver come out of this pass's own
    firings, exactly as they do for the pooled pass.

    ``level_column`` is spelled out rather than derived from ``prefix`` so the
    tier can be published under a name of its own.

    ``without_street`` withholds the street antecedent through
    combined_risk_without_street, the same marginalisation the pooled "_bp_"
    pass uses. Done on the ACTIVATIONS rather than by handing the building's
    depth in as the street's: the three object types carry their own cutoffs
    (OBJECT_FUZZIFICATION), so fuzzifying a building depth on the road scale
    would be the same number only for as long as those cutoffs happen to
    coincide, which is a coincidence the module deliberately does not rely on.
    """
    if crisp:
        inputs = {
            name: hazard_class(depth_max, "road" if name == "street" else name)
            for name, depth_max in depths.items()
        }
        _marginalise_unknown_levels(inputs, unknown)
        risk = crisp_combined_risk_without_street if without_street else crisp_combined_risk
    else:
        inputs = {
            name: fuzzify(depth_max, "road" if name == "street" else name)
            for name, depth_max in depths.items()
        }
        _marginalise_unknown(inputs, unknown)
        risk = combined_risk_without_street if without_street else combined_risk

    if without_street:
        _, levels, strengths = risk(inputs["building"], inputs["property"])
    else:
        _, levels, strengths = risk(
            inputs["building"], inputs["property"], inputs["street"]
        )
    rule = dominant_rule(strengths, COMBINED_RULES)
    return {
        level_column: levels,
        f"{prefix}_rule": rule,
        f"{prefix}_driver": np.array(
            [COMBINED_RULE_DRIVER.get(label, "") for label in rule], dtype=object
        ),
    }


def _ensemble_mean_columns(inputs, unknown, prefix="ensemble_mean_hazard",
                           without_street=False, crisp=False):
    """The combined FIS on each object's ensemble-mean h_max: MEAN OF MAXIMA.

    A different question from ``overall_building_hazard_level``, not a
    smoothing of it. That one classifies every member and pools the tiers by
    vote; this averages the depths first and classifies once. The cutoffs make
    the two disagree whenever the ensemble straddles a boundary -- members at
    0.10 m and 0.50 m mean 0.30 m and elect H2 here, while the vote reports a
    tie broken to the more severe term from a 50/50 split. Which is the
    honest reading is a modelling choice, so both are reported.

    The average is taken AFTER exposure, over each object's own h_max, so the
    depth being averaged is already the deepest cell of that member's ring.
    Averaging the maps instead gives the other order and a different number --
    that is ``final_risk_ensemble_mean``, see _mean_map_columns.

    ``without_street`` runs the same average over the building and its property
    alone, published under its own prefix -- the averaged counterpart of the
    pooled "_bp_" pass, so the building-and-property configuration can be read
    with the ensemble collapsed either side of the inference.
    """
    return _single_realisation_columns(
        {name: ensemble_mean_depth(depth_max) for name, depth_max in inputs.items()},
        unknown, f"{prefix}_level", prefix, without_street=without_street,
        crisp=crisp,
    )


def _mean_map_columns(mean_map_depths, unknown, prefix="final_risk_ensemble_mean",
                      crisp=False):
    """The combined FIS on h_max read off the ENSEMBLE-MEAN FLOOD MAP.

    The whole ensemble is collapsed one step earlier than anywhere else here:
    the members' depth rasters are averaged cell by cell, exposure runs once on
    that single map, and each object's h_max is the deepest cell of its ring IN
    THE MEAN MAP. So this is the MAXIMUM OF THE MEAN, where
    ``ensemble_mean_hazard_level`` is the mean of the maxima.

    The two are not the same number and the inequality only runs one way:
    max(mean) <= mean(max), by Jensen, since the members need not put their
    deepest cell in the same place. A ring whose members each peak at 0.6 m but
    in different cells averages to a shallower map everywhere, so this reading
    is the more conservative of the two -- it reports the depth of the average
    flood rather than the average of the members' worst cells.

    ``mean_map_depths`` maps "building"/"property"/"street" to that object's
    (n_features,) h_max from the mean map, already reprojected onto the
    building rows.
    """
    return _single_realisation_columns(
        mean_map_depths, unknown, prefix, prefix, crisp=crisp,
    )


def overall_building_hazard(building, property_, street, unknown=None,
                            mean_map=None, suffix="", crisp=False):
    """Run the inference for a set of buildings and return it as a column dict.

    Each of the first three arguments is a (n_members, n_features) array of ONE
    depth indicator, all three aligned to the same features and all three the
    SAME indicator. Nothing here reads which one it was given: h_max and h_mean
    go through the same fuzzification, the same cutoffs and the same rules, and
    the caller separates the two passes with ``suffix`` (see below). Mixing
    indicators ACROSS the three objects would be a fourth thing again, and is
    not what any caller does.

    The three crisp depths follow two independent paths, and the split is the
    point of the design:

      * REPORTED, not inferred from -- each object's h_max is compared to the
        cutoffs (hazard_class) to give ``{object}_hazard_level``, so a row says
        what was measured at the footprint, the section and the street.
      * INFERRED FROM -- the same crisp h_max is fuzzified and the three sets
        of memberships go into the combined FIS. The object tiers above never
        enter it; combining happens in the fuzzy domain so a borderline depth
        keeps its partial membership right through to the output.

    Both paths run per ensemble member, so member m holds its own object
    readings and its own overall hazard level. Members are pooled only
    afterwards, which is what makes every likelihood agreement over complete
    inferences rather than over depths.

    ``unknown`` maps an input name to a boolean feature mask marking objects
    with no such link -- a building on no property parcel, say. Those inputs
    are not evidence of a dry surrounding, so instead of letting a zero depth
    fuzzify to "H1" the antecedent is marginalised out: the input takes the
    building's own activations, which in COMBINED_RULES makes every property
    term collapse (max(b, b) = b, min(b, b) = b), leaving the rule as if it had
    not been asked. The reported level and likelihood for such an input are NaN
    rather than a fabricated tier.

    The FIS is run up to four times, the passes differing only in WHERE the
    ensemble is collapsed:

      * ``overall_building_hazard_*`` -- per member, pooled by vote afterwards.
        The reported reading, and the only one carrying a likelihood.
      * ``overall_building_hazard_bp_*`` -- the same, with the street withheld
        by marginalisation (see combined_risk_without_street).
      * ``ensemble_mean_hazard_*`` -- each object's h_max averaged over members,
        then classified once: the mean of the maxima.
      * ``final_risk_ensemble_mean`` -- h_max read off the ensemble-mean flood
        map, then classified once: the maximum of the mean. Emitted only when
        ``mean_map`` is supplied, since it cannot be recovered from the member
        depths -- the averaging happens before exposure, upstream of anything
        this function is given.

    ``mean_map`` maps "building"/"property"/"street" to that object's
    (n_features,) depth from the ensemble-mean flood map, on the same features
    and already reprojected onto the building rows -- the same indicator as the
    three member stacks.

    ``suffix`` is appended to EVERY emitted column name, which is how a second
    pass on a different indicator is published without colliding with the
    reported one. Empty by default, so the h_max pass keeps the names every
    downstream consumer already reads. Applied uniformly at the end rather than
    threaded through each helper, so a column cannot acquire a name in one pass
    that it does not have in the other.
    """
    inputs = {"building": building, "property": property_, "street": street}
    activations, member_levels = {}, {}
    for name, depth_max in inputs.items():
        # "street" is the reporting name; its cutoffs are the road ones.
        object_type = "road" if name == "street" else name
        # The two paths, from the same crisp depth: memberships for the FIS,
        # and a cutoff comparison for the reported per-object tier. The crisp
        # combination needs only the second, so the fuzzification is skipped
        # rather than computed and dropped.
        if not crisp:
            activations[name] = fuzzify(depth_max, object_type)
        member_levels[name] = hazard_class(depth_max, object_type)

    unknown = {name: np.asarray(mask, dtype=bool)
               for name, mask in (unknown or {}).items() if mask is not None}
    if crisp:
        # A copy, so marginalising an absent parcel for the COMBINATION cannot
        # reach the per-object tiers reported below -- those say what was
        # measured at that object, and a building's class standing in for its
        # missing parcel is not a measurement of the parcel.
        combined_inputs = {name: levels.copy()
                           for name, levels in member_levels.items()}
        _marginalise_unknown_levels(combined_inputs, unknown)
    else:
        combined_inputs = activations
        _marginalise_unknown(combined_inputs, unknown)

    columns = {}
    for name in inputs:
        levels, likelihood, _ = hazard_likelihoods(member_levels[name])
        mask = unknown.get(name)
        if mask is not None:
            # The object was never assessed, so it has no tier to report; the
            # flag keeps that distinguishable from a genuine "H1". Emitted
            # whenever a mask was supplied, even an all-known one, so the output
            # schema does not depend on the data.
            if mask.any():
                levels = np.where(mask, np.nan, levels)
                likelihood = np.where(mask, np.nan, likelihood)
            columns[f"{name}_known"] = ~mask
        columns[f"{name}_hazard_level"] = levels
        columns[f"{name}_hazard_likelihood"] = likelihood

    # Two passes of the same combined FIS: the reported one over all three
    # objects, and a "_bp_" one with the street withheld (see
    # combined_risk_without_street). Both are combined per member and pooled
    # only afterwards, so the _bp_ pass is a genuine FIS output and not the
    # crisp OR of the two pooled object tiers.
    all_three = crisp_combined_risk if crisp else combined_risk
    pair_only = (crisp_combined_risk_without_street if crisp
                 else combined_risk_without_street)
    for prefix, pass_ in (
        ("overall_building_hazard",
         all_three(combined_inputs["building"], combined_inputs["property"],
                   combined_inputs["street"])),
        ("overall_building_hazard_bp",
         pair_only(combined_inputs["building"], combined_inputs["property"])),
    ):
        _, member_risk, strengths = pass_
        columns.update(_combined_columns(member_risk, strengths, prefix))

    # The averaged passes, with the ensemble collapsed before the inference
    # rather than after it: over the objects' h_max, and -- if the caller ran
    # exposure on the mean flood map -- over the maps themselves.
    columns.update(_ensemble_mean_columns(inputs, unknown, crisp=crisp))
    # The same mean of maxima over the building and its property alone, so the
    # BuildProp configuration has an averaged reading as well as a pooled one.
    # Both are emitted every pass; nothing downstream has to choose here.
    columns.update(_ensemble_mean_columns(
        inputs, unknown, prefix="ensemble_mean_hazard_bp", without_street=True,
        crisp=crisp))
    if mean_map is not None:
        columns.update(_mean_map_columns(mean_map, unknown, crisp=crisp))

    # Taken on the combined FIS's own per-tier probabilities, which
    # _combined_columns already emitted as overall_building_hazard_prob_*.
    combined_probs = np.stack(
        [columns[f"overall_building_hazard_prob_{term}"] for term in TERMS], axis=1
    )
    columns[f"overall_building_hazard_prob_at_least_{TERMS[MIDDLE]}"] = exceedance(
        combined_probs
    )[:, MIDDLE]
    columns["final_risk_level"] = final_risk(combined_probs)

    if suffix:
        columns = {f"{name}{suffix}": values for name, values in columns.items()}
    return columns


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

# Warning-colour hues, so the membership figure reads in the same visual
# language as the maps: safe/middle/severe as yellow/orange/red. Indexed by
# POSITION, so it serves either scale without being keyed on a term name.
_MF_PLOT_PALETTE = ("#ffdd01", "#ff6600", "#ff0e0e")
_MF_PLOT_INK = "#0b0b0b"
_MF_PLOT_MUTED = "#898781"
_MF_PLOT_GRID = "#e1e0d9"
_MF_PLOT_AXIS = "#c3c2b7"

def plot_membership_functions(object_type, ax=None, depth_max_plot=None, n=1000):
    """Plot the three depth membership functions for one object type.

    Reads the same trapezoid_breakpoints the classifier uses, so the figure
    can never drift from the live rule base.

    Parameters
    ----------
    object_type : {"building", "property", "road"}
    ax : matplotlib Axes, optional
        Axes to draw into; a new figure is created if omitted.
    depth_max_plot : float, optional
        Right edge of the x-axis in metres. The severe set plateaus out to
        DEPTH_UNIVERSE_MAX_M, so the tail is cropped; defaults to a little
        past the upper tier boundary, which is where the structure ends.
    n : int
        Sample points used to trace each membership curve.

    Returns
    -------
    (Figure, Axes)
    """
    import matplotlib.pyplot as plt

    points = trapezoid_breakpoints(object_type)
    lo_cut, hi_cut = OBJECT_FUZZIFICATION[object_type]["cutoffs"]
    if depth_max_plot is None:
        depth_max_plot = round(hi_cut * 1.7, 1)

    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 3.6))
    fig = ax.figure

    x = np.linspace(0, depth_max_plot, n)

    # ARR tier boundaries the fuzzy sets soften, drawn behind the curves.
    for cutoff in (lo_cut, hi_cut):
        ax.axvline(cutoff, color=_MF_PLOT_GRID, lw=1, ls="--", zorder=0)
        ax.text(cutoff, 1.05, f"{cutoff:.2f} m", color=_MF_PLOT_MUTED,
                fontsize=11, ha="center", va="bottom")

    for index, term in enumerate(object_terms(object_type)):
        a, b, c, d = points[term]
        y = _trapezoid(x, a, b, c, d)
        color = _MF_PLOT_PALETTE[index]
        ax.plot(x, y, lw=2.5, color=color, solid_capstyle="round", zorder=3)
        ax.fill_between(x, y, color=color, alpha=0.12, lw=0, zorder=1)
        # Label at the midpoint of the set's plateau, clipped to the visible
        # range so a set whose plateau runs off the axis is still labelled.
        label_x = np.clip((b + min(c, depth_max_plot)) / 2, 0, depth_max_plot * 0.97)
        label = TERM_LABELS[term]
        ax.text(label_x, 0.52, f"$\\mu_\\mathrm{{{label}}}$", color=_MF_PLOT_INK,
                fontsize=15, fontweight="bold", ha="center", va="bottom", zorder=4)
        ax.text(label_x, 0.46, f"$A$ = {label}", color=_MF_PLOT_MUTED,
                fontsize=11, ha="center", va="top", zorder=4)

    terms = ", ".join(TERM_LABELS[term] for term in object_terms(object_type))
    ax.set_title(
        f"$\\mu_A: X \\to [0, 1]$,   $A \\in \\{{${terms}$\\}}$,"
        f"   $X = [0,\\ {DEPTH_UNIVERSE_MAX_M:.0f}]$ m",
        color=_MF_PLOT_INK, fontsize=14, pad=26,
    )
    ax.set_xlim(0, depth_max_plot)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(
        "Flood depth $h^\\mathrm{max} \\in X$ (m)",
        color=_MF_PLOT_INK, fontsize=13,
    )
    ax.set_ylabel("$\\mu_A(h)$", color=_MF_PLOT_INK, fontsize=13)
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(colors=_MF_PLOT_MUTED, labelsize=11)
    ax.grid(axis="y", color=_MF_PLOT_GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MF_PLOT_AXIS)

    fig.tight_layout()
    return fig, ax



    # def save_impact_forecast(self, impact_forecast, flood_maps, issue_time, output_dir=OUTPUT_DIR):
    #     """
    #     Persist an impact forecast to .
    #     """
    #     issue_str = pd.to_datetime(issue_time).strftime("%Y%m%dT%H%M%S")
    #     run_dir = os.path.join(output_dir, issue_str)
    #     os.makedirs(run_dir, exist_ok=True)

    #     # One long frame per feature type, all three carrying the same
    #     # timestamp/issue_time pair so a row is always traceable to the run
    #     # that produced it as well as the hour it describes.
    #     longs = {
    #         name: self._concat_timestamped(impact_forecast[f"{name}_impact"], issue_time)
    #         for name in ("building", "road", "property")
    #     }

    #     gpkg_path = os.path.join(run_dir, f"impact_{issue_str}.gpkg")
    #     for name, layer in (("building", "buildings"), ("road", "roads"),
    #                         ("property", "properties")):
    #         # One layer per feature type, every timestamp stacked as rows.
    #         longs[name].to_file(gpkg_path, layer=layer, driver="GPKG")

    #         # Also export WGS84 GeoJSON per feature type for web/GIS tools
    #         # (e.g. ArcGIS Online / Scene Viewer), which do not read GeoPackage,
    #         # plus the same frame with a string valid_time field so the layer
    #         # can be time-enabled in ArcGIS Online / Map Viewer.
    #         self._write_geojson(longs[name], os.path.join(run_dir, f"{layer}.geojson"))
    #         self._write_timeseries_geojson(
    #             longs[name], os.path.join(run_dir, f"{layer}_timeseries.geojson")
    #         )

    #     # The inundation extent is deliberately NOT written here. get_flood_extent
    #     # computes it per issue time to be PUBLISHED (see arcgis_module.LAYERS),
    #     # and the multiband GeoTIFF written below already holds the same field
    #     # locally at full resolution -- the polygons are a classified, simplified,
    #     # dissolved reading of its p50/p90 bands, so saving them too spent ~5 MB
    #     # a run on a lossy copy of a file already in the same directory.

    #     # Summarise the ensemble per timestamp into one multiband GeoTIFF,
    #     # georeferenced the same way as the zonal sampling so it aligns with
    #     # the vector layers.
    #     transform, raster_crs, nodata = get_sample_georeference()
    #     for timestamp, flood_maps_ensemble in flood_maps.items():
    #         # Coerce (possibly torch, possibly extra dims) to a 3D float32 stack
    #         # of (n_members, H, W).
    #         ensemble = flood_maps_ensemble
    #         if hasattr(ensemble, "detach"):
    #             ensemble = ensemble.detach().cpu().numpy()
    #         ensemble = np.squeeze(np.asarray(ensemble, dtype="float32"))
    #         if ensemble.ndim != 3:
    #             raise ValueError(
    #                 f"flood map ensemble must be (n_members, H, W), got shape {ensemble.shape}"
    #             )

    #         timestamp_str = pd.to_datetime(timestamp).strftime("%Y%m%dT%H%M%S")
    #         self._write_geotiff(
    #             os.path.join(run_dir, f"flood_map_{timestamp_str}.tif"),
    #             *self._summarise_ensemble(ensemble), transform, raster_crs, nodata,
    #         )
    #     return run_dir

    # def get_flood_extent(self, flood_maps, bands=FLOOD_EXTENT_BANDS):
    #     """Inundation extent as classified depth-band polygons, per timestamp.

    #     The full-resolution rasters stay on disk; this is the form the extent
    #     travels in. Polygons publish through the same feature-layer path as the
    #     buildings, roads and properties, which is what makes the extent
    #     queryable and time-enabled beside them rather than a picture that has
    #     to be read off a legend.

    #     Both the median and the reasonable-worst-case band are emitted and
    #     tagged, so the map can show a best estimate and the spread around it as
    #     two readings of the same hour -- publishing one alone would throw away
    #     the reason the percentiles are computed.

    #     Returns ``{timestamp: GeoDataFrame}`` with one row per depth class per
    #     band, in the raster's CRS.
    #     """
    #     transform, raster_crs, _ = get_sample_georeference()
    #     extent_series = {}
    #     for timestamp, flood_maps_ensemble in flood_maps.items():
    #         ensemble = flood_maps_ensemble
    #         if hasattr(ensemble, "detach"):
    #             ensemble = ensemble.detach().cpu().numpy()
    #         ensemble = np.squeeze(np.asarray(ensemble, dtype="float32"))

    #         summary, band_names = self._summarise_ensemble(ensemble)
    #         frames = []
    #         for band in bands:
    #             depth = summary[band_names.index(band)]
    #             frame = self._depth_band_polygons(depth, transform, raster_crs)
    #             frame["band"] = band
    #             frames.append(frame)

    #         extent_series[timestamp] = gpd.GeoDataFrame(
    #             pd.concat(frames, ignore_index=True), geometry="geometry", crs=raster_crs
    #         )
    #     return extent_series

    # def get_inundation_grid(self, flood_maps, block=INUNDATION_BLOCK,
    #                         threshold=WET_DEPTH_MIN):
    #     """Inundated area as a coarsened cell grid, per timestamp.

    #     The extent polygons say where each depth CLASS reaches; this says what
    #     the depth actually is, cell by cell, in the ground between the modelled
    #     objects. Only the ensemble mean is gridded -- the spread travels as a
    #     ``depth_std`` field on the same block rather than as a second geometry,
    #     so a reader sees the best estimate and its uncertainty in one click
    #     instead of comparing two layers.

    #     Blocks are ``block`` x ``block`` model cells (see INUNDATION_BLOCK), and
    #     every block holding modelled ground is emitted -- dry ones included --
    #     so the result is a contiguous surface over the study area rather than
    #     isolated patches of water. ``threshold`` no longer decides what is
    #     emitted, only what counts towards ``wet_fraction`` and ``area_ha``.

    #     Returns ``{timestamp: GeoDataFrame}`` with one row per in-domain block,
    #     in the raster's CRS.
    #     """
    #     transform, raster_crs, _ = get_sample_georeference()
    #     grid_series = {}
    #     for timestamp, flood_maps_ensemble in flood_maps.items():
    #         ensemble = flood_maps_ensemble
    #         if hasattr(ensemble, "detach"):
    #             ensemble = ensemble.detach().cpu().numpy()
    #         ensemble = np.squeeze(np.asarray(ensemble, dtype="float32"))
    #         if ensemble.ndim != 3:
    #             raise ValueError(
    #                 f"flood map ensemble must be (n_members, H, W), got shape {ensemble.shape}"
    #             )

    #         summary, band_names = self._summarise_ensemble(ensemble)
    #         grid_series[timestamp] = self._block_grid(
    #             summary[band_names.index("depth_mean")],
    #             summary[band_names.index("depth_std")],
    #             transform, raster_crs, block, threshold,
    #         )
    #     return grid_series

    # @staticmethod
    # def _block_grid(depth, spread, transform, crs, block, threshold):
    #     """Aggregate a depth/spread raster pair into inundated block polygons."""
    #     cell_area = abs(transform.a) * abs(transform.e)
    #     rows, cols = depth.shape
    #     # Trim any partial block at the far edge rather than pad it: a padded
    #     # block would carry cells that do not exist and dilute its statistics.
    #     rows, cols = rows // block * block, cols // block * block

    #     def to_blocks(array):
    #         """(H, W) -> (H/block, W/block, block*block), cells last."""
    #         return (
    #             array[:rows, :cols]
    #             .reshape(rows // block, block, cols // block, block)
    #             .transpose(0, 2, 1, 3)
    #             .reshape(rows // block, cols // block, block * block)
    #         )

    #     depth_cells, spread_cells = to_blocks(depth), to_blocks(spread)
    #     # NaN is outside the computational domain, so it is neither wet nor a
    #     # zero depth -- nan_to_num only decides the comparison, and in_domain is
    #     # what every statistic here is measured over.
    #     in_domain = np.isfinite(depth_cells)
    #     wet = np.nan_to_num(depth_cells) >= threshold
    #     n_wet, n_domain = wet.sum(axis=-1), in_domain.sum(axis=-1)

    #     # Floored below the wet threshold before averaging, which is what
    #     # WET_DEPTH_MIN is for and what _gpu_zonal_stats already does. Sub-
    #     # threshold output is the decoder's noise around zero on dry ground, and
    #     # ~60% of in-domain cells come back very slightly NEGATIVE: averaged raw
    #     # it publishes negative depths, and clipping only the negative half
    #     # would bias every dry block upwards instead. Flooring both halves
    #     # leaves dry ground at exactly 0.0.
    #     depth_cells = np.where(wet, depth_cells, 0.0)

    #     # Every block holding any modelled ground, not just the wet ones, so the
    #     # layer is a surface covering the study area rather than islands of water
    #     # on a blank background. The domain is irregular and fills only ~39% of
    #     # its own bounding box, so this is still far from a full rectangle.
    #     selected = n_domain > 0

    #     # Averaged over the IN-DOMAIN cells, dry ones entering as zero. That
    #     # makes this a straight downsample of the raster: the surface stays
    #     # continuous across the water's edge, and a block of marginal ground
    #     # keeps a real spread instead of being flattened away. A partly-wet
    #     # block therefore reads shallow BECAUSE it is mostly dry -- depth_mean
    #     # is wet_fraction x mean-depth-where-wet, the same conflation the
    #     # WET_DEPTH_MIN comment describes -- so the two columns have to be read
    #     # together, and wet_fraction is what makes that possible.
    #     def domain_mean(cells):
    #         return np.divide(
    #             np.where(in_domain, np.nan_to_num(cells), 0.0).sum(axis=-1), n_domain,
    #             out=np.zeros(n_domain.shape, dtype=float), where=n_domain > 0,
    #         )

    #     depth_mean, depth_std = domain_mean(depth_cells), domain_mean(spread_cells)

    #     block_rows, block_cols = np.nonzero(selected)
    #     block_transform = transform * Affine.scale(block)
    #     left = block_transform.c + block_cols * block_transform.a
    #     top = block_transform.f + block_rows * block_transform.e
    #     # e is negative for a north-up raster, so top + e is the lower edge.
    #     geometry = shapely.box(
    #         left, top + block_transform.e, left + block_transform.a, top
    #     )

    #     return gpd.GeoDataFrame(
    #         {
    #             # Four places, not three: dilution puts a block holding one wet
    #             # cell in 144 at ~0.0003 m, which rounds to a flat 0.000 at
    #             # three and becomes indistinguishable from genuinely dry ground.
    #             # depth_mean == 0 has to keep meaning dry.
    #             "depth_mean": np.round(depth_mean[selected], 4),
    #             "depth_std": np.round(depth_std[selected], 4),
    #             "wet_fraction": np.round(
    #                 np.divide(
    #                     n_wet[selected], n_domain[selected],
    #                     out=np.zeros(int(selected.sum()), dtype=float),
    #                     where=n_domain[selected] > 0,
    #                 ), 3,
    #             ),
    #             # The WET area inside the block, not the block's footprint, so
    #             # the column sums to the true inundated area instead of the
    #             # area of every block the water clipped a corner of.
    #             "area_ha": np.round(n_wet[selected] * cell_area / 1e4, 4),
    #         },
    #         geometry=geometry, crs=crs,
    #     )

    # @staticmethod
    # def _depth_band_polygons(depth, transform, crs,
    #                          breaks=DEPTH_BREAKS_M, min_area_m2=MIN_EXTENT_POLYGON_M2):
    #     """Vectorise one depth raster into one multipolygon per depth class."""
    #     # NaN marks cells outside the computational domain. digitize would send
    #     # them to the top bin, painting the whole dry domain as the deepest
    #     # class, so they are masked to 0 (nothing) before classifying rather
    #     # than after.
    #     finite = np.isfinite(depth)
    #     classes = np.digitize(np.where(finite, depth, -1.0), breaks).astype("int16")
    #     classes[~finite] = 0

    #     shapes = [
    #         (shape(geom), int(value))
    #         for geom, value in features.shapes(
    #             classes, mask=classes > 0, transform=transform
    #         )
    #     ]
    #     frame = gpd.GeoDataFrame(
    #         {"depth_class": [value for _, value in shapes]},
    #         geometry=[geom for geom, _ in shapes],
    #         crs=crs,
    #     )

    #     # Single-cell speckle is most of the polygon count and none of the
    #     # extent -- dropping it below a tenth of a hectare cuts the shape count
    #     # by ~85% without moving the mapped area.
    #     frame = frame[frame.area >= min_area_m2].copy()
    #     if frame.empty:
    #         return gpd.GeoDataFrame(
    #             {"depth_class": [], "depth_min_m": [], "depth_max_m": [], "area_ha": []},
    #             geometry=[], crs=crs,
    #         )

    #     # Half a cell: enough to drop the staircase edge the rasterisation
    #     # introduces, not enough to move a boundary by a cell.
    #     frame["geometry"] = frame.simplify(2.0)
    #     # buffer(0) repairs the self-intersections simplification can leave
    #     # behind; AGOL rejects an invalid ring at push time, not at write time.
    #     invalid = ~frame.is_valid
    #     if invalid.any():
    #         frame.loc[invalid, "geometry"] = frame.loc[invalid, "geometry"].buffer(0)

    #     dissolved = frame.dissolve(by="depth_class").reset_index()
    #     # The class bounds travel with the row so the layer explains its own
    #     # symbology instead of depending on a legend kept somewhere else.
    #     upper = list(breaks[1:]) + [None]
    #     dissolved["depth_min_m"] = [breaks[c - 1] for c in dissolved["depth_class"]]
    #     dissolved["depth_max_m"] = [upper[c - 1] for c in dissolved["depth_class"]]
    #     dissolved["area_ha"] = dissolved.area / 1e4
    #     return dissolved

    # @staticmethod
    # def _summarise_ensemble(ensemble):
    #     """Reduce an (n_members, H, W) depth ensemble to the reported bands.

    #     A mean and a spread say how uncertain a cell is but not where the
    #     plausible depths actually lie, and the spread is only interpretable
    #     relative to the depth it sits on. The percentiles put the same
    #     uncertainty back in metres: p10-p90 is the range to plan against, read
    #     directly against the ARR class limits the hazard classes use, and p90 is
    #     the reasonable worst case a warning should be built on.

    #     Returns a (5, H, W) stack and the matching band names.
    #     """
    #     # These are nanpercentile's semantics -- a cell outside the computational
    #     # domain is nodata, not a zero depth -- but taken over the in-domain
    #     # cells only rather than by calling nanpercentile on the whole grid.
    #     # numpy has no vectorised NaN-aware percentile, so that call falls back
    #     # to a Python-level pass over all 5.9M columns and costs ~50 s per
    #     # timestamp, even though ~60% of the grid lies outside the domain and
    #     # can only ever be NaN. Masking first puts the rest on the C path and
    #     # returns the same values bit for bit in ~0.6 s.
    #     finite = np.isfinite(ensemble)
    #     complete = finite.all(axis=0)
    #     percentiles = np.full((3,) + ensemble.shape[1:], np.nan, dtype="float32")
    #     if complete.any():
    #         percentiles[:, complete] = np.percentile(
    #             ensemble[:, complete], [10, 50, 90], axis=0
    #         )

    #     # A cell finite in some members but not all cannot take the fast path.
    #     # The decoder has never produced one -- every member shares the same
    #     # domain mask, so a cell is NaN in all ten or none -- but routing any
    #     # through nanpercentile keeps this an optimisation of the existing
    #     # reduction rather than a redefinition of it.
    #     partial = finite.any(axis=0) & ~complete
    #     if partial.any():
    #         percentiles[:, partial] = np.nanpercentile(
    #             ensemble[:, partial], [10, 50, 90], axis=0
    #         )

    #     p10, p50, p90 = percentiles
    #     bands = np.stack(
    #         [ensemble.mean(axis=0), ensemble.std(axis=0), p10, p50, p90]
    #     ).astype("float32")
    #     return bands, ("depth_mean", "depth_std", "depth_p10", "depth_p50", "depth_p90")

    # @staticmethod
    # def _concat_timestamped(impact_series, issue_time=None):
    #     """Concatenate a ``{timestamp: GeoDataFrame}`` dict (get_impact_forecast's
    #     per-feature-type output) into one long-format GeoDataFrame with added
    #     ``timestamp`` and ``issue_time`` columns (datetime64) -- one row per
    #     feature per timestamp.

    #     ``issue_time`` is stamped alongside the valid time because the two are
    #     not recoverable from each other: successive hourly runs forecast
    #     overlapping valid times, and without the issue time an appended layer
    #     cannot tell a re-run from a new forecast.
    #     """
    #     if not impact_series:
    #         raise ValueError(
    #             "empty impact series -- get_impact_forecast returned no frames for "
    #             "this feature type, so there is nothing to concatenate"
    #         )
    #     frames = []
    #     for timestamp, gdf in impact_series.items():
    #         frame = gdf.copy()
    #         frame["timestamp"] = pd.Timestamp(timestamp)
    #         if issue_time is not None:
    #             frame["issue_time"] = pd.Timestamp(issue_time)
    #         frames.append(frame)
    #     first = next(iter(impact_series.values()))
    #     return gpd.GeoDataFrame(
    #         pd.concat(frames, ignore_index=True), geometry=first.geometry.name, crs=first.crs
    #     )

    # def _write_geojson(self, gdf, path):
    #     """Write a GeoDataFrame to WGS84 GeoJSON for web/GIS consumption.

    #     Reprojects to EPSG:4326 and stringifies datetime columns (e.g.
    #     ``timestamp``, or an onset column with NaT) so they serialise cleanly.
    #     """
    #     out = gdf.to_crs(epsg=4326).copy()
    #     for col in out.columns:
    #         if pd.api.types.is_datetime64_any_dtype(out[col]):
    #             out[col] = out[col].astype(str)
    #     out.to_file(path, driver="GeoJSON")

    # def _write_timeseries_geojson(self, impact_long, path, use_centroids=False):
    #     """Write a long-format WGS84 GeoJSON with a string ``valid_time`` field
    #     (derived from the ``timestamp`` column), letting ArcGIS Online /
    #     Map Viewer time-enable the layer.

    #     Parameters:
    #     impact_long (GeoDataFrame): Long-format impact output (one row per
    #         feature per timestamp, with a ``timestamp`` column) -- see
    #         :meth:`_concat_timestamped`.
    #     path (str): Destination GeoJSON path.
    #     use_centroids (bool): Emit centroids (points) instead of polygons, to
    #         keep the (heavily duplicated across timestamps) geometry small.
    #     """
    #     out = impact_long.copy()
    #     if use_centroids:
    #         out[out.geometry.name] = out.geometry.centroid
    #     out = out.to_crs(epsg=4326)
    #     out["valid_time"] = pd.to_datetime(out["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    #     out = out.drop(columns=["timestamp"])
    #     if "issue_time" in out.columns:
    #         out["issue_time"] = pd.to_datetime(out["issue_time"]).dt.strftime(
    #             "%Y-%m-%dT%H:%M:%S"
    #         )
    #     out.to_file(path, driver="GeoJSON")

    # def _write_geotiff(self, path, array, band_names, transform, crs, nodata):
    #     """Write a 2D array or an (n_bands, H, W) stack to a GeoTIFF at ``path``.

    #     ``band_names`` are set as band descriptions so a multiband file is
    #     self-describing -- "Band 3" in QGIS is not a useful label for a
    #     percentile, and the alternative is a naming convention nobody outside
    #     this module knows about.
    #     """
    #     array = np.asarray(array)
    #     if array.ndim == 2:
    #         array = array[np.newaxis]
    #     band_names = (band_names,) if isinstance(band_names, str) else tuple(band_names)
    #     if len(band_names) != array.shape[0]:
    #         raise ValueError(
    #             f"got {len(band_names)} band names for {array.shape[0]} bands"
    #         )

    #     with rasterio.open(
    #         path,
    #         'w',
    #         driver='GTiff',
    #         height=array.shape[1],
    #         width=array.shape[2],
    #         count=array.shape[0],
    #         dtype=array.dtype,
    #         crs=crs,
    #         transform=transform,
    #         nodata=nodata,
    #     ) as dst:
    #         for index, name in enumerate(band_names, start=1):
    #             dst.write(array[index - 1], index)
    #             dst.set_band_description(index, name)
