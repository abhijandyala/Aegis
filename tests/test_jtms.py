"""Acceptance tests for the pure-Python justification-based TMS.

ASCII only in every assertion string: ``->``, never a Unicode arrow.
"""

from __future__ import annotations

import pytest

from aegis import jtms


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def jt():
    """Return a clean evidence graph for each test."""
    jtms.reset()
    return jtms


def build_mmsi_spoof(jt) -> None:
    """Build the shared MMSI-spoof fixture."""
    jt.build_mmsi_spoof_demo()


# ===========================================================================
# ACCEPTANCE 1: the worked MMSI-spoof example, in full
# ===========================================================================

def test_graph_construction_alone_already_knocks_out_identity_b_via_refutes(jt):
    """Before any retract() call: propagate() alone, right after building the
    graph, already blocks identity_b_confirmed via the refutes edge from
    spoof_detected -- broadcast_b_mmsi is STILL believed the whole time. This
    is the sharper demo claim: the system notices on graph construction /
    first propagation, not because of any retraction.
    """
    build_mmsi_spoof(jt)

    # Raw, pre-propagation state: every Conclusion still at its literal "IN"
    # node-default, per the Conclusion node docstring.
    assert jt.conclusion_status("identity_a_confirmed") == "IN"
    assert jt.conclusion_status("identity_b_confirmed") == "IN"
    assert jt.conclusion_status("spoof_detected") == "IN"

    flipped = jt.propagate()

    assert flipped == ["identity_b_confirmed"]
    assert jt.conclusion_status("identity_b_confirmed") == "OUT"
    assert jt.conclusion_status("spoof_detected") == "IN"
    assert jt.conclusion_status("identity_a_confirmed") == "IN"

    # The raw Fact is untouched -- the block came entirely from `refutes`.
    assert jt.fact_believed("broadcast_b_mmsi") is True


def test_retract_returns_flipped_ids_naming_identity_b_confirmed(jt):
    """retract('broadcast_b_mmsi'), as the FIRST-EVER propagation on a fresh
    graph, returns identity_b_confirmed in its flipped list. spoof_detected
    does NOT flip (it computes to IN, which is also its raw default -- "it
    was already IN"). identity_a_confirmed never flips.
    """
    build_mmsi_spoof(jt)

    flipped = jt.retract("broadcast_b_mmsi")

    assert flipped == ["identity_b_confirmed"]
    assert "spoof_detected" not in flipped
    assert "identity_a_confirmed" not in flipped

    assert jt.conclusion_status("identity_b_confirmed") == "OUT"
    assert jt.conclusion_status("spoof_detected") == "IN"
    assert jt.conclusion_status("identity_a_confirmed") == "IN"
    assert jt.fact_believed("broadcast_b_mmsi") is False


def test_identity_a_confirmed_never_flips_across_the_whole_scenario(jt):
    build_mmsi_spoof(jt)

    f1 = jt.propagate()
    f2 = jt.retract("broadcast_b_mmsi")
    f3 = jt.reinstate("broadcast_b_mmsi")

    assert "identity_a_confirmed" not in f1
    assert "identity_a_confirmed" not in f2
    assert "identity_a_confirmed" not in f3
    assert jt.conclusion_status("identity_a_confirmed") == "IN"


def test_spoof_detected_has_its_own_justification_independent_of_b(jt):
    """spoof_detected is justified by physically_impossible AND EITHER
    broadcast -- retracting b's broadcast does not touch it, because
    j_spoof_a (keyed on broadcast_a_mmsi) is untouched.
    """
    build_mmsi_spoof(jt)
    jt.propagate()

    jt.retract("broadcast_b_mmsi")
    # j_spoof_b (keyed on broadcast_b_mmsi) is dead, but j_spoof_a (keyed on
    # broadcast_a_mmsi) still stands, so spoof_detected is untouched.
    assert jt.conclusion_status("spoof_detected") == "IN"

    # Retracting the OTHER broadcast now kills j_spoof_a too -- both of
    # spoof_detected's OR'd justifications are gone, so THIS is the call that
    # finally takes it (and identity_a_confirmed, its own only support) out.
    flipped = jt.retract("broadcast_a_mmsi")
    assert "spoof_detected" in flipped
    assert jt.conclusion_status("spoof_detected") == "OUT"
    assert "identity_a_confirmed" in flipped
    assert jt.conclusion_status("identity_a_confirmed") == "OUT"


# ===========================================================================
# ACCEPTANCE 2: reinstate() is the symmetric, and non-trivial, case
# ===========================================================================

def test_reinstate_is_straightforward_when_nothing_else_blocks(jt):
    """A Conclusion with no refutes edge: retract, then reinstate its only
    antecedent, and it flips OUT then back IN."""
    jt.add_fact("f1", "the one and only premise")
    jt.add_conclusion("c1", "derived from f1")
    jt.justify("c1", "j1", supports_ids=["f1"])
    jt.propagate()
    assert jt.conclusion_status("c1") == "IN"

    flipped_out = jt.retract("f1")
    assert flipped_out == ["c1"]
    assert jt.conclusion_status("c1") == "OUT"

    flipped_in = jt.reinstate("f1")
    assert flipped_in == ["c1"]
    assert jt.conclusion_status("c1") == "IN"


def test_reinstate_stays_blocked_while_an_inhibitor_is_still_in(jt):
    """The meaningful reinstate case: identity_b_confirmed's antecedent comes
    back, but spoof_detected (the inhibitor) is still IN, so it stays OUT.
    """
    build_mmsi_spoof(jt)
    jt.propagate()
    jt.retract("broadcast_b_mmsi")
    assert jt.conclusion_status("identity_b_confirmed") == "OUT"

    flipped = jt.reinstate("broadcast_b_mmsi")

    # broadcast_b_mmsi is believed again, but spoof_detected (still IN, via
    # j_spoof_a on broadcast_a_mmsi) still refutes j_b1.
    assert flipped == []
    assert jt.conclusion_status("identity_b_confirmed") == "OUT"
    assert jt.fact_believed("broadcast_b_mmsi") is True


def test_reinstate_flips_back_in_once_the_inhibitor_is_also_gone(jt):
    """Retract BOTH the antecedent and (transitively) the inhibitor's own
    support, then reinstate just the antecedent: now nothing blocks it.
    """
    build_mmsi_spoof(jt)
    jt.propagate()
    jt.retract("broadcast_b_mmsi")
    assert jt.conclusion_status("identity_b_confirmed") == "OUT"

    # Kill spoof_detected entirely: its only remaining justification
    # (j_spoof_a) needs physically_impossible.
    flipped = jt.retract("physically_impossible")
    assert "spoof_detected" in flipped
    assert jt.conclusion_status("spoof_detected") == "OUT"
    # identity_b_confirmed is STILL out -- its own antecedent is still gone.
    assert jt.conclusion_status("identity_b_confirmed") == "OUT"

    flipped = jt.reinstate("broadcast_b_mmsi")
    assert flipped == ["identity_b_confirmed"]
    assert jt.conclusion_status("identity_b_confirmed") == "IN"


# ===========================================================================
# ACCEPTANCE 3: OR across justifications
# ===========================================================================

def test_two_justifications_or_survives_losing_one(jt):
    jt.add_fact("fa", "premise A")
    jt.add_fact("fb", "premise B")
    jt.add_conclusion("c1", "derived from A or B")
    jt.justify("c1", "jA", supports_ids=["fa"])
    jt.justify("c1", "jB", supports_ids=["fb"])
    jt.propagate()
    assert jt.conclusion_status("c1") == "IN"

    flipped = jt.retract("fa")
    assert flipped == []
    assert jt.conclusion_status("c1") == "IN"


def test_two_justifications_or_fails_only_when_both_lost(jt):
    jt.add_fact("fa", "premise A")
    jt.add_fact("fb", "premise B")
    jt.add_conclusion("c1", "derived from A or B")
    jt.justify("c1", "jA", supports_ids=["fa"])
    jt.justify("c1", "jB", supports_ids=["fb"])
    jt.propagate()

    jt.retract("fa")
    assert jt.conclusion_status("c1") == "IN"

    flipped = jt.retract("fb")
    assert flipped == ["c1"]
    assert jt.conclusion_status("c1") == "OUT"


# ===========================================================================
# ACCEPTANCE 4: AND within one justification
# ===========================================================================

def test_and_within_one_justification_flips_on_either_antecedent(jt):
    jt.add_fact("fa", "premise A")
    jt.add_fact("fb", "premise B")
    jt.add_conclusion("c1", "derived from A and B")
    jt.justify("c1", "j1", supports_ids=["fa", "fb"])
    jt.propagate()
    assert jt.conclusion_status("c1") == "IN"

    assert jt.retract("fa") == ["c1"]
    assert jt.conclusion_status("c1") == "OUT"

    jt.reset()
    jt.add_fact("fa", "premise A")
    jt.add_fact("fb", "premise B")
    jt.add_conclusion("c1", "derived from A and B")
    jt.justify("c1", "j1", supports_ids=["fa", "fb"])
    jt.propagate()

    assert jt.retract("fb") == ["c1"]
    assert jt.conclusion_status("c1") == "OUT"


# ===========================================================================
# ACCEPTANCE 5: fixpoint propagation through a chain, in ONE retract() call
# ===========================================================================

def test_chain_propagates_to_fixpoint_in_a_single_retract_call(jt):
    """Fact -> Conclusion A -> (A supports Conclusion B). Retracting the Fact
    must flip BOTH A and B in one retract() call, not two."""
    jt.add_fact("f1", "the root premise")
    jt.add_conclusion("cA", "A, derived from f1")
    jt.add_conclusion("cB", "B, derived from A")
    jt.justify("cA", "jA", supports_ids=["f1"])
    jt.justify("cB", "jB", supports_ids=["cA"])
    jt.propagate()
    assert jt.conclusion_status("cA") == "IN"
    assert jt.conclusion_status("cB") == "IN"

    flipped = jt.retract("f1")

    assert sorted(flipped) == ["cA", "cB"]
    assert jt.conclusion_status("cA") == "OUT"
    assert jt.conclusion_status("cB") == "OUT"


def test_longer_chain_of_four_flips_all_in_one_call(jt):
    jt.add_fact("f1", "root")
    jt.add_conclusion("c1", "derived 1")
    jt.add_conclusion("c2", "derived 2")
    jt.add_conclusion("c3", "derived 3")
    jt.justify("c1", "j1", supports_ids=["f1"])
    jt.justify("c2", "j2", supports_ids=["c1"])
    jt.justify("c3", "j3", supports_ids=["c2"])
    jt.propagate()
    assert jt.conclusion_status("c3") == "IN"

    flipped = jt.retract("f1")
    assert sorted(flipped) == ["c1", "c2", "c3"]


# ===========================================================================
# ACCEPTANCE 6: the iteration cap on a cyclic justification graph
# ===========================================================================

def test_cyclic_justification_graph_raises_instead_of_hanging(jt):
    """A refutes B, B refutes A, nothing else grounds either one: a classic
    JTMS oscillation. propagate() must raise a clear error within MAX_PASSES
    rather than looping forever or crashing unhelpfully.
    """
    jt.add_conclusion("cA", "mutually refuting A")
    jt.add_conclusion("cB", "mutually refuting B")
    jt.justify("cA", "jA", refutes_ids=["cB"])
    jt.justify("cB", "jB", refutes_ids=["cA"])

    with pytest.raises(RuntimeError) as excinfo:
        jt.propagate()

    msg = str(excinfo.value)
    assert "fixpoint" in msg
    assert str(jt.MAX_PASSES) in msg


# ===========================================================================
# ACCEPTANCE 7: explain() -- structure correctness before/after retraction
# ===========================================================================

def test_explain_reports_satisfied_and_unsatisfied_correctly(jt):
    build_mmsi_spoof(jt)
    jt.propagate()

    before = jt.explain("identity_b_confirmed")
    assert before["status"] == "OUT"
    assert len(before["justifications"]) == 1
    j = before["justifications"][0]
    assert j["justification_id"] == "j_b1"
    assert j["satisfied"] is False
    # The antecedent (broadcast_b_mmsi) is still fine on its own...
    ant = j["antecedents"][0]
    assert ant["id"] == "broadcast_b_mmsi"
    assert ant["in_or_believed"] is True
    assert ant["satisfied"] is True
    # ...but the inhibitor (spoof_detected) is IN, so it is NOT satisfied.
    inhib = j["inhibitors"][0]
    assert inhib["id"] == "spoof_detected"
    assert inhib["in_or_believed"] is True
    assert inhib["satisfied"] is False

    jt.retract("broadcast_b_mmsi")
    after = jt.explain("identity_b_confirmed")
    assert after["status"] == "OUT"
    j2 = after["justifications"][0]
    ant2 = j2["antecedents"][0]
    assert ant2["in_or_believed"] is False
    assert ant2["satisfied"] is False


def test_explain_on_identity_a_confirmed_shows_a_satisfied_justification(jt):
    build_mmsi_spoof(jt)
    jt.propagate()

    exp = jt.explain("identity_a_confirmed")
    assert exp["status"] == "IN"
    assert exp["justifications"][0]["satisfied"] is True
    assert exp["justifications"][0]["antecedents"][0]["id"] == "broadcast_a_mmsi"


# ===========================================================================
# ACCEPTANCE 8: brief_for() -- deterministic, non-empty, ASCII-safe
# ===========================================================================

def test_brief_for_is_ascii_and_non_empty(jt):
    build_mmsi_spoof(jt)
    jt.propagate()
    jt.retract("broadcast_b_mmsi")

    for concl_id in ("identity_a_confirmed", "identity_b_confirmed", "spoof_detected"):
        line = jt.brief_for(concl_id)
        assert isinstance(line, str)
        assert len(line) > 0
        line.encode("ascii")  # raises UnicodeEncodeError if not ASCII-safe
        assert "→" not in line  # never a unicode arrow


def test_brief_for_shows_the_transition_when_the_last_propagate_flipped_it(jt):
    build_mmsi_spoof(jt)
    jt.propagate()

    flipped_brief = jt.brief_for("identity_b_confirmed")
    assert "IN -> OUT" in flipped_brief

    stable_brief = jt.brief_for("identity_a_confirmed")
    assert "->" not in stable_brief
    assert "IN" in stable_brief


def test_brief_for_is_deterministic_for_the_same_state(jt):
    build_mmsi_spoof(jt)
    jt.propagate()
    jt.retract("broadcast_b_mmsi")

    first = jt.brief_for("identity_b_confirmed")
    second = jt.brief_for("identity_b_confirmed")
    assert first == second


# ===========================================================================
# ACCEPTANCE 9: reset() actually clears state
# ===========================================================================

def test_reset_clears_state_between_scenario_runs(jt):
    build_mmsi_spoof(jt)
    jt.propagate()
    assert jt.conclusion_status("identity_b_confirmed") == "OUT"
    assert jt.fact_ids() != []

    jt.reset()

    assert jt.fact_ids() == []
    assert jt.conclusion_ids() == []

    # Second run in the same session: no pollution from the first.
    build_mmsi_spoof(jt)
    assert jt.conclusion_status("identity_b_confirmed") == "IN"  # raw default again
    flipped = jt.propagate()
    assert flipped == ["identity_b_confirmed"]
    assert jt.conclusion_status("identity_b_confirmed") == "OUT"


def test_duplicate_ids_raise_clearly(jt):
    jt.add_fact("f1", "premise")
    with pytest.raises(ValueError):
        jt.add_fact("f1", "premise again")

    jt.add_conclusion("c1", "derived")
    with pytest.raises(ValueError):
        jt.add_conclusion("c1", "derived again")

    jt.justify("c1", "j1", supports_ids=["f1"])
    with pytest.raises(ValueError):
        jt.justify("c1", "j1", supports_ids=["f1"])
