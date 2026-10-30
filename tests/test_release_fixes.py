# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Problems and questions raised after the 0.6.0 release."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tenqsp as tq

NO_STATES = "model algebraic_pk\nparam C0=10, kel=0.2\nobservable conc = C0*exp(-kel*t)\n"


@pytest.mark.parametrize("text", [NO_STATES, "model empty\n", ""])
def test_a_model_with_no_states_is_refused_in_plain_words(text):
    with pytest.raises(tq.ModelError, match=r"has no states: no equation d_t\(\.\.\.\)"):
        tq.Model.from_text(text)


def test_a_python_model_with_no_states_is_refused_in_plain_words():
    m = tq.Model("algebraic_pk")
    with pytest.raises(tq.ModelError, match=r"m\.ode\(X, \.\.\.\) in Python"):
        m.validate()


ORAL = """model oral
param CL=4, V=30, ka=1
d_t(gut) = -ka*gut
d_t(A) = ka*gut - CL/V*A
observable conc = A/V
"""
RTOL = 1e-8


@pytest.mark.timeout(300)
def test_padding_with_zero_doses_and_extra_times_changes_results_within_tolerance():
    """For many subjects in one shape: doses of amount 0 and output times nobody reads
    add segment boundaries and stopping points, so the solver's steps change; the results
    agree to the solver's tolerance, not to round-off. Values and gradients alike."""
    m = tq.Model.from_text(ORAL)
    sim = m.simulator(outputs=["conc"], rtol=RTOL, atol=1e-12)
    doses = tq.bolus("gut", 100, times=[0, 12])
    padded = doses + tq.bolus("gut", 0.0, times=[5, 30])
    times = [1.0, 6.0, 24.0]
    plain = tq.Scenario(regimen=doses, times=times)
    wide = tq.Scenario(regimen=padded, times=[1.0, 2.5, 6.0, 9.0, 24.0])
    pick = [0, 2, 4]

    a = np.asarray(sim(plain)["conc"])
    b = np.asarray(sim(wide)["conc"])[pick]
    np.testing.assert_allclose(b, a, rtol=100 * RTOL)

    f = sim.function(plain, wrt=["CL", "V", "ka"])
    g = sim.function(wide, wrt=["CL", "V", "ka"])
    x = jnp.asarray(np.asarray(f.x0))
    np.testing.assert_allclose(np.asarray(g(x)["conc"])[pick], a, rtol=100 * RTOL)
    ja = np.asarray(jax.jacfwd(lambda x: f(x)["conc"])(x))
    jb = np.asarray(jax.jacfwd(lambda x: g(x)["conc"][jnp.asarray(pick)])(x))
    np.testing.assert_allclose(jb, ja, rtol=1e-5, atol=1e-9 * np.abs(ja).max())


DOSES = {
    "ID": [1, 1, 1, 2, 2, 3],
    "TIME": [0, 12, 6, 0, 24, 0],
    "AMT": [100, 100, 0, 200, 50, 0],
    "CMT": [1, 1, 2, 1, 1, 2],
    "RATE": [0, 0, 0, 0, 25, 0],
    "EVID": [1, 1, 0, 1, 1, 0],
}
TARGETS = {1: "depot", 2: "centr"}


def test_regimens_from_table_gives_each_subject_its_own_regimen():
    got = tq.regimens_from_table(DOSES, targets=TARGETS)
    assert list(got) == [1, 2, 3]  # in the order the subjects first appear
    for sid in (1, 2, 3):
        rows = [i for i, v in enumerate(DOSES["ID"]) if v == sid]
        alone = tq.regimen_from_table(
            {k: [v[i] for i in rows] for k, v in DOSES.items()}, targets=TARGETS
        )
        assert repr(got[sid]) == repr(alone)
    assert len(list(got[3])) == 0  # observations only: an empty regimen
    assert repr(got[2]) == repr(
        tq.bolus("depot", 200, times=[0]) + tq.infusion("depot", amount=50, rate=25, times=[24])
    )


def test_regimens_from_table_keys_and_refusals():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"ID": [1.0, 1.0, 2.0], "TIME": [0, 1, 0], "AMT": [1, 1, 1], "CMT": "c"})
    assert list(tq.regimens_from_table(df)) == [1, 2]  # whole-number IDs as int
    assert list(tq.regimens_from_table({**DOSES, "ID": ["a", "a", "a", "b", "b", "c"]},
                                       targets=TARGETS)) == ["a", "b", "c"]  # fmt: skip
    no_id = {k: v for k, v in DOSES.items() if k != "ID"}
    with pytest.raises(ValueError, match="needs an ID column"):
        tq.regimens_from_table(no_id, targets=TARGETS)
    with pytest.raises(ValueError, match="Row 1: the ID is missing"):
        tq.regimens_from_table({**DOSES, "ID": [1, float("nan"), 1, 2, 2, 3]}, targets=TARGETS)
    with pytest.raises(ValueError, match="tq.regimens_from_table"):
        tq.regimen_from_table(DOSES, targets=TARGETS)
