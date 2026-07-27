import jax.numpy as jnp
from types import SimpleNamespace

from src.algorithms.reppo.ff_reppo import (
    invariance_aux_loss,
    batch_orthonormality_loss,
    gershgorin_loss,
    successor_feature_td_loss,
    compute_successor_start_mean,
    fit_sr_dice_ratio,
)

# ---------------------------------------------------------------------------
# Fixed hand-checkable tensors.
#
# Batch size B = 4
# Feature dimension d = 8
#
# Phi, Phi_next, Phi_pred, Phi_start are [4, 8].
# S must be [8, 8], because psi = Phi @ S.
# nu is [8].
# ---------------------------------------------------------------------------

PHI = jnp.array([
    [2., 0., 0., 0., 0., 0., 0., 0.],
    [0., 2., 0., 0., 0., 0., 0., 0.],
    [0., 0., 2., 0., 0., 0., 0., 0.],
    [0., 0., 0., 2., 0., 0., 0., 0.],
], dtype=jnp.float32)

NEXT_PHI = jnp.array([
    [1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0., 0., 0.],
    [0., 0., 0., 1., 0., 0., 0., 0.],
], dtype=jnp.float32)

PRED_PHI = jnp.array([
    [2., 0., 0., 0., 0., 0., 0., 0.],
    [0., 3., 0., 0., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0., 0., 0.],
    [0., 0., 0., -1., 0., 0., 0., 0.],
], dtype=jnp.float32)

START_PHI = jnp.array([
    [1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0., 0., 0.],
    [0., 0., 0., 1., 0., 0., 0., 0.],
], dtype=jnp.float32)

S = jnp.diag(jnp.array(
    [0.5, 1.0, 1.5, 2.0, 0.25, -0.5, 0.75, 1.25],
    dtype=jnp.float32,
))

NU = jnp.array(
    [1.0, -1.0, 0.5, 2.0, 0.0, 0.0, 0.0, 0.0],
    dtype=jnp.float32,
)

WEIGHTS = jnp.array([0.25, 0.25, 0.25, 0.25], dtype=jnp.float32)
CONTINUATION = jnp.array([1.0, 1.0, 0.0, 1.0], dtype=jnp.float32)

GAMMA = 0.5
EPS = 1e-5


def check_scalar(name, actual, expected):
    actual = float(actual)
    print(f"\n{name}")
    print(f"actual_loss   = {actual:.10f}")
    print(f"expected_loss = {expected:.10f}")
    print(f"match         = {abs(actual - expected) < 1e-6}")


def test_invariance():
    # Residual squared norms:
    # row 1: 1
    # row 2: 4
    # row 3: 0
    # row 4: 4
    #
    # Per-sample 1/2 ||residual||^2:
    # [0.5, 2.0, 0.0, 2.0]
    #
    # Uniform weighted mean:
    # 0.25 * (0.5 + 2 + 0 + 2) = 1.125
    expected = 1.125

    actual, _ = invariance_aux_loss(
        PRED_PHI,
        NEXT_PHI,
        WEIGHTS,
    )

    check_scalar("INVARIANCE LOSS", actual, expected)


def test_orthonormality():
    # Phi^T Xi Phi =
    # diag(1,1,1,1,0,0,0,0)
    #
    # Difference from I_8 has four entries equal to -1.
    # Sum squared error = 4.
    #
    # Loss = 0.5 * 4 / 64 = 0.03125
    expected = 0.03125

    expected_gram = jnp.diag(jnp.array(
        [1., 1., 1., 1., 0., 0., 0., 0.],
        dtype=jnp.float32,
    ))

    actual, gram, _ = batch_orthonormality_loss(
        PHI,
        WEIGHTS,
    )

    check_scalar("ORTHONORMALITY LOSS", actual, expected)
    print(f"gram_shape    = {gram.shape}")
    print(f"gram_matches_expected = {bool(jnp.allclose(gram, expected_gram))}")
    print(f"gram_is_identity      = {bool(jnp.allclose(gram, jnp.eye(8)))}")


def test_gershgorin():
    # TD features = Phi - gamma * continuation * NextPhi
    #
    # With gamma=0.5 and continuation=[1,1,0,1]:
    # row amplitudes become [1.5, 1.5, 2.0, 1.5].
    #
    # A = Phi^T Xi TD =
    # diag(0.75,0.75,1.0,0.75,0,0,0,0)
    #
    # Off-diagonal radii are all zero.
    # First 4 rows have positive margin and zero violation.
    # Last 4 rows have margin=0 and violation=eps=1e-5.
    #
    # Loss = (4 * 1e-5) / 64 = 6.25e-7
    expected = 6.25e-7

    expected_td_matrix = jnp.diag(jnp.array(
        [0.75, 0.75, 1.0, 0.75, 0., 0., 0., 0.],
        dtype=jnp.float32,
    ))

    actual, td_matrix, _ = gershgorin_loss(
        PHI,
        NEXT_PHI,
        WEIGHTS,
        CONTINUATION,
        gamma=GAMMA,
        eps=EPS,
    )

    check_scalar("GERSHGORIN LOSS", actual, expected)
    print(f"td_matrix_shape = {td_matrix.shape}")
    print(
        "td_matrix_matches_expected = "
        f"{bool(jnp.allclose(td_matrix, expected_td_matrix))}"
    )


def test_successor():
    # S = diag(0.5, 1.0, 1.5, 2.0, ...)
    #
    # psi = Phi @ S has active amplitudes:
    # [1, 2, 3, 4]
    #
    # next_psi = NextPhi @ S:
    # [0.5, 1, 1.5, 2]
    #
    # target = Phi + gamma * continuation * next_psi
    #
    # row 1 target amplitude = 2 + 0.5*0.5 = 2.25
    # row 2 target amplitude = 2 + 0.5*1.0 = 2.50
    # row 3 target amplitude = 2             = 2.00
    # row 4 target amplitude = 2 + 0.5*2.0 = 3.00
    #
    # residual amplitudes:
    # [-1.25, -0.5, 1.0, 1.0]
    #
    # per-sample losses:
    # [0.78125, 0.125, 0.5, 0.5]
    #
    # weighted mean =
    # 0.25 * (0.78125 + 0.125 + 0.5 + 0.5)
    # = 0.4765625
    expected = 0.4765625

    actual, _ = successor_feature_td_loss(
        PHI,
        NEXT_PHI,
        WEIGHTS,
        CONTINUATION,
        S,
        gamma=GAMMA,
    )

    check_scalar("SUCCESSOR TD LOSS", actual, expected)


def test_ratio():
    # START_PHI @ S rows have active amplitudes:
    # [0.5, 1.0, 1.5, 2.0]
    #
    # mean over 4 rows:
    # m = [0.125, 0.25, 0.375, 0.5, 0, 0, 0, 0]
    #
    # rho = Phi @ nu = [2, -2, 1, 4]
    #
    # First term:
    # 1/2 * mean(rho^2)
    # = 1/2 * (4 + 4 + 1 + 16)/4
    # = 3.125
    #
    # nu^T m =
    # 1*0.125 + (-1)*0.25 + 0.5*0.375 + 2*0.5
    # = 1.0625
    #
    # (1-gamma) * nu^T m
    # = 0.5 * 1.0625
    # = 0.53125
    #
    # ratio loss = 3.125 - 0.53125 = 2.59375
    expected = 2.59375

    dice_params = {
        "sr_dice_successor": S,
        "sr_dice_nu": NU,
    }

    successor_start_mean = compute_successor_start_mean(
        dice_params,
        START_PHI,
    )

    expected_start_mean = jnp.array(
        [0.125, 0.25, 0.375, 0.5, 0., 0., 0., 0.],
        dtype=jnp.float32,
    )

    batch = SimpleNamespace(
        done=jnp.zeros((4,), dtype=jnp.float32),
        truncated=jnp.zeros((4,), dtype=jnp.float32),
        extras={"sr_dice_phi": PHI},
    )

    hparams = SimpleNamespace(
        mask_truncated=False,
        gamma=GAMMA,
    )

    rho, metrics = fit_sr_dice_ratio(
        dice_params,
        None,
        batch,
        hparams,
        None,
        successor_start_mean,
    )

    actual = metrics["sr_dice/ratio_loss"]

    check_scalar("SR-DICE RATIO LOSS", actual, expected)
    print(
        "successor_start_mean_matches_expected = "
        f"{bool(jnp.allclose(successor_start_mean, expected_start_mean))}"
    )
    print(f"rho          = {rho}")
    print(f"expected_rho = {jnp.array([2., -2., 1., 4.])}")
    print(
        "rho_matches_expected = "
        f"{bool(jnp.allclose(rho, jnp.array([2., -2., 1., 4.]))) }"
    )

# ===========================================================================
# Fixed saved-statistics BatchNorm fixture.
#
# This mirrors the training-time representation path:
# raw features -> BatchNorm in eval mode using SAVED running mean/variance
#              -> unchanged loss functions.
#
# No batch statistics are computed from PHI in this test.
# ===========================================================================

BN_EPS = 1e-5

BN_RUNNING_MEAN = jnp.zeros((8,), dtype=jnp.float32)

# Chosen so sqrt(var + eps) =
# [2, 1, 2/3, 1/2, 1, 1, 1, 1].
BN_RUNNING_VAR = jnp.array([
    4.0 - BN_EPS,
    1.0 - BN_EPS,
    (4.0 / 9.0) - BN_EPS,
    0.25 - BN_EPS,
    1.0 - BN_EPS,
    1.0 - BN_EPS,
    1.0 - BN_EPS,
    1.0 - BN_EPS,
], dtype=jnp.float32)

BN_SCALE = jnp.ones((8,), dtype=jnp.float32)
BN_BIAS = jnp.zeros((8,), dtype=jnp.float32)


def apply_saved_batch_norm(x):
    return (
        BN_SCALE
        * (x - BN_RUNNING_MEAN)
        / jnp.sqrt(BN_RUNNING_VAR + BN_EPS)
        + BN_BIAS
    )


BN_PHI_EXPECTED = jnp.array([
    [1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 2., 0., 0., 0., 0., 0., 0.],
    [0., 0., 3., 0., 0., 0., 0., 0.],
    [0., 0., 0., 4., 0., 0., 0., 0.],
], dtype=jnp.float32)

BN_NEXT_PHI_EXPECTED = jnp.array([
    [0.5, 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0., 0.],
    [0., 0., 1.5, 0., 0., 0., 0., 0.],
    [0., 0., 0., 2., 0., 0., 0., 0.],
], dtype=jnp.float32)

BN_PRED_PHI_EXPECTED = jnp.array([
    [1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 3., 0., 0., 0., 0., 0., 0.],
    [0., 0., 1.5, 0., 0., 0., 0., 0.],
    [0., 0., 0., -2., 0., 0., 0., 0.],
], dtype=jnp.float32)

BN_START_PHI_EXPECTED = jnp.array([
    [0.5, 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0., 0.],
    [0., 0., 1.5, 0., 0., 0., 0., 0.],
    [0., 0., 0., 2., 0., 0., 0., 0.],
], dtype=jnp.float32)


def get_bn_features():
    bn_phi = apply_saved_batch_norm(PHI)
    bn_next_phi = apply_saved_batch_norm(NEXT_PHI)
    bn_pred_phi = apply_saved_batch_norm(PRED_PHI)
    bn_start_phi = apply_saved_batch_norm(START_PHI)

    print("\nSAVED-STATISTICS BATCH NORM")
    print(f"BN_PHI match       = {bool(jnp.allclose(bn_phi, BN_PHI_EXPECTED, atol=1e-6))}")
    print(f"BN_NEXT_PHI match  = {bool(jnp.allclose(bn_next_phi, BN_NEXT_PHI_EXPECTED, atol=1e-6))}")
    print(f"BN_PRED_PHI match  = {bool(jnp.allclose(bn_pred_phi, BN_PRED_PHI_EXPECTED, atol=1e-6))}")
    print(f"BN_START_PHI match = {bool(jnp.allclose(bn_start_phi, BN_START_PHI_EXPECTED, atol=1e-6))}")

    return bn_phi, bn_next_phi, bn_pred_phi, bn_start_phi


def test_invariance_after_batch_norm():
    _, bn_next_phi, bn_pred_phi, _ = get_bn_features()

    # Residual active amplitudes = [0.5, 2, 0, -4].
    # Per-sample losses = [0.125, 2, 0, 8].
    # Uniform mean = 2.53125.
    expected = 2.53125

    actual, _ = invariance_aux_loss(
        bn_pred_phi,
        bn_next_phi,
        WEIGHTS,
    )

    check_scalar("BN -> INVARIANCE LOSS", actual, expected)


def test_orthonormality_after_batch_norm():
    bn_phi = apply_saved_batch_norm(PHI)

    # Gram = diag(0.25, 1, 2.25, 4, 0, 0, 0, 0).
    # ||Gram-I||_F^2 = 15.125.
    # Loss = 0.5 * 15.125 / 64 = 0.1181640625.
    expected = 0.1181640625

    expected_gram = jnp.diag(jnp.array(
        [0.25, 1.0, 2.25, 4.0, 0., 0., 0., 0.],
        dtype=jnp.float32,
    ))

    actual, gram, _ = batch_orthonormality_loss(
        bn_phi,
        WEIGHTS,
    )

    check_scalar("BN -> ORTHONORMALITY LOSS", actual, expected)
    print(f"bn_gram_matches_expected = {bool(jnp.allclose(gram, expected_gram, atol=1e-6))}")


def test_gershgorin_after_batch_norm():
    bn_phi = apply_saved_batch_norm(PHI)
    bn_next_phi = apply_saved_batch_norm(NEXT_PHI)

    # TD active amplitudes = [0.75, 1.5, 3, 3].
    # A = diag(0.1875, 0.75, 2.25, 3, 0, 0, 0, 0).
    # Only the last four zero rows violate by eps.
    # Loss = 4*1e-5 / 64 = 6.25e-7.
    expected = 6.25e-7

    expected_td_matrix = jnp.diag(jnp.array(
        [0.1875, 0.75, 2.25, 3.0, 0., 0., 0., 0.],
        dtype=jnp.float32,
    ))

    actual, td_matrix, _ = gershgorin_loss(
        bn_phi,
        bn_next_phi,
        WEIGHTS,
        CONTINUATION,
        gamma=GAMMA,
        eps=EPS,
    )

    check_scalar("BN -> GERSHGORIN LOSS", actual, expected)
    print(f"bn_td_matrix_matches_expected = {bool(jnp.allclose(td_matrix, expected_td_matrix, atol=1e-6))}")


def test_successor_after_batch_norm():
    bn_phi = apply_saved_batch_norm(PHI)
    bn_next_phi = apply_saved_batch_norm(NEXT_PHI)

    # psi active amplitudes = [0.5, 2, 4.5, 8].
    # next_psi amplitudes = [0.25, 1, 2.25, 4].
    # targets = [1.125, 2.5, 3, 6].
    # residuals = [-0.625, -0.5, 1.5, 2].
    # Per-sample losses = [0.1953125, 0.125, 1.125, 2].
    # Uniform mean = 0.861328125.
    expected = 0.861328125

    actual, _ = successor_feature_td_loss(
        bn_phi,
        bn_next_phi,
        WEIGHTS,
        CONTINUATION,
        S,
        gamma=GAMMA,
    )

    check_scalar("BN -> SUCCESSOR TD LOSS", actual, expected)


def test_ratio_after_batch_norm():
    bn_phi = apply_saved_batch_norm(PHI)
    bn_start_phi = apply_saved_batch_norm(START_PHI)

    # BN(start_phi) @ S active amplitudes = [0.25, 1, 2.25, 4].
    # successor_start_mean =
    # [0.0625, 0.25, 0.5625, 1, 0, 0, 0, 0].
    #
    # rho = BN(Phi) @ nu = [1, -2, 1.5, 8].
    #
    # 1/2 E[rho^2] = 8.90625.
    # (1-gamma) nu^T successor_start_mean = 1.046875.
    # Ratio loss = 7.859375.
    expected = 7.859375

    dice_params = {
        "sr_dice_successor": S,
        "sr_dice_nu": NU,
    }

    successor_start_mean = compute_successor_start_mean(
        dice_params,
        bn_start_phi,
    )

    expected_start_mean = jnp.array(
        [0.0625, 0.25, 0.5625, 1.0, 0., 0., 0., 0.],
        dtype=jnp.float32,
    )

    batch = SimpleNamespace(
        done=jnp.zeros((4,), dtype=jnp.float32),
        truncated=jnp.zeros((4,), dtype=jnp.float32),
        extras={"sr_dice_phi": bn_phi},
    )

    hparams = SimpleNamespace(
        mask_truncated=False,
        gamma=GAMMA,
    )

    rho, metrics = fit_sr_dice_ratio(
        dice_params,
        None,
        batch,
        hparams,
        None,
        successor_start_mean,
    )

    actual = metrics["sr_dice/ratio_loss"]

    check_scalar("BN -> SR-DICE RATIO LOSS", actual, expected)
    print(
        "bn_successor_start_mean_matches_expected = "
        f"{bool(jnp.allclose(successor_start_mean, expected_start_mean, atol=1e-6))}"
    )
    print(f"bn_rho          = {rho}")
    print(f"bn_expected_rho = {jnp.array([1., -2., 1.5, 8.])}")
    print(
        "bn_rho_matches_expected = "
        f"{bool(jnp.allclose(rho, jnp.array([1., -2., 1.5, 8.]), atol=1e-6))}"
    )

if __name__ == "__main__":
    print("===== RAW FEATURE LOSS TESTS =====")
    test_invariance()
    test_orthonormality()
    test_gershgorin()
    test_successor()
    test_ratio()

    print("\n===== SAVED-STATISTICS BATCH NORM + LOSS TESTS =====")
    test_invariance_after_batch_norm()
    test_orthonormality_after_batch_norm()
    test_gershgorin_after_batch_norm()
    test_successor_after_batch_norm()
    test_ratio_after_batch_norm()

    print("\nFinished fixed 4x8 raw and saved-statistics BatchNorm loss tests.")