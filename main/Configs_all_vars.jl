module ConfigsMod

export configs

configs = [

    # ── Example 1: single-var training, only data loss ──────
    (
        name            = "data",
        active         =["data"],
    ),

    # ── Example 2: single-var, data + physics losses ─────────
    (
        name            = "data_phy",
        active         =["data", "physics"],
    ),

    # ── Example 3: all-vars training, higher lr ───────────────
    (
        name            = "data_mass",
        active         =["data", "mass"],
    ),

    # ── Example 3: all-vars training, higher lr ───────────────
    (
        name            = "data_deriv",
        active         =["data", "firstderiv"],
    ),
    # ── Example 3: all-vars training, higher lr ───────────────
    (
        name            = "data_neg",
        active         =["data", "negativity"],
    ),
    # ── Example 3: all-vars training, higher lr ───────────────
    (
        name            = "data_period",
        active         =["data", "periodicity"],
    ),
    # ── Example 3: all-vars training, higher lr ───────────────
    (
        name            = "data_zm",
        active         =["data", "zero_mean"],
    ),
]
end