module ConfigsMod

export configs

configs = [

    # ── Example 1: single-var training, only data loss ──────
    (
        name            = "data",
        active         =["data"],
        early_stopping_start = 40,
        training_vars = [1,2,3,4,5,6]
    ),

    # ── Example 2: single-var, data + physics losses ─────────
    (
        name            = "data_phy",
        active         =["data", "physics"],
        early_stopping_start = 40,
        training_vars = [5,6]

    ),

    # ── Example 3: all-vars training, higher lr ───────────────
    (
        name            = "data_mass",
        active         =["data", "mass"],
        early_stopping_start = 40,
        training_vars = [5,6]
    ),

    # ── Example 4: all-vars training, higher lr ───────────────
    (
        name            = "data_deriv",
        active         =["data", "firstderiv"],
        early_stopping_start = 40,
        training_vars = [5,6]

    ),
    # ── Example 5: all-vars training, higher lr ───────────────
    (
        name            = "data_neg",
        active         =["data", "negativity"],
        early_stopping_start = 40,
        training_vars = [6]
    ),
    # ── Example 6: all-vars training, higher lr ───────────────
    (
        name            = "data_period",
        active         =["data", "periodicity"],
        early_stopping_start = 40,
        training_vars = [1,2,3,4,5,6]
    ),
    # ── Example 7: all-vars training, higher lr ───────────────
    (
        name            = "data_zm",
        active         =["data", "zero_mean"],
        early_stopping_start = 40,
        training_vars = [1,2,3,4,5,6]
    ),
]
end