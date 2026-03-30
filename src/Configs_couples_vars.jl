module ConfigsMod

export configs

configs = [

    (
        name            = "deriv",
        active         =["firstderiv"],
        early_stopping_start = 40,
        training_vars = [1,2,3,4,5,6]
    ),

    (
        name            = "deriv_mass",
        active         =["firstderiv", "mass"],
        early_stopping_start = 40,
        training_vars = [1,2,3,4,5,6]

    ),

    # ── Example 3: all-vars training, higher lr ───────────────
    (
        name            = "deriv_neg",
        active         =["firstderiv", "negativity"],
        early_stopping_start = 40,
        training_vars = [1,2,3,4,5,6]
    ),

    # ── Example 4: all-vars training, higher lr ───────────────
    (
        name            = "deriv_period",
        active         =["periodicity", "firstderiv"],
        early_stopping_start = 40,
        training_vars = [1,2,3,4,5,6]

    ),
    # ── Example 5: all-vars training, higher lr ───────────────
    (
        name            = "deriv_zm",
        active         =["firstderiv", "zero_mean"],
        early_stopping_start = 40,
        training_vars = [1,2,3,4,5,6]
    )
]
end