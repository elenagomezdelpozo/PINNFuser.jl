module ConfigsMod

export configs

configs = [
    # loss 1 first derivative only
    (
        name            = "deriv",
        active         =["firstderiv"],
        early_stopping_start = 100,
        training_vars = [4,5,6]
    ),
    # loss 2 mass conservation
    (
        name            = "deriv_mass",
        active         =["firstderiv", "mass"],
        early_stopping_start = 100,
        training_vars = [4,5,6]
    ),
    # loss 3 negativity
    (
        name            = "deriv_neg",
        active         =["firstderiv", "negativity"],
        early_stopping_start = 100,
        training_vars = [4,5,6]
    ),
    # loss 4 periodicity
    (
        name            = "deriv_period",
        active         =["periodicity", "firstderiv"],
        early_stopping_start = 100,
        training_vars = [4,5,6]

    ),
    # loss 5 zero mean
    (
        name            = "deriv_zero_mean",
        active         =["firstderiv", "zero_mean"],
        early_stopping_start = 100,
        training_vars = [4,5,6]
    ),
    # loss 6 physics
    (
        name            = "deriv_phy",
        active         =["firstderiv", "physics"],
        early_stopping_start = 100,
        training_vars = [4,5,6]
    )
]
end