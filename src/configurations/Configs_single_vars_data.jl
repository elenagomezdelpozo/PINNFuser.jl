module ConfigsMod

export configs

configs = [
    # loss 1 data only
    (
        name            = "data",
        active         =["data"],
        early_stopping_start = 40,
        training_vars = [1,2,3,4,5,6]

    ),
    # loss 2 data and physics
    (
        name            = "data_phy",
        active         =["data", "physics"],
        early_stopping_start = 40,
        training_vars = [1,2,3,4,5,6]

    ),
    # loss 3 data and mass conservation
    (
        name            = "data_mass",
        active         =["data", "mass"],
        early_stopping_start = 40,
        training_vars = [1,2,3,4,5,6]
    ),
    # loss 4 data and negativity
    (
        name            = "data_neg",
        active         =["data", "negativity"],
        early_stopping_start = 40,
        training_vars = [1,2,3,4,5,6]

    ),
    # loss 5 data and first derivative
    (
        name            = "data_deriv",
        active         =["data", "firstderiv"],
        early_stopping_start = 40,
        training_vars = [1,2,3,4,5,6]

        ),
    # loss 6 data and periodicity
    (
        name            = "data_period",
        active         =["data", "periodicity"],
        early_stopping_start = 40,
        training_vars = [1,2,3,4,5,6]

        ),
    # loss 7 data and zero mean
    (
        name            = "data_zm",
        active         =["data", "zero_mean"],
        early_stopping_start = 40,
        training_vars = [1,2,3,4,5,6]

        ),
]
end