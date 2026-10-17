from gui.gui import StimulusApp


def main(
    *,
    auto_random_trials: bool = False,
    inter_trial_s: float = 1.0,
    auto_plan_trials: int = 15,
    auto_plan_target_tile_id: int = 4,
    auto_plan_target_repeats: int = 0,
    auto_plan_target_epochs: int = 12,
    sequences_override: int | None = None,
    auto_max_trials: int | None = None,
    stim_control_dir: str | None = None,
) -> None:
    app = StimulusApp(
        auto_random_trials=auto_random_trials,
        inter_trial_s=inter_trial_s,
        auto_plan_trials=auto_plan_trials,
        auto_plan_target_tile_id=auto_plan_target_tile_id,
        auto_plan_target_repeats=auto_plan_target_repeats,
        auto_plan_target_epochs=auto_plan_target_epochs,
        sequences_override=sequences_override,
        auto_max_trials=auto_max_trials,
        stim_control_dir=stim_control_dir,
    )
    app.run()


if __name__ == "__main__":
    main()
