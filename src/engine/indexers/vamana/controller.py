class AlphaController:
    def __init__(
        self,
        target_degree: float,
        kp: float,
        ki: float,
        alpha_init: float,
        alpha_min: float = 1.0,
        alpha_max: float = 2.0,
    ):
        self.target = target_degree
        self.kp = kp
        self.ki = ki
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        if not (alpha_min <= alpha_init <= alpha_max):
            raise ValueError(f"alpha_init must be in [{alpha_min}, {alpha_max}]")

        self.alpha: float = alpha_init
        self.prev_error: float = 0.0

    def feedback(self, avg_degree: float) -> None:
        error = self.target - avg_degree

        saturating_high = self.alpha >= self.alpha_max and error > 0
        saturating_low = self.alpha <= self.alpha_min and error < 0

        if not (saturating_high or saturating_low):
            delta = self.kp * (error - self.prev_error) + self.ki * error
            self.alpha += delta
            self.alpha = max(self.alpha_min, min(self.alpha_max, self.alpha))

        self.prev_error = error

    def get_alpha(self) -> float:
        return self.alpha
