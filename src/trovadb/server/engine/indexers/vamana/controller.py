class AlphaController:
    def __init__(
        self,
        target_degree: float,
        kp: float,
        ki: float,
        alpha_init: float,
        alpha_min: float = 1.0,
        alpha_max: float = 2.0,
        deadband: float = 0.2,
    ):
        self.target = target_degree
        self.kp = kp
        self.ki = ki
        self.alpha_init = alpha_init
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.deadband = deadband

        if not (alpha_min <= alpha_init <= alpha_max):
            raise ValueError(f"alpha_init must be in [{alpha_min}, {alpha_max}]")

        self.alpha: float = alpha_init
        self.integral: float = 0.0

    def feedback(self, avg_degree: float) -> None:
        error = self.target - avg_degree

        if abs(error) < self.deadband:
            error = 0.0

        integral = self.integral + error
        alpha = self.alpha_init + self.kp * error + self.ki * integral

        if alpha > self.alpha_max:
            self.alpha = self.alpha_max
            if error < 0:
                self.integral = integral
        elif alpha < self.alpha_min:
            self.alpha = self.alpha_min
            if error > 0:
                self.integral = integral
        else:
            self.integral = integral
            self.alpha = alpha

    def get_alpha(self) -> float:
        return self.alpha

    def reset(self):
        self.alpha = self.alpha_init
        self.integral = 0.0
