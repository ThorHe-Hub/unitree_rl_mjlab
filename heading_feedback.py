"""Pure Python heading feedback selection and the existing PID control law.

Call command() before advance(): both modes use the previous sample's estimate.
The independent integral is retained even when PID uses fused heading.
"""
import math


class HeadingFeedback:
    def __init__(self, mode="integrated", filt="dirty", offset_tau=0.1):
        if mode not in ("integrated", "fusion_heading", "fusion_offset_smoothed"):
            raise ValueError(f"Unknown feedback mode: {mode}")
        if mode != "integrated" and filt != "fusion":
            raise ValueError("fusion_heading requires --rig.filt fusion")
        if not (math.isfinite(offset_tau) and offset_tau > 0):
            raise ValueError("offset_tau must be positive and finite")
        self.mode = mode
        # Smoothing time constant of f-i for fusion_offset_smoothed. The default 0.1 s
        # is the value prespecified for the original smoothing batch; keep it unchanged.
        self.offset_tau = offset_tau
        self.reset()

    def reset(self):
        self.integrated_yaw = 0.0
        self.value = 0.0
        self.smoothed_offset = 0.0

    def advance(self, used_yaw, dt, fusion_heading=None):
        if not math.isfinite(used_yaw) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Angular rate and positive dt must be finite")
        if self.mode != "integrated" and (
                fusion_heading is None or not math.isfinite(fusion_heading)):
            raise ValueError("Fusion heading is unavailable or non-finite")
        self.integrated_yaw += used_yaw * dt
        if self.mode == "fusion_offset_smoothed":
            # Smooth only f-i, preserving gyro increments.
            a = math.exp(-dt / self.offset_tau)
            self.smoothed_offset = a * self.smoothed_offset + (1-a) * (float(fusion_heading)-self.integrated_yaw)
            self.value = self.integrated_yaw + self.smoothed_offset
        else:
            self.value = self.integrated_yaw if self.mode == "integrated" else float(fusion_heading)

    def command(self, controller, turning, dt):
        """Use the legacy gains, saturation and target-lock behavior in both modes.

        Heading states are continuous (unwrapped). Taking a raw difference keeps
        the same control law across feedback modes, including multiple turns.
        The sensor's wrapped innovation handles the +/-pi measurement boundary.
        """
        if turning:
            controller.yaw_target = self.value
            controller.is_turning = True
            controller.error_integral = 0.0
            controller.last_error = 0.0
            return 0.0
        if controller.is_turning:
            controller.yaw_target = self.value
            controller.is_turning = False
        error = controller.yaw_target - self.value
        derivative = (error - controller.last_error) / dt
        controller.error_integral = max(-0.15, min(0.15, controller.error_integral + error * dt))
        correction = (controller.Kp * error + controller.Ki * controller.error_integral
                      + controller.Kd * derivative)
        controller.last_error = error
        return max(-1.6, min(1.6, correction))
