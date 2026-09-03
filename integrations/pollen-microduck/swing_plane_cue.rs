//! Reference adapter for the released MicroDuck swing policy.
//!
//! The policy uses the first three command slots as an IMU-derived swing-plane cue:
//! `[0, body_y_in_start_frame.x, body_y_in_start_frame.z]`.  Capture `start_quat`
//! only after the IMU is ready and the swing is stationary.  Both quaternions use the
//! official runtime's scalar-first, trunk-to-world convention `[w, x, y, z]`.

#[derive(Debug, Clone, Copy)]
pub struct SwingPlaneCue {
    start_quat: [f64; 4],
}

impl SwingPlaneCue {
    /// Anchor the still-start frame. Returns `None` for a non-finite/degenerate sample.
    pub fn capture(start_quat: [f64; 4]) -> Option<Self> {
        normalise_quat(start_quat).map(|start_quat| Self { start_quat })
    }

    /// Build the exact three values expected in observation slots 48..51.
    ///
    /// Invalid live samples fail closed to a zero correction cue rather than injecting
    /// NaNs into the policy. The surrounding controller should still stop on loss of a
    /// trustworthy IMU stream; this fallback is not a substitute for that safety check.
    pub fn command_twist(&self, current_quat: [f64; 4]) -> [f64; 3] {
        let Some(current_quat) = normalise_quat(current_quat) else {
            return [0.0; 3];
        };
        let relative = mul(conjugate(self.start_quat), current_quat);
        let body_y_in_start_frame = rotate(relative, [0.0, 1.0, 0.0]);
        [
            0.0,
            finite_or_zero(body_y_in_start_frame[0]),
            finite_or_zero(body_y_in_start_frame[2]),
        ]
    }
}

fn finite_or_zero(value: f64) -> f64 {
    if value.is_finite() { value } else { 0.0 }
}

fn normalise_quat(q: [f64; 4]) -> Option<[f64; 4]> {
    let norm_sq = q.iter().map(|x| x * x).sum::<f64>();
    if !norm_sq.is_finite() || norm_sq < 1.0e-12 {
        return None;
    }
    let inverse_norm = norm_sq.sqrt().recip();
    Some(q.map(|x| x * inverse_norm))
}

fn conjugate(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

/// Hamilton product, scalar-first.
fn mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let ([aw, ax, ay, az], [bw, bx, by, bz]) = (a, b);
    [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]
}

/// Active rotation `q * v * q^-1`, matching `duck-control/src/imu.rs`.
fn rotate(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let qv = [q[1], q[2], q[3]];
    let t = cross(qv, v).map(|x| 2.0 * x);
    let c = cross(qv, t);
    [
        v[0] + q[0] * t[0] + c[0],
        v[1] + q[0] * t[1] + c[1],
        v[2] + q[0] * t[2] + c[2],
    ]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

    fn axis_angle(axis: [f64; 3], angle: f64) -> [f64; 4] {
        let s = (angle * 0.5).sin();
        [(angle * 0.5).cos(), axis[0] * s, axis[1] * s, axis[2] * s]
    }

    fn assert_near(actual: [f64; 3], expected: [f64; 3]) {
        for i in 0..3 {
            assert!((actual[i] - expected[i]).abs() < 1.0e-12, "{actual:?} != {expected:?}");
        }
    }

    #[test]
    fn still_start_is_zero() {
        let cue = SwingPlaneCue::capture([1.0, 0.0, 0.0, 0.0]).unwrap();
        assert_near(cue.command_twist([1.0, 0.0, 0.0, 0.0]), [0.0; 3]);
    }

    #[test]
    fn sagittal_pitch_is_unobserved() {
        let cue = SwingPlaneCue::capture([1.0, 0.0, 0.0, 0.0]).unwrap();
        assert_near(cue.command_twist(axis_angle([0.0, 1.0, 0.0], FRAC_PI_2)), [0.0; 3]);
    }

    #[test]
    fn yaw_and_roll_have_the_training_signs() {
        let cue = SwingPlaneCue::capture([1.0, 0.0, 0.0, 0.0]).unwrap();
        assert_near(cue.command_twist(axis_angle([0.0, 0.0, 1.0], FRAC_PI_2)), [0.0, -1.0, 0.0]);
        assert_near(cue.command_twist(axis_angle([1.0, 0.0, 0.0], FRAC_PI_2)), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn cue_is_invariant_to_a_common_world_frame_rotation() {
        let common = axis_angle([0.0, 0.0, 1.0], FRAC_PI_4);
        let motion = axis_angle([1.0, 0.0, 0.0], FRAC_PI_4);
        let cue = SwingPlaneCue::capture(common).unwrap();
        let actual = cue.command_twist(mul(common, motion));
        let reference = SwingPlaneCue::capture([1.0, 0.0, 0.0, 0.0])
            .unwrap()
            .command_twist(motion);
        assert_near(actual, reference);
    }

    #[test]
    fn invalid_samples_fail_closed() {
        assert!(SwingPlaneCue::capture([0.0; 4]).is_none());
        let cue = SwingPlaneCue::capture([1.0, 0.0, 0.0, 0.0]).unwrap();
        assert_near(cue.command_twist([f64::NAN, 0.0, 0.0, 0.0]), [0.0; 3]);
    }
}
