use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum Decision {
    Allow,
    Halt,
}

#[derive(Clone, Copy)]
pub struct Policy {
    pub threshold: f64,
}

impl Policy {
    pub fn new(threshold: f64) -> Result<Self, &'static str> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err("threshold must be a finite number between 0 and 1");
        }
        Ok(Self { threshold })
    }

    pub fn decide(self, risk: f64) -> Result<Decision, &'static str> {
        if !(0.0..=1.0).contains(&risk) {
            return Err("risk must be a finite number between 0 and 1");
        }
        Ok(if risk >= self.threshold {
            Decision::Halt
        } else {
            Decision::Allow
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_threshold_halts_even_at_endpoints() {
        for threshold in [0.0, 0.5, 1.0] {
            assert_eq!(
                Policy::new(threshold).unwrap().decide(threshold),
                Ok(Decision::Halt)
            );
        }
        assert_eq!(Policy::new(0.5).unwrap().decide(0.49), Ok(Decision::Allow));
    }

    #[test]
    fn invalid_scores_and_thresholds_fail_closed() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.01, 1.01] {
            assert!(Policy::new(invalid).is_err());
            assert!(Policy::new(0.5).unwrap().decide(invalid).is_err());
        }
    }
}
