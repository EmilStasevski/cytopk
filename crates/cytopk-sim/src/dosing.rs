//! Dose records and validated dosing schedules.
//!
//! Conventions (see `docs/models.md`):
//! - A bolus adds `amount` to the target state instantaneously at `time`.
//! - An infusion adds `amount / duration` per unit time to the target state on `[time, time + duration)`.
//! - The output at time `t` includes every bolus given at exactly `t`.

use crate::SimError;

/// One dose: an IV bolus (`duration == None`) or a zero-order infusion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dose {
    /// Start time (days).
    pub time: f64,
    /// Amount (mg).
    pub amount: f64,
    /// Index of the state (compartment) that receives the dose. `0` is the central compartment.
    pub state: usize,
    /// Infusion duration (days); `None` means bolus.
    pub duration: Option<f64>,
}

impl Dose {
    /// A bolus of `amount` into `state` at `time`.
    pub fn bolus(time: f64, amount: f64, state: usize) -> Self {
        Self {
            time,
            amount,
            state,
            duration: None,
        }
    }

    /// A zero-order infusion of `amount` into `state`, starting at `time` and lasting `duration`.
    pub fn infusion(time: f64, amount: f64, duration: f64, state: usize) -> Self {
        Self {
            time,
            amount,
            state,
            duration: Some(duration),
        }
    }

    /// Infusion rate (amount per day), or `None` for a bolus.
    pub fn rate(&self) -> Option<f64> {
        self.duration.map(|d| self.amount / d)
    }

    /// Time at which the dose has been fully delivered (`time` for a bolus).
    pub fn end_time(&self) -> f64 {
        self.time + self.duration.unwrap_or(0.0)
    }

    fn validate(&self, index: usize) -> Result<(), SimError> {
        let bad = |reason| Err(SimError::InvalidDose { index, reason });
        if !self.time.is_finite() || self.time < 0.0 {
            return bad("time must be finite and >= 0");
        }
        if !self.amount.is_finite() || self.amount < 0.0 {
            return bad("amount must be finite and >= 0");
        }
        if let Some(d) = self.duration {
            if !d.is_finite() || d <= 0.0 {
                return bad("infusion duration must be finite and > 0");
            }
            if !self.end_time().is_finite() {
                return bad("infusion end time overflows");
            }
        }
        Ok(())
    }
}

/// A validated list of doses, sorted by start time (stable, so equal-time doses keep their order).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Schedule {
    doses: Vec<Dose>,
}

impl Schedule {
    /// Validates and sorts `doses`. The input does not need to be sorted.
    /// Error indices refer to positions in the input.
    pub fn new(mut doses: Vec<Dose>) -> Result<Self, SimError> {
        for (i, d) in doses.iter().enumerate() {
            d.validate(i)?;
        }
        doses.sort_by(|a, b| a.time.total_cmp(&b.time));
        Ok(Self { doses })
    }

    /// A schedule without doses.
    pub fn empty() -> Self {
        Self::default()
    }

    /// `n` copies of `dose`, every `interval` days starting at `dose.time`.
    pub fn repeated(dose: Dose, interval: f64, n: usize) -> Result<Self, SimError> {
        if !interval.is_finite() || interval < 0.0 {
            return Err(SimError::InvalidOptions(
                "dosing interval must be finite and >= 0",
            ));
        }
        let doses = (0..n)
            .map(|i| Dose {
                time: dose.time + interval * i as f64,
                ..dose
            })
            .collect();
        Self::new(doses)
    }

    /// The doses, sorted by start time.
    pub fn doses(&self) -> &[Dose] {
        &self.doses
    }

    /// Total amount over all doses.
    pub fn total_amount(&self) -> f64 {
        self.doses.iter().map(|d| d.amount).sum()
    }

    /// Checks that every dose targets an existing state.
    pub(crate) fn check_states(
        &self,
        model: &'static str,
        n_states: usize,
    ) -> Result<(), SimError> {
        match self.doses.iter().find(|d| d.state >= n_states) {
            Some(d) => Err(SimError::DoseStateOutOfRange {
                model,
                state: d.state,
                n_states,
            }),
            None => Ok(()),
        }
    }

    /// Writes the total infusion rate per state that applies on `[t, next event)` into `rates`.
    /// Recomputed from scratch at each event so that no rounding error builds up.
    pub(crate) fn infusion_rates_at(&self, t: f64, rates: &mut [f64]) {
        rates.fill(0.0);
        for d in &self.doses {
            if let Some(r) = d.rate()
                && d.time <= t
                && t < d.end_time()
            {
                rates[d.state] += r;
            }
        }
    }

    /// All times at which the right-hand side or the state jumps (dose starts, infusion ends).
    pub(crate) fn event_times(&self) -> Vec<f64> {
        let mut ts: Vec<f64> = self
            .doses
            .iter()
            .flat_map(|d| [Some(d.time), d.duration.map(|_| d.end_time())])
            .flatten()
            .collect();
        ts.sort_by(f64::total_cmp);
        ts.dedup();
        ts
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn constructors_and_rate() {
        let b = Dose::bolus(1.0, 100.0, 0);
        assert_eq!(b.rate(), None);
        assert_eq!(b.end_time(), 1.0);
        let i = Dose::infusion(2.0, 100.0, 0.5, 0);
        assert_eq!(i.rate(), Some(200.0));
        assert_eq!(i.end_time(), 2.5);
    }

    #[test]
    fn new_sorts_and_validates() {
        let s = Schedule::new(vec![Dose::bolus(7.0, 1.0, 0), Dose::bolus(0.0, 2.0, 0)]).unwrap();
        assert_eq!(s.doses()[0].time, 0.0);
        assert_eq!(s.total_amount(), 3.0);
        for (bad, idx) in [
            (Dose::bolus(-1.0, 1.0, 0), 1),
            (Dose::bolus(f64::NAN, 1.0, 0), 1),
            (Dose::bolus(0.0, -1.0, 0), 1),
            (Dose::bolus(0.0, f64::INFINITY, 0), 1),
            (Dose::infusion(0.0, 1.0, 0.0, 0), 1),
            (Dose::infusion(0.0, 1.0, f64::NAN, 0), 1),
        ] {
            let err = Schedule::new(vec![Dose::bolus(0.0, 1.0, 0), bad]).unwrap_err();
            assert!(matches!(err, SimError::InvalidDose { index, .. } if index == idx));
        }
    }

    #[test]
    fn repeated_and_empty() {
        let s = Schedule::repeated(Dose::infusion(0.0, 10.0, 0.1, 0), 7.0, 4).unwrap();
        let times: Vec<f64> = s.doses().iter().map(|d| d.time).collect();
        assert_eq!(times, vec![0.0, 7.0, 14.0, 21.0]);
        assert!(Schedule::repeated(Dose::bolus(0.0, 1.0, 0), -1.0, 2).is_err());
        assert!(Schedule::empty().doses().is_empty());
    }

    #[test]
    fn rates_and_events() {
        let s = Schedule::new(vec![
            Dose::infusion(0.0, 10.0, 1.0, 0),
            Dose::infusion(0.5, 4.0, 2.0, 0),
            Dose::bolus(0.5, 1.0, 1),
        ])
        .unwrap();
        let mut r = [0.0; 2];
        s.infusion_rates_at(0.5, &mut r);
        assert_eq!(r, [12.0, 0.0]);
        s.infusion_rates_at(1.0, &mut r); // first infusion ended (half-open interval)
        assert_eq!(r, [2.0, 0.0]);
        s.infusion_rates_at(2.5, &mut r);
        assert_eq!(r, [0.0, 0.0]);
        assert_eq!(s.event_times(), vec![0.0, 0.5, 1.0, 2.5]);
        assert!(s.check_states("m", 2).is_ok());
        assert!(s.check_states("m", 1).is_err());
    }
}
