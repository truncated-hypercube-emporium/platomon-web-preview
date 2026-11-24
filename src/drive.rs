//! Per-solid inputs: which metric drives each solid's size, spin, colour
//! and complexity (its shape), as the metric's value or as how much it
//! changed over a period (delta).
//!
//! A solid is described by a [`Binding`], written `SHAPE|SIZE|SPIN|COLOUR|COMPLEXITY`:
//! `SHAPE` is a solid id or `auto`, each input a metric (`cpu`, `nic:Ethernet`,
//! ...), `delta:METRIC` for its change, or `off`. [`Drives`] turns the
//! metrics' channels into each solid's channel every frame.

use std::collections::VecDeque;

use crate::metrics::{Channel, Ema, MetricSpec, MetricsState, Reading, RollingPeak};
use crate::scene::sdf::SolidKind;

/// Most solids the Metrics dialog sets up.
pub const MAX_SOLIDS: usize = 12;
/// Delta period, seconds.
pub const DELTA_PERIOD_RANGE: (f32, f32) = (1.0, 600.0);
pub const DELTA_PERIOD_DEFAULT: f32 = 10.0;
const DELTA_PREFIX: &str = "delta:";
/// How often a delta input remembers its value, seconds.
const SAMPLE_EVERY: f32 = 0.1;
/// A change this small (of the 0..=1 value) never reads as full scale.
const DELTA_FLOOR: f64 = 0.05;
/// The delta's rolling peak fades by this fraction a second.
const DELTA_DECAY: f64 = 0.02;
/// Delta outputs settle with this time constant, seconds.
const DELTA_TAU: f32 = 0.35;
/// A complexity-driven solid changes shape only once its input is this far
/// (in steps) past the middle between two shapes.
const SHAPE_HYSTERESIS: f32 = 0.35;

/// What an input drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Drive {
    Size,
    Spin,
    Color,
    Complexity,
}

impl Drive {
    pub const ALL: [Drive; 4] = [Drive::Size, Drive::Spin, Drive::Color, Drive::Complexity];

    pub fn id(self) -> &'static str {
        match self {
            Drive::Size => "size",
            Drive::Spin => "spin",
            Drive::Color => "colour",
            Drive::Complexity => "complexity",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Drive::Size => "Size",
            Drive::Spin => "Spin",
            Drive::Color => "Colour",
            Drive::Complexity => "Complexity",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// One input: a metric (or none), absolute or its change.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Input {
    pub metric: Option<MetricSpec>,
    /// The change over the delta period instead of the value.
    pub delta: bool,
}

impl Input {
    pub const OFF: Input = Input { metric: None, delta: false };

    pub fn abs(m: MetricSpec) -> Self {
        Self { metric: Some(m), delta: false }
    }

    /// `off`, `METRIC` or `delta:METRIC`.
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();
        if s.is_empty() || ["off", "none"].iter().any(|o| s.eq_ignore_ascii_case(o)) {
            return Ok(Self::OFF);
        }
        let (delta, m) = match s.get(..DELTA_PREFIX.len()) {
            Some(p) if p.eq_ignore_ascii_case(DELTA_PREFIX) => (true, &s[DELTA_PREFIX.len()..]),
            _ => (false, s),
        };
        Ok(Self { metric: Some(MetricSpec::parse(m)?), delta })
    }

    pub fn text(&self) -> String {
        match (&self.metric, self.delta) {
            (None, _) => "off".into(),
            (Some(m), false) => m.key(),
            (Some(m), true) => format!("{DELTA_PREFIX}{}", m.key()),
        }
    }
}

/// One solid: its shape and its four inputs (in [`Drive::ALL`] order).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    /// `None` = automatic.
    pub solid: Option<SolidKind>,
    pub inputs: [Input; 4],
}

impl Binding {
    /// The classic solid: `metric` drives size, spin and colour.
    pub fn of(metric: MetricSpec) -> Self {
        let i = Input::abs(metric);
        Self { solid: None, inputs: [i.clone(), i.clone(), i, Input::OFF] }
    }

    /// `SHAPE|SIZE|SPIN|COLOUR|COMPLEXITY`; missing spin and colour follow
    /// size, missing complexity is off.
    pub fn parse(s: &str) -> Result<Self, String> {
        let f: Vec<&str> = s.split('|').map(str::trim).collect();
        if f.len() < 2 || f.len() > 5 {
            return Err(format!("expected SHAPE|SIZE|SPIN|COLOUR|COMPLEXITY, got '{s}'"));
        }
        let solid = match f[0] {
            "" => None,
            v if v.eq_ignore_ascii_case("auto") => None,
            v => Some(SolidKind::from_id(v).ok_or_else(|| format!("unknown solid '{v}' (see --list solids)"))?),
        };
        let size = Input::parse(f[1])?;
        let at = |k: usize, d: &Input| f.get(k).map_or(Ok(d.clone()), |v| Input::parse(v));
        let inputs = [size.clone(), at(2, &size)?, at(3, &size)?, at(4, &Input::OFF)?];
        if inputs.iter().all(|i| i.metric.is_none()) {
            return Err(format!("'{s}': a solid needs at least one input"));
        }
        Ok(Self { solid, inputs })
    }

    pub fn text(&self) -> String {
        let shape = self.solid.map_or("auto", |k| k.id());
        let inputs: Vec<String> = self.inputs.iter().map(Input::text).collect();
        format!("{shape}|{}", inputs.join("|"))
    }

    pub fn input(&self, d: Drive) -> &Input {
        &self.inputs[d.index()]
    }

    /// The metric the solid is known by: its size input's, else the first set.
    pub fn primary(&self) -> Option<&MetricSpec> {
        self.inputs.iter().find_map(|i| i.metric.as_ref())
    }
}

/// The shapes from simplest to most complex.
pub fn ladder() -> Vec<SolidKind> {
    let mut l = SolidKind::ALL.to_vec();
    l.sort_by(|a, b| a.complexity().total_cmp(&b.complexity()));
    l
}

/// A delta input's memory: its value over the last period.
#[derive(Debug, Clone)]
struct Delta {
    hist: VecDeque<(f32, [f32; 3])>,
    next: f32,
    peak: RollingPeak,
    out: [Ema; 3],
}

impl Delta {
    fn new() -> Self {
        Self { hist: VecDeque::new(), next: 0.0, peak: RollingPeak::new(DELTA_FLOOR, DELTA_DECAY), out: [Ema::new(DELTA_TAU); 3] }
    }

    /// How far `now` is from the value `period` seconds ago (the oldest
    /// remembered until then), each 0..=1 of the largest change lately.
    fn step(&mut self, t: f32, now: [f32; 3], period: f32, dt: f32) -> [f32; 3] {
        if t >= self.next {
            self.hist.push_back((t, now));
            self.next = t + SAMPLE_EVERY;
        }
        while self.hist.len() > 1 && self.hist[1].0 <= t - period {
            self.hist.pop_front();
        }
        let then = self.hist.front().map_or(now, |h| h.1);
        let d = [0, 1, 2].map(|k| (now[k] - then[k]).abs());
        let p = self.peak.update(d.iter().copied().fold(0.0, f32::max) as f64, dt as f64) as f32;
        [0, 1, 2].map(|k| self.out[k].update((d[k] / p).clamp(0.0, 1.0), dt).clamp(0.0, 1.0))
    }
}

#[derive(Debug, Clone)]
struct Link {
    /// Index into the sources.
    src: Option<usize>,
    delta: Option<Delta>,
}

/// Every solid's inputs, fed from one [`MetricsState`] over the distinct
/// metrics they use ([`Drives::sources`]).
pub struct Drives {
    sources: Vec<MetricSpec>,
    links: Vec<[Link; 4]>,
    /// Complexity: the shape each solid starts from (its index in the
    /// ladder) and the step it is on now.
    base: Vec<usize>,
    step: Vec<usize>,
    ladder: Vec<SolidKind>,
    period: f32,
    t: f32,
    channels: Vec<Channel>,
    color: Vec<Channel>,
}

impl Drives {
    /// `solids`: each solid's shape at rest and its inputs.
    pub fn new(solids: &[(SolidKind, [Input; 4])], period: f32) -> Self {
        let mut sources: Vec<MetricSpec> = Vec::new();
        // primary metrics first, so the stats bar follows the solids
        let order = solids.iter().flat_map(|(_, ins)| ins.iter()).filter_map(|i| i.metric.as_ref());
        for m in order {
            if !sources.contains(m) {
                sources.push(m.clone());
            }
        }
        let links = solids
            .iter()
            .map(|(_, ins)| {
                ins.clone().map(|i| Link {
                    src: i.metric.as_ref().and_then(|m| sources.iter().position(|s| s == m)),
                    delta: (i.delta && i.metric.is_some()).then(Delta::new),
                })
            })
            .collect();
        let ladder = ladder();
        let base: Vec<usize> = solids.iter().map(|(k, _)| ladder.iter().position(|l| l == k).unwrap_or(0)).collect();
        let n = solids.len();
        let period = if period.is_finite() { period.clamp(DELTA_PERIOD_RANGE.0, DELTA_PERIOD_RANGE.1) } else { DELTA_PERIOD_DEFAULT };
        Self {
            sources,
            links,
            step: vec![0; n],
            base,
            ladder,
            period,
            t: 0.0,
            channels: vec![Channel::default(); n],
            color: vec![Channel::default(); n],
        }
    }

    /// The distinct metrics the solids use (what the [`MetricsState`] reads).
    pub fn sources(&self) -> &[MetricSpec] {
        &self.sources
    }

    pub fn len(&self) -> usize {
        self.links.len()
    }

    pub fn is_empty(&self) -> bool {
        self.links.is_empty()
    }

    /// Advance by `dt` seconds with the metrics' smoothed channels.
    pub fn tick(&mut self, m: &MetricsState, dt: f32) {
        self.t += dt.max(0.0);
        let (t, period) = (self.t, self.period);
        for (i, links) in self.links.iter_mut().enumerate() {
            // each input's (available, [level, spin y, spin x], activity)
            let mut vals = [(false, [0.0f32; 3], 0.0f32); 4];
            for (v, l) in vals.iter_mut().zip(links.iter_mut()) {
                let Some(src) = l.src.filter(|&s| s < m.len()) else { continue };
                let c = m.channel(src);
                let now = [c.level, c.spin_y, c.spin_x];
                *v = match &mut l.delta {
                    Some(d) if c.available => {
                        let x = d.step(t, now, period, dt);
                        (true, x, x[0])
                    }
                    _ => (c.available, now, c.activity),
                };
            }
            let [size, spin, color, cplx] = vals;
            let on = |d: Drive, v: (bool, [f32; 3], f32)| links[d.index()].src.is_some() && v.0;
            self.channels[i] = Channel {
                available: vals.iter().zip(Drive::ALL).any(|(&v, d)| on(d, v)),
                level: if on(Drive::Size, size) { size.1[0] } else { 0.0 },
                spin_y: if on(Drive::Spin, spin) { spin.1[1] } else { 0.0 },
                spin_x: if on(Drive::Spin, spin) { spin.1[2] } else { 0.0 },
                activity: if on(Drive::Size, size) { size.2 } else { 0.0 },
            };
            self.color[i] =
                Channel { available: on(Drive::Color, color), level: if on(Drive::Color, color) { color.1[0] } else { 0.0 }, ..Channel::default() };
            if on(Drive::Complexity, cplx) {
                let steps = self.ladder.len() - 1 - self.base[i];
                let target = cplx.1[0].clamp(0.0, 1.0) * steps as f32;
                let cur = self.step[i] as f32;
                if (target - cur).abs() > 0.5 + SHAPE_HYSTERESIS {
                    self.step[i] = (target.round() as usize).min(steps);
                }
            }
        }
    }

    /// Each solid's channel: size and activity from its size input, spin
    /// from its spin input.
    pub fn channels(&self) -> &[Channel] {
        &self.channels
    }

    /// Each solid's colour input, as a channel (its level is the value).
    pub fn color_channels(&self) -> &[Channel] {
        &self.color
    }

    /// Each solid's shape now.
    pub fn shapes(&self) -> impl Iterator<Item = SolidKind> + '_ {
        self.base.iter().zip(&self.step).map(|(&b, &s)| self.ladder[(b + s).min(self.ladder.len() - 1)])
    }

    /// Each solid's reading for its tag: its primary metric's.
    pub fn readings<'a>(&self, m: &'a MetricsState) -> Vec<Option<&'a Reading>> {
        let all = m.readings();
        self.links.iter().map(|l| l.iter().find_map(|x| x.src).and_then(|s| all.get(s).copied())).collect()
    }

    /// Each solid's activity weight: its primary metric's kind's.
    pub fn weights(&self, w: &crate::config::Weights) -> Vec<f32> {
        self.links
            .iter()
            .map(|l| l.iter().find_map(|x| x.src).map_or(0.0, |s| w.get(self.sources[s].kind)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collector::RawMetrics;
    use crate::config::Config;

    fn spec(s: &str) -> MetricSpec {
        MetricSpec::parse(s).unwrap()
    }

    #[test]
    fn bindings_round_trip_and_default_sensibly() {
        let b = Binding::parse("auto|cpu").unwrap();
        assert_eq!(b, Binding::of(spec("cpu")));
        assert_eq!(b.text(), "auto|cpu|cpu|cpu|off");
        let b = Binding::parse("cube|off|delta:ram|nic:Ethernet 2|delta:disk-dev:C:\\").unwrap();
        assert_eq!(b.solid, Some(SolidKind::Cube));
        assert_eq!(b.inputs[0], Input::OFF);
        assert_eq!(b.inputs[1], Input { metric: Some(spec("ram")), delta: true });
        assert_eq!(b.primary(), Some(&spec("ram")));
        assert_eq!(Binding::parse(&b.text()).unwrap(), b);
        assert!(Binding::parse("auto|off|off|off|off").is_err(), "no input");
        assert!(Binding::parse("cpu").is_err(), "no shape field");
        assert!(Binding::parse("blob|cpu").is_err());
        assert!(Binding::parse("auto|cpu-temp").is_err());
        assert_eq!(Input::parse("DELTA:gpu").unwrap().text(), "delta:gpu");
    }

    fn raw(seq: u64, cpu: f32) -> RawMetrics {
        RawMetrics { seq, elapsed: 0.25, cpu_total: Some(cpu), cpu_self: Some(0.0), mem_used: Some(4), mem_total: Some(8), ..RawMetrics::default() }
    }

    fn run(d: &mut Drives, m: &mut MetricsState, seq: &mut u64, cpu: f32, secs: f32) {
        let steps = (secs / 0.05) as usize;
        for k in 0..steps {
            if k % 5 == 0 {
                *seq += 1;
                m.ingest(&raw(*seq, cpu));
            }
            m.tick(0.05);
            d.tick(m, 0.05);
        }
    }

    #[test]
    fn inputs_route_and_delta_reads_change_not_level() {
        let solids = [
            (SolidKind::Cube, Binding::parse("cube|cpu|ram|ram|off").unwrap().inputs),
            (SolidKind::Tetrahedron, Binding::parse("tetrahedron|delta:cpu|off|cpu|cpu").unwrap().inputs),
        ];
        let mut d = Drives::new(&solids, 2.0);
        assert_eq!(d.sources(), [spec("cpu"), spec("ram")]);
        let mut m = MetricsState::new(&Config::default(), d.sources());
        let mut seq = 0;
        run(&mut d, &mut m, &mut seq, 60.0, 20.0);
        let c = d.channels();
        assert!((c[0].level - 0.6).abs() < 0.01, "absolute cpu size: {:?}", c[0]);
        assert!(c[0].spin_y < 0.01, "ram spin: memory is not changing");
        assert!(c[1].level < 0.05, "steady cpu: no change over the period: {:?}", c[1]);
        assert_eq!((c[1].spin_y, c[1].spin_x), (0.0, 0.0), "spin off");
        assert!((d.color_channels()[0].level - 0.5).abs() < 0.01, "ram colour");
        // a jump reads as a full-scale change, then fades once it is older than the period
        run(&mut d, &mut m, &mut seq, 10.0, 1.5);
        assert!(d.channels()[1].level > 0.8, "{:?}", d.channels()[1]);
        run(&mut d, &mut m, &mut seq, 10.0, 6.0);
        assert!(d.channels()[1].level < 0.05, "{:?}", d.channels()[1]);
        assert!(d.readings(&m)[1].is_some_and(|r| r.pct.is_some()));
    }

    #[test]
    fn complexity_climbs_the_ladder_with_hysteresis() {
        let l = ladder();
        assert_eq!(l[0], SolidKind::Tetrahedron);
        assert!(l.windows(2).all(|w| w[0].complexity() <= w[1].complexity()));
        let solids = [(SolidKind::Tetrahedron, Binding::parse("tetrahedron|off|off|off|cpu").unwrap().inputs)];
        let mut d = Drives::new(&solids, 10.0);
        let mut m = MetricsState::new(&Config::default(), d.sources());
        let mut seq = 0;
        run(&mut d, &mut m, &mut seq, 0.0, 5.0);
        assert_eq!(d.shapes().next(), Some(SolidKind::Tetrahedron));
        run(&mut d, &mut m, &mut seq, 100.0, 10.0);
        assert_eq!(d.shapes().next(), Some(*l.last().unwrap()), "full load: the most complex");
        run(&mut d, &mut m, &mut seq, 50.0, 10.0);
        let mid = d.shapes().next().unwrap();
        assert!(mid != l[0] && mid != *l.last().unwrap(), "{mid:?}");
    }
}
