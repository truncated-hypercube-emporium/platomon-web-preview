//! Sound level and spectrum of what the computer is playing.
//!
//! * [`Envelope`] turns raw readings into a smooth 0..=1 level with automatic
//!   gain, so quiet and loud material both move the picture.
//! * [`Analyzer`] splits a block of samples into [`BANDS`] log-spaced
//!   frequency bands; [`bass_weighted`] folds them into one value in which low
//!   frequencies count most.
//! * On Windows (screensaver build) [`OutputMeter`] reads the default output
//!   device — the one selected in Windows sound settings — on a background
//!   thread: its WASAPI peak meter, and a loopback capture of what it plays for
//!   the spectrum. It follows a switch to another device.

/// Readings → smooth, auto-gained level.
#[derive(Debug, Clone)]
pub struct Envelope {
    env: f32,
    reference: f32,
    floor: f32,
}

impl Default for Envelope {
    fn default() -> Self {
        Self::new()
    }
}

impl Envelope {
    /// Rise and fall time constants, seconds.
    pub const ATTACK: f32 = 0.03;
    pub const RELEASE: f32 = 0.25;
    /// How long the auto-gain reference takes to fall back after a loud part.
    pub const GAIN_DECAY: f32 = 8.0;
    /// Smallest reference: below this, sound counts as (near) silence and is
    /// not amplified to full size.
    pub const REF_FLOOR: f32 = 0.05;

    pub fn new() -> Self {
        Self::with_floor(Self::REF_FLOOR)
    }

    /// An envelope whose auto-gain reference never falls below `floor`
    /// (smaller for quiet signals such as single treble bands).
    pub fn with_floor(floor: f32) -> Self {
        let floor = if floor.is_finite() { floor.clamp(1e-4, 1.0) } else { Self::REF_FLOOR };
        Self { env: 0.0, reference: floor, floor }
    }

    /// Feed the reading since the last call; returns the level 0..=1.
    pub fn update(&mut self, value: f32, dt: f32) -> f32 {
        let p = if value.is_finite() { value.clamp(0.0, 1.0) } else { 0.0 };
        let dt = dt.clamp(0.0, 1.0);
        let tau = if p > self.env { Self::ATTACK } else { Self::RELEASE };
        self.env += (p - self.env) * (1.0 - (-dt / tau).exp());
        self.reference = (self.reference * (-dt / Self::GAIN_DECAY).exp()).max(self.env).max(self.floor);
        self.level()
    }

    pub fn level(&self) -> f32 {
        (self.env / self.reference).clamp(0.0, 1.0)
    }

    /// The auto-gain reference (what counts as full level now).
    pub fn reference(&self) -> f32 {
        self.reference
    }

    /// Change the smallest reference; takes effect on the next update.
    pub fn set_floor(&mut self, floor: f32) {
        if floor.is_finite() {
            self.floor = floor.clamp(1e-4, 1.0);
        }
    }
}

// ------------------------------------------------------------------ spectrum

/// Number of frequency bands.
pub const BANDS: usize = 12;
/// FFT block length, samples (about 43 ms at 48 kHz: enough for bass).
pub const FFT_N: usize = 2048;
/// Band range, Hz.
pub const BAND_LO: f32 = 20.0;
pub const BAND_HI: f32 = 16_000.0;
/// Frequency where a band's weight in [`bass_weighted`] has fallen to half.
pub const BASS_HZ: f32 = 150.0;

/// Edge `i` (0..=BANDS) of the log-spaced bands, Hz.
pub fn band_edge(i: usize) -> f32 {
    BAND_LO * (BAND_HI / BAND_LO).powf(i as f32 / BANDS as f32)
}

/// Geometric centre of band `b`, Hz.
pub fn band_centre(b: usize) -> f32 {
    (band_edge(b) * band_edge(b + 1)).sqrt()
}

/// Weight of band `b`: 1 at the lowest frequencies, ½ at [`BASS_HZ`], falling
/// further with frequency.
pub fn band_weight(b: usize) -> f32 {
    1.0 / (1.0 + band_centre(b) / BASS_HZ)
}

/// One value 0..=1 from the band amplitudes, low frequencies counting most:
/// a bass tone of a given loudness scores far higher than a treble one.
pub fn bass_weighted(bands: &[f32; BANDS]) -> f32 {
    let v: f32 = bands.iter().enumerate().map(|(b, &e)| band_weight(b) * if e.is_finite() { e.max(0.0) } else { 0.0 }).sum();
    v.clamp(0.0, 1.0)
}

// ------------------------------------------------------- solids ↔ frequencies

/// Auto-gain floor for a solid's band level ([`band_mix`]): single bands,
/// treble especially, are far quieter than the overall peak.
pub const BAND_REF_FLOOR: f32 = 0.01;
/// Width (in bands, one standard deviation) of a solid's slice of the
/// spectrum in [`band_mix`]: neighbouring solids overlap a little.
pub const BAND_SPREAD: f32 = 1.25;

/// How complex a solid is: its edge count (= faces + vertices − 2), faces
/// breaking ties (the icosahedron ranks above the dodecahedron).
pub fn complexity(kind: crate::scene::sdf::SolidKind) -> f32 {
    kind.complexity()
}

/// Each solid's place on the frequency gradient, 0 (bass) ..= 1 (treble):
/// the simplest solid in the scene gets 0, the most complex 1, the rest in
/// between on a log scale of [`complexity`]. Equal solids share a place; a
/// scene of one kind sits in the middle.
pub fn frequency_gradient(kinds: &[crate::scene::sdf::SolidKind]) -> Vec<f32> {
    let c: Vec<f32> = kinds.iter().map(|&k| complexity(k).ln()).collect();
    let (lo, hi) = c.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    c.iter().map(|&v| if hi - lo > 1e-6 { (v - lo) / (hi - lo) } else { 0.5 }).collect()
}

/// Frequency where [`tilt`] is 1, Hz.
pub const TILT_HZ: f32 = 1000.0;

/// Spectral tilt correction for band `b`: `√(f / TILT_HZ)`. Music falls off
/// about 3 dB per octave (a pink spectrum), so after this every band of
/// typical material reads about equally loud and treble solids move too.
pub fn tilt(b: usize) -> f32 {
    (band_centre(b) / TILT_HZ).sqrt()
}

/// A solid's level from the band amplitudes: a Gaussian-weighted average of
/// the tilt-corrected bands around band `pos × (BANDS − 1)`, so `pos` 0
/// hears the bass and 1 the treble.
pub fn band_mix(bands: &[f32; BANDS], pos: f32) -> f32 {
    let p = if pos.is_finite() { pos.clamp(0.0, 1.0) } else { 0.0 };
    let centre = p * (BANDS - 1) as f32;
    let (mut sum, mut wsum) = (0.0, 0.0);
    for (b, &a) in bands.iter().enumerate() {
        let d = (b as f32 - centre) / BAND_SPREAD;
        let w = (-0.5 * d * d).exp();
        sum += w * tilt(b) * if a.is_finite() { a.clamp(0.0, 1.0) } else { 0.0 };
        wsum += w;
    }
    (sum / wsum).min(1.0)
}

/// Each solid's auto-gain never reaches below this share of the loudest
/// solid's: a sound with nothing in a solid's slice leaves it calm instead
/// of amplifying the leftovers to full size.
pub const BAND_RELATIVE_FLOOR: f32 = 0.1;

/// Per-solid levels 0..=1 from the spectrum, each solid listening to its
/// place on the frequency gradient ([`frequency_gradient`]).
#[derive(Debug, Clone)]
pub struct SolidLevels {
    places: Vec<f32>,
    env: Vec<Envelope>,
    levels: Vec<f32>,
}

impl SolidLevels {
    pub fn new(places: Vec<f32>) -> Self {
        let env = places.iter().map(|_| Envelope::with_floor(BAND_REF_FLOOR)).collect();
        let levels = vec![0.0; places.len()];
        Self { places, env, levels }
    }

    pub fn places(&self) -> &[f32] {
        &self.places
    }

    /// Feed the latest band amplitudes; returns one level per solid.
    pub fn update(&mut self, bands: &[f32; BANDS], dt: f32) -> &[f32] {
        let loudest = self.env.iter().map(Envelope::reference).fold(0.0, f32::max);
        let floor = BAND_REF_FLOOR.max(BAND_RELATIVE_FLOOR * loudest);
        for ((&p, e), l) in self.places.iter().zip(&mut self.env).zip(&mut self.levels) {
            e.set_floor(floor);
            *l = e.update(band_mix(bands, p), dt);
        }
        &self.levels
    }
}

/// In-place radix-2 complex FFT; `re.len()` must be a power of two.
pub fn fft(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two() && im.len() == n);
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * std::f32::consts::PI / len as f32;
        let (wr, wi) = (ang.cos(), ang.sin());
        for start in (0..n).step_by(len) {
            let (mut cr, mut ci) = (1.0f32, 0.0f32);
            for k in 0..len / 2 {
                let (a, b) = (start + k, start + k + len / 2);
                let (tr, ti) = (re[b] * cr - im[b] * ci, re[b] * ci + im[b] * cr);
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                let nr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = nr;
            }
        }
        len <<= 1;
    }
}

/// Hann-windowed FFT → band amplitudes.
pub struct Analyzer {
    window: Vec<f32>,
    gain: f32,
    re: Vec<f32>,
    im: Vec<f32>,
}

impl Default for Analyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl Analyzer {
    pub fn new() -> Self {
        let window: Vec<f32> = (0..FFT_N)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / FFT_N as f32).cos())
            .collect();
        // A full-scale sine then reads as amplitude 1 at its bin.
        let gain = 2.0 / window.iter().sum::<f32>();
        Self { window, gain, re: vec![0.0; FFT_N], im: vec![0.0; FFT_N] }
    }

    /// Amplitude (0..≈1 of full scale) of the strongest component in each
    /// band of the last [`FFT_N`] `samples` (mono) at `rate` Hz.
    pub fn bands(&mut self, samples: &[f32], rate: f32) -> [f32; BANDS] {
        let mut out = [0.0; BANDS];
        if samples.len() < FFT_N || rate <= 0.0 {
            return out;
        }
        let s = &samples[samples.len() - FFT_N..];
        for ((re, &x), &w) in self.re.iter_mut().zip(s).zip(&self.window) {
            *re = if x.is_finite() { x } else { 0.0 } * w;
        }
        self.im.fill(0.0);
        fft(&mut self.re, &mut self.im);
        let hz_per_bin = rate / FFT_N as f32;
        for k in 1..FFT_N / 2 {
            let f = k as f32 * hz_per_bin;
            if !(BAND_LO..BAND_HI).contains(&f) {
                continue;
            }
            let b = (((f / BAND_LO).ln() / (BAND_HI / BAND_LO).ln()) * BANDS as f32) as usize;
            let a = self.re[k].hypot(self.im[k]) * self.gain;
            let b = b.min(BANDS - 1);
            out[b] = out[b].max(a);
        }
        out
    }
}

#[cfg(all(windows, feature = "screensaver"))]
pub use win::OutputMeter;

#[cfg(all(windows, feature = "screensaver"))]
mod win {
    use super::{Analyzer, BANDS, FFT_N};
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    #[repr(C)]
    struct Guid(u32, u16, u16, [u8; 8]);

    const CLSID_MM_DEVICE_ENUMERATOR: Guid =
        Guid(0xBCDE0395, 0xE52F, 0x467C, [0x8E, 0x3D, 0xC4, 0x57, 0x92, 0x91, 0x69, 0x2E]);
    const IID_IMM_DEVICE_ENUMERATOR: Guid =
        Guid(0xA95664D2, 0x9614, 0x4F35, [0xA7, 0x46, 0xDE, 0x8D, 0xB6, 0x36, 0x17, 0xE6]);
    const IID_IAUDIO_METER_INFORMATION: Guid =
        Guid(0xC02216F6, 0x8C67, 0x4B5B, [0x9D, 0x00, 0xD0, 0x08, 0xE7, 0x3E, 0x00, 0x64]);
    const IID_IAUDIO_CLIENT: Guid = Guid(0x1CB9AD4C, 0xDBFA, 0x4C32, [0xB1, 0x78, 0xC2, 0xF5, 0x68, 0xA7, 0x03, 0xB2]);
    const IID_IAUDIO_CAPTURE_CLIENT: Guid =
        Guid(0xC8ADBD64, 0xE71E, 0x48A0, [0xA4, 0xDE, 0x18, 0x5C, 0x39, 0x5C, 0xD3, 0x17]);
    const CLSCTX_ALL: u32 = 0x17;
    const COINIT_MULTITHREADED: u32 = 0;
    const E_RENDER: i32 = 0;
    const E_CONSOLE: i32 = 0;
    const AUDCLNT_SHAREMODE_SHARED: i32 = 0;
    const AUDCLNT_STREAMFLAGS_LOOPBACK: u32 = 0x0002_0000;
    const AUDCLNT_BUFFERFLAGS_SILENT: u32 = 0x2;
    const WAVE_FORMAT_PCM: u16 = 1;
    const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;
    const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;

    #[link(name = "ole32")]
    unsafe extern "system" {
        fn CoInitializeEx(reserved: *mut c_void, flags: u32) -> i32;
        fn CoUninitialize();
        fn CoCreateInstance(
            clsid: *const Guid,
            outer: *mut c_void,
            ctx: u32,
            iid: *const Guid,
            out: *mut *mut c_void,
        ) -> i32;
        fn CoTaskMemFree(p: *mut c_void);
    }

    /// Slot `i` of a COM object's vtable.
    ///
    /// # Safety
    /// `obj` must be a live COM interface pointer with at least `i + 1` slots.
    unsafe fn slot(obj: *mut c_void, i: usize) -> *const c_void {
        // SAFETY: a COM pointer points at its vtable pointer.
        unsafe { *(*(obj as *const *const *const c_void)).add(i) }
    }

    /// An owned COM interface pointer, released on drop.
    struct Com(*mut c_void);

    impl Drop for Com {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: IUnknown::Release is slot 2 of every interface.
                unsafe {
                    let release: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(slot(self.0, 2));
                    release(self.0);
                }
            }
        }
    }

    fn enumerator() -> Option<Com> {
        let mut e = std::ptr::null_mut();
        // SAFETY: valid GUIDs and out pointer.
        let hr = unsafe {
            CoCreateInstance(&CLSID_MM_DEVICE_ENUMERATOR, std::ptr::null_mut(), CLSCTX_ALL, &IID_IMM_DEVICE_ENUMERATOR, &mut e)
        };
        (hr >= 0 && !e.is_null()).then(|| Com(e))
    }

    /// The default output device and its id.
    fn default_device(e: &Com) -> Option<(Com, Vec<u16>)> {
        // SAFETY: IMMDeviceEnumerator::GetDefaultAudioEndpoint (slot 4) and
        // IMMDevice::GetId (slot 5) with valid out pointers; the id string is
        // copied, then freed with CoTaskMemFree as documented.
        unsafe {
            let get_default: unsafe extern "system" fn(*mut c_void, i32, i32, *mut *mut c_void) -> i32 =
                std::mem::transmute(slot(e.0, 4));
            let mut dev = std::ptr::null_mut();
            if get_default(e.0, E_RENDER, E_CONSOLE, &mut dev) < 0 || dev.is_null() {
                return None;
            }
            let dev = Com(dev);
            let get_id: unsafe extern "system" fn(*mut c_void, *mut *mut u16) -> i32 = std::mem::transmute(slot(dev.0, 5));
            let mut p = std::ptr::null_mut();
            let mut id = Vec::new();
            if get_id(dev.0, &mut p) >= 0 && !p.is_null() {
                let mut n = 0;
                while *p.add(n) != 0 {
                    n += 1;
                }
                id.extend_from_slice(std::slice::from_raw_parts(p, n));
                CoTaskMemFree(p as *mut c_void);
            }
            Some((dev, id))
        }
    }

    /// IMMDevice::Activate (slot 3).
    fn activate(dev: &Com, iid: &Guid) -> Option<Com> {
        let mut out = std::ptr::null_mut();
        // SAFETY: valid interface, GUID and out pointer.
        let hr = unsafe {
            let f: unsafe extern "system" fn(*mut c_void, *const Guid, u32, *mut c_void, *mut *mut c_void) -> i32 =
                std::mem::transmute(slot(dev.0, 3));
            f(dev.0, iid, CLSCTX_ALL, std::ptr::null_mut(), &mut out)
        };
        (hr >= 0 && !out.is_null()).then(|| Com(out))
    }

    /// IAudioMeterInformation::GetPeakValue (slot 3).
    fn read_peak(meter: &Com) -> Option<f32> {
        let mut v = 0.0f32;
        // SAFETY: valid interface and out pointer.
        let hr = unsafe {
            let get: unsafe extern "system" fn(*mut c_void, *mut f32) -> i32 = std::mem::transmute(slot(meter.0, 3));
            get(meter.0, &mut v)
        };
        (hr >= 0).then_some(v)
    }

    #[derive(Clone, Copy)]
    enum Sample {
        F32,
        I16,
    }

    /// A loopback capture of everything the device plays.
    struct Capture {
        client: Com,
        capture: Com,
        channels: usize,
        rate: f32,
        sample: Sample,
    }

    impl Capture {
        fn open(dev: &Com) -> Option<Self> {
            let client = activate(dev, &IID_IAUDIO_CLIENT)?;
            // SAFETY: IAudioClient GetMixFormat (8), Initialize (3),
            // GetService (14), Start (10) with valid pointers; the mix format
            // is read within its documented size and freed after Initialize
            // (which copies it).
            unsafe {
                let get_mix: unsafe extern "system" fn(*mut c_void, *mut *mut u8) -> i32 =
                    std::mem::transmute(slot(client.0, 8));
                let mut fmt = std::ptr::null_mut();
                if get_mix(client.0, &mut fmt) < 0 || fmt.is_null() {
                    return None;
                }
                let u16_at = |o: usize| u16::from_le_bytes([*fmt.add(o), *fmt.add(o + 1)]);
                let u32_at = |o: usize| u32::from_le_bytes([*fmt.add(o), *fmt.add(o + 1), *fmt.add(o + 2), *fmt.add(o + 3)]);
                let tag = u16_at(0);
                let channels = u16_at(2) as usize;
                let rate = u32_at(4) as f32;
                let bits = u16_at(14);
                let kind = if tag == WAVE_FORMAT_EXTENSIBLE && u16_at(16) >= 22 { u32_at(24) as u16 } else { tag };
                let sample = match (kind, bits) {
                    (WAVE_FORMAT_IEEE_FLOAT, 32) => Some(Sample::F32),
                    (WAVE_FORMAT_PCM, 16) => Some(Sample::I16),
                    _ => None,
                };
                let init: unsafe extern "system" fn(*mut c_void, i32, u32, i64, i64, *const u8, *const Guid) -> i32 =
                    std::mem::transmute(slot(client.0, 3));
                // 200 ms shared-mode buffer.
                let hr = init(client.0, AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_LOOPBACK, 2_000_000, 0, fmt, std::ptr::null());
                CoTaskMemFree(fmt as *mut c_void);
                let sample = sample?;
                if hr < 0 || channels == 0 || rate <= 0.0 {
                    return None;
                }
                let get_service: unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32 =
                    std::mem::transmute(slot(client.0, 14));
                let mut cap = std::ptr::null_mut();
                if get_service(client.0, &IID_IAUDIO_CAPTURE_CLIENT, &mut cap) < 0 || cap.is_null() {
                    return None;
                }
                let capture = Com(cap);
                let start: unsafe extern "system" fn(*mut c_void) -> i32 = std::mem::transmute(slot(client.0, 10));
                if start(client.0) < 0 {
                    return None;
                }
                Some(Self { client, capture, channels, rate, sample })
            }
        }

        /// Pass every captured frame (mono mix) to `push`. Err = the stream
        /// broke (device gone): reopen.
        fn drain(&self, mut push: impl FnMut(f32)) -> Result<(), ()> {
            // SAFETY: IAudioCaptureClient GetNextPacketSize (5), GetBuffer (3)
            // and ReleaseBuffer (4); the buffer holds `frames × channels`
            // samples of the negotiated format until ReleaseBuffer.
            unsafe {
                let next: unsafe extern "system" fn(*mut c_void, *mut u32) -> i32 = std::mem::transmute(slot(self.capture.0, 5));
                let get: unsafe extern "system" fn(*mut c_void, *mut *mut u8, *mut u32, *mut u32, *mut u64, *mut u64) -> i32 =
                    std::mem::transmute(slot(self.capture.0, 3));
                let release: unsafe extern "system" fn(*mut c_void, u32) -> i32 = std::mem::transmute(slot(self.capture.0, 4));
                loop {
                    let mut packet = 0u32;
                    if next(self.capture.0, &mut packet) < 0 {
                        return Err(());
                    }
                    if packet == 0 {
                        return Ok(());
                    }
                    let (mut data, mut frames, mut flags) = (std::ptr::null_mut(), 0u32, 0u32);
                    if get(self.capture.0, &mut data, &mut frames, &mut flags, std::ptr::null_mut(), std::ptr::null_mut()) < 0 {
                        return Err(());
                    }
                    let (n, ch) = (frames as usize, self.channels);
                    if flags & AUDCLNT_BUFFERFLAGS_SILENT != 0 || data.is_null() {
                        (0..n).for_each(|_| push(0.0));
                    } else {
                        match self.sample {
                            Sample::F32 => {
                                let s = std::slice::from_raw_parts(data as *const f32, n * ch);
                                for f in s.chunks_exact(ch) {
                                    push(f.iter().sum::<f32>() / ch as f32);
                                }
                            }
                            Sample::I16 => {
                                let s = std::slice::from_raw_parts(data as *const i16, n * ch);
                                for f in s.chunks_exact(ch) {
                                    push(f.iter().map(|&v| v as f32 / 32768.0).sum::<f32>() / ch as f32);
                                }
                            }
                        }
                    }
                    release(self.capture.0, frames);
                }
            }
        }
    }

    impl Drop for Capture {
        fn drop(&mut self) {
            // SAFETY: IAudioClient::Stop (slot 11) on a started client.
            unsafe {
                let stop: unsafe extern "system" fn(*mut c_void) -> i32 = std::mem::transmute(slot(self.client.0, 11));
                stop(self.client.0);
            }
        }
    }

    /// The default output device's meter and capture.
    struct Source {
        id: Vec<u16>,
        meter: Option<Com>,
        capture: Option<Capture>,
    }

    impl Source {
        fn open(dev: &Com, id: Vec<u16>) -> Self {
            Self { id, meter: activate(dev, &IID_IAUDIO_METER_INFORMATION), capture: Capture::open(dev) }
        }
    }

    /// Reads the default output device on a background thread.
    pub struct OutputMeter {
        /// Highest peak since the last [`OutputMeter::take_peak`], as f32 bits.
        peak: Arc<AtomicU32>,
        bands: Arc<Mutex<[f32; BANDS]>>,
        stop: Arc<AtomicBool>,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    impl OutputMeter {
        const POLL: Duration = Duration::from_millis(5);
        /// Check this often whether the default device changed.
        const CHECK: Duration = Duration::from_secs(2);
        /// New samples between spectra (about 10 ms at 48 kHz).
        const HOP: usize = 512;
        /// No samples for this long = silence (loopback sends nothing then).
        const QUIET: Duration = Duration::from_millis(100);

        pub fn spawn() -> Self {
            let peak = Arc::new(AtomicU32::new(0));
            let bands = Arc::new(Mutex::new([0.0; BANDS]));
            let stop = Arc::new(AtomicBool::new(false));
            let (p, b, s) = (peak.clone(), bands.clone(), stop.clone());
            let thread = std::thread::Builder::new()
                .name("platomon-audio".into())
                .spawn(move || {
                    // SAFETY: balanced with CoUninitialize below on this thread.
                    let init = unsafe { CoInitializeEx(std::ptr::null_mut(), COINIT_MULTITHREADED) } >= 0;
                    if let Some(e) = enumerator() {
                        Self::poll(&e, &p, &b, &s);
                    }
                    if init {
                        // SAFETY: matches the successful CoInitializeEx.
                        unsafe { CoUninitialize() };
                    }
                })
                .ok();
            Self { peak, bands, stop, thread }
        }

        fn poll(e: &Com, peak: &AtomicU32, bands: &Mutex<[f32; BANDS]>, stop: &AtomicBool) {
            let mut src: Option<Source> = None;
            let mut checked: Option<Instant> = None;
            let mut ring = vec![0.0f32; FFT_N];
            let mut write = 0usize;
            let mut fresh = 0usize;
            let mut block = vec![0.0f32; FFT_N];
            let mut analyzer = Analyzer::new();
            let mut last_data = Instant::now();
            while !stop.load(Ordering::Relaxed) {
                if checked.is_none_or(|t| t.elapsed() >= Self::CHECK) {
                    checked = Some(Instant::now());
                    match default_device(e) {
                        Some((dev, id)) if src.as_ref().is_none_or(|s| s.id != id) => src = Some(Source::open(&dev, id)),
                        None => src = None,
                        _ => {}
                    }
                }
                let mut broken = false;
                let mut rate = 0.0;
                if let Some(s) = &src {
                    match s.meter.as_ref().map(read_peak) {
                        Some(Some(v)) => {
                            // Non-negative f32s order like their bit patterns.
                            let v = if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
                            peak.fetch_max(v.to_bits(), Ordering::Relaxed);
                        }
                        Some(None) => broken = true,
                        None => {}
                    }
                    if let Some(c) = &s.capture {
                        rate = c.rate;
                        let r = c.drain(|v| {
                            ring[write] = v;
                            write = (write + 1) % FFT_N;
                            fresh += 1;
                        });
                        broken |= r.is_err();
                    }
                }
                if broken {
                    src = None;
                    checked = None;
                }
                if fresh >= Self::HOP && rate > 0.0 {
                    fresh = 0;
                    last_data = Instant::now();
                    block[..FFT_N - write].copy_from_slice(&ring[write..]);
                    block[FFT_N - write..].copy_from_slice(&ring[..write]);
                    let v = analyzer.bands(&block, rate);
                    if let Ok(mut g) = bands.lock() {
                        *g = v;
                    }
                } else if last_data.elapsed() > Self::QUIET {
                    last_data = Instant::now();
                    ring.fill(0.0);
                    if let Ok(mut g) = bands.lock() {
                        *g = [0.0; BANDS];
                    }
                }
                std::thread::sleep(Self::POLL);
            }
        }

        /// Highest peak (0..=1) since the last call, then reset.
        pub fn take_peak(&self) -> f32 {
            f32::from_bits(self.peak.swap(0, Ordering::Relaxed))
        }

        /// The latest band amplitudes (all 0 in silence or without capture).
        pub fn bands(&self) -> [f32; BANDS] {
            self.bands.lock().map(|g| *g).unwrap_or([0.0; BANDS])
        }
    }

    impl Drop for OutputMeter {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(t) = self.thread.take() {
                let _ = t.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_stays_at_zero() {
        let mut e = Envelope::new();
        for _ in 0..200 {
            assert_eq!(e.update(0.0, 1.0 / 60.0), 0.0);
        }
    }

    #[test]
    fn rises_fast_falls_slower_and_auto_gains() {
        let dt = 1.0 / 60.0;
        let mut e = Envelope::new();
        // a quiet tone (peak 0.2) reaches full level: the reference follows it
        let mut l = 0.0;
        for _ in 0..30 {
            l = e.update(0.2, dt);
        }
        assert!(l > 0.95, "auto-gained to full: {l}");
        // attack: from silence to most of the way within ~0.1 s
        let mut e = Envelope::new();
        let mut t = 0;
        while e.update(0.5, dt) < 0.9 {
            t += 1;
        }
        assert!(t <= 8, "attack frames {t}");
        // release: after the sound stops, it takes several frames to fall
        let after: Vec<f32> = (0..30).map(|_| e.update(0.0, dt)).collect();
        assert!(after[0] > 0.8 && after[5] > 0.3, "falls gradually: {after:?}");
        assert!(after[29] < 0.2, "and does fall: {}", after[29]);
        // after a loud burst, a quiet part shows as a small level at first
        let mut e = Envelope::new();
        for _ in 0..30 {
            e.update(1.0, dt);
        }
        for _ in 0..30 {
            e.update(0.2, dt);
        }
        assert!(e.level() < 0.35, "quiet after loud: {}", e.level());
        // ... and grows back as the reference decays
        for _ in 0..60 * 20 {
            e.update(0.2, dt);
        }
        assert!(e.level() > 0.95, "re-gained: {}", e.level());
    }

    #[test]
    fn near_silence_is_not_amplified() {
        let mut e = Envelope::new();
        for _ in 0..600 {
            e.update(0.005, 1.0 / 60.0);
        }
        assert!(e.level() < 0.15, "noise floor stays small: {}", e.level());
    }

    /// One FFT block of a sine at `hz`.
    fn sine(hz: f32, amp: f32, rate: f32) -> Vec<f32> {
        (0..FFT_N).map(|i| amp * (2.0 * std::f32::consts::PI * hz * i as f32 / rate).sin()).collect()
    }

    #[test]
    fn fft_finds_a_sine() {
        let n = 64;
        let mut re: Vec<f32> = (0..n).map(|i| (2.0 * std::f32::consts::PI * 5.0 * i as f32 / n as f32).cos()).collect();
        let mut im = vec![0.0; n];
        fft(&mut re, &mut im);
        let mag: Vec<f32> = re.iter().zip(&im).map(|(r, i)| r.hypot(*i)).collect();
        assert!((mag[5] - n as f32 / 2.0).abs() < 1e-3 && (mag[59] - n as f32 / 2.0).abs() < 1e-3);
        assert!(mag.iter().enumerate().all(|(k, &m)| k == 5 || k == 59 || m < 1e-3));
    }

    #[test]
    fn bands_place_tones_and_measure_amplitude() {
        let mut a = Analyzer::new();
        let rate = 48_000.0;
        // tones on FFT bins 2, 19 and 213 (≈47 Hz, 445 Hz, 4.99 kHz), away from band edges
        let bin = |k: f32| k * rate / FFT_N as f32;
        for (hz, amp) in [(bin(2.0), 0.5), (bin(19.0), 0.8), (bin(213.0), 0.3)] {
            let b = a.bands(&sine(hz, amp, rate), rate);
            let strongest = (0..BANDS).max_by(|&i, &j| b[i].total_cmp(&b[j])).unwrap();
            assert!(band_edge(strongest) <= hz && hz < band_edge(strongest + 1), "{hz} Hz in band {strongest}: {b:?}");
            assert!((b[strongest] - amp).abs() < 0.12 * amp, "{hz} Hz amplitude {} vs {amp}", b[strongest]);
        }
        assert_eq!(a.bands(&[0.0; FFT_N], rate), [0.0; BANDS], "silence");
        assert_eq!(a.bands(&[0.5; 10], rate), [0.0; BANDS], "too short");
    }

    #[test]
    fn low_frequencies_weigh_most() {
        let mut a = Analyzer::new();
        let rate = 48_000.0;
        let score = |hz: f32, a: &mut Analyzer| bass_weighted(&a.bands(&sine(hz, 0.5, rate), rate));
        let (bass, mid, treble) = (score(50.0, &mut a), score(800.0, &mut a), score(6000.0, &mut a));
        assert!(bass > 2.0 * mid && mid > 2.0 * treble, "{bass} {mid} {treble}");
        assert!(bass > 0.3, "a loud bass tone scores high: {bass}");
        for b in 1..BANDS {
            assert!(band_weight(b) < band_weight(b - 1));
        }
        assert!((band_weight(0) - 1.0).abs() < 0.25);
        assert_eq!(bass_weighted(&[f32::NAN; BANDS]), 0.0);
        assert_eq!(bass_weighted(&[10.0; BANDS]), 1.0, "clamped");
    }

    /// Reads the real default output device for 2 s and prints the highest
    /// peak and band amplitudes (play something to see them move): `cargo test
    /// --features screensaver meter_reads -- --ignored --nocapture`.
    #[cfg(all(windows, feature = "screensaver"))]
    #[test]
    #[ignore]
    fn meter_reads_output_device() {
        let m = OutputMeter::spawn();
        let (mut max, mut bmax) = (0.0f32, [0.0f32; BANDS]);
        for _ in 0..40 {
            std::thread::sleep(std::time::Duration::from_millis(50));
            let p = m.take_peak();
            assert!((0.0..=1.0).contains(&p));
            max = max.max(p);
            for (o, v) in bmax.iter_mut().zip(m.bands()) {
                *o = o.max(v);
            }
        }
        let bands: Vec<String> = bmax.iter().map(|v| format!("{v:.3}")).collect();
        eprintln!("highest output peak over 2 s: {max:.3}");
        eprintln!("highest band amplitudes (low→high): {}", bands.join(" "));
        eprintln!("bass-weighted: {:.3}", bass_weighted(&bmax));
    }

    #[test]
    fn simple_solids_get_bass_complex_ones_treble() {
        use crate::scene::sdf::SolidKind as K;
        let kinds = [K::Icosahedron, K::Cube, K::Tetrahedron, K::Dodecahedron, K::Octahedron];
        let g = frequency_gradient(&kinds);
        assert_eq!(g[2], 0.0, "tetrahedron: lowest");
        assert_eq!(g[0], 1.0, "icosahedron: highest");
        assert!(g[2] < g[1] && g[1] < g[4] && g[4] < g[3] && g[3] < g[0], "gradient order {g:?}");
        // one kind only: middle of the spectrum
        assert_eq!(frequency_gradient(&[K::Cube, K::Cube]), vec![0.5, 0.5]);
        // Archimedean solids extend the scale
        let g = frequency_gradient(&[K::Tetrahedron, K::Icosahedron, K::TruncatedIcosahedron]);
        assert!(g[0] == 0.0 && g[1] > 0.3 && g[1] < 0.8 && g[2] == 1.0, "{g:?}");
    }

    #[test]
    fn band_mix_follows_the_solids_place() {
        let mut a = Analyzer::new();
        let rate = 48_000.0;
        let bin = |k: f32| k * rate / FFT_N as f32;
        let bass = a.bands(&sine(bin(2.0), 0.5, rate), rate);
        let treble = a.bands(&sine(bin(300.0), 0.5, rate), rate);
        // a bass tone moves the bass solid, not the treble one, and vice versa
        assert!(band_mix(&bass, 0.0) > 5.0 * band_mix(&bass, 1.0), "{bass:?}");
        assert!(band_mix(&treble, 1.0) > 5.0 * band_mix(&treble, 0.0), "{treble:?}");
        // in between, the level slides along the gradient
        let mid: Vec<f32> = (0..=4).map(|i| band_mix(&treble, i as f32 / 4.0)).collect();
        assert!(mid.windows(2).all(|w| w[0] <= w[1] + 1e-6), "{mid:?}");
        assert_eq!(band_mix(&[0.0; BANDS], 0.5), 0.0);
        assert_eq!(band_mix(&[f32::NAN; BANDS], f32::NAN), 0.0);
        // band envelopes auto-gain small levels
        let mut e = Envelope::with_floor(BAND_REF_FLOOR);
        let mut l = 0.0;
        for _ in 0..30 {
            l = e.update(0.02, 1.0 / 60.0);
        }
        assert!(l > 0.95, "{l}");
    }

    #[test]
    fn solid_levels_split_the_spectrum() {
        let mut a = Analyzer::new();
        let rate = 48_000.0;
        let bin = |k: f32| k * rate / FFT_N as f32;
        let dt = 1.0 / 60.0;
        let run = |bands: [f32; BANDS]| {
            let mut s = SolidLevels::new(vec![0.0, 0.5, 1.0]);
            for _ in 0..120 {
                s.update(&bands, dt);
            }
            s.update(&bands, dt).to_vec()
        };
        // a bass tone: the simplest solid at full size, the most complex calm
        let l = run(a.bands(&sine(bin(2.0), 0.3, rate), rate));
        assert!(l[0] > 0.9 && l[2] < 0.15, "bass tone {l:?}");
        // a treble tone: the other way round
        let l = run(a.bands(&sine(bin(400.0), 0.3, rate), rate));
        assert!(l[2] > 0.9 && l[0] < 0.15, "treble tone {l:?}");
        // a mid tone: the middle solid leads
        let l = run(a.bands(&sine(bin(25.0), 0.3, rate), rate));
        assert!(l[1] > 0.9 && l[0] < 0.5 && l[2] < 0.5, "mid tone {l:?}");
        // a pink-ish spectrum (falling 3 dB/octave, like music): all move
        let mut pink = [0.0; BANDS];
        for (b, v) in pink.iter_mut().enumerate() {
            *v = 0.05 / tilt(b);
        }
        let l = run(pink);
        assert!(l.iter().all(|&v| v > 0.9), "pink {l:?}");
        // silence: all still
        assert!(run([0.0; BANDS]).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn junk_input_is_safe() {
        let mut e = Envelope::new();
        for v in [f32::NAN, f32::INFINITY, -1.0, 5.0] {
            let l = e.update(v, 0.016);
            assert!((0.0..=1.0).contains(&l));
        }
    }
}
