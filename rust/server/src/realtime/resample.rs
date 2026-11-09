//! `media_io.StreamResampler`: a stream that arrives in pieces of any size,
//! resampled with the same output however it was split. A Kaiser-windowed
//! sinc lowpass at the lower Nyquist, applied polyphase: output `k` is
//! centered on input time `k * rate_in / rate_out`. Each output waits for
//! half a kernel of input past its center; `flush` emits the rest as if the
//! input went on as silence and starts over.
//!
//! The arithmetic follows numpy's: the kernel from `np.sinc` and `np.kaiser`
//! (Cephes' `i0`), each output a float64 sum in numpy's pairwise order, cast to
//! float32. Only `sin`, from the platform's libm, can differ from numpy's.

/// Cephes' Chebyshev series, as numpy's `_chbevl`.
fn chbevl(x: f64, vals: &[f64]) -> f64 {
    let mut b0 = vals[0];
    let mut b1 = 0.0;
    let mut b2 = 0.0;
    for &v in &vals[1..] {
        b2 = b1;
        b1 = b0;
        b0 = x * b1 - b2 + v;
    }
    0.5 * (b0 - b2)
}

// numpy's `_i0A` and `_i0B`, copied from its source
const I0_A: [f64; 30] = [-4.4153416464793395e-18, 3.3307945188222384e-17, -2.431279846547955e-16, 1.715391285555133e-15, -1.1685332877993451e-14, 7.676185498604936e-14, -4.856446783111929e-13, 2.95505266312964e-12, -1.726826291441556e-11, 9.675809035373237e-11, -5.189795601635263e-10, 2.6598237246823866e-09, -1.300025009986248e-08, 6.046995022541919e-08, -2.670793853940612e-07, 1.1173875391201037e-06, -4.4167383584587505e-06, 1.6448448070728896e-05, -5.754195010082104e-05, 0.00018850288509584165, -0.0005763755745385824, 0.0016394756169413357, -0.004324309995050576, 0.010546460394594998, -0.02373741480589947, 0.04930528423967071, -0.09490109704804764, 0.17162090152220877, -0.3046826723431984, 0.6767952744094761];
const I0_B: [f64; 25] = [-7.233180487874754e-18, -4.830504485944182e-18, 4.46562142029676e-17, 3.461222867697461e-17, -2.8276239805165836e-16, -3.425485619677219e-16, 1.7725601330565263e-15, 3.8116806693526224e-15, -9.554846698828307e-15, -4.150569347287222e-14, 1.54008621752141e-14, 3.8527783827421426e-13, 7.180124451383666e-13, -1.7941785315068062e-12, -1.3215811840447713e-11, -3.1499165279632416e-11, 1.1889147107846439e-11, 4.94060238822497e-10, 3.3962320257083865e-09, 2.266668990498178e-08, 2.0489185894690638e-07, 2.8913705208347567e-06, 6.889758346916825e-05, 0.0033691164782556943, 0.8044904110141088];

/// numpy's `i0`.
fn i0(x: f64) -> f64 {
    let x = x.abs();
    if x <= 8.0 {
        x.exp() * chbevl(x / 2.0 - 2.0, &I0_A)
    } else {
        x.exp() * chbevl(32.0 / x - 2.0, &I0_B) / x.sqrt()
    }
}

/// numpy's sum of a float64 run: pairwise, eight accumulators per block of at
/// most 128, so a row adds up in the order numpy adds it.
fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &v in a {
            res += v;
        }
        return res;
    }
    if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - n % 8 {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

pub struct StreamResampler {
    up: usize,
    down: usize,
    half: usize,
    kernel: Vec<f64>,
    buf: Vec<f64>,
    /// input index of `buf[0]`
    base: usize,
    /// input samples pushed
    n_in: usize,
    /// output samples emitted
    k_out: usize,
}

impl StreamResampler {
    pub fn new(rate_in: usize, rate_out: usize) -> StreamResampler {
        let (half_width, beta, rolloff) = (48usize, 10.0f64, 0.95f64);
        let g = gcd(rate_in, rate_out);
        let (up, down) = (rate_out / g, rate_in / g);
        let half = half_width * up.max(down);
        let m = 2 * half + 1;
        let cutoff = rolloff * 0.5 / up.max(down) as f64;
        let alpha = (m - 1) as f64 / 2.0;
        let i0_beta = i0(beta);
        let kernel = (0..m)
            .map(|i| {
                let n = i as f64 - half as f64;
                let x = 2.0 * cutoff * n;
                // np.sinc: sin(pi x) / (pi x), with 1e-20 standing in for 0
                let y = std::f64::consts::PI * if x == 0.0 { 1.0e-20 } else { x };
                let sinc = y.sin() / y;
                let r = (i as f64 - alpha) / alpha;
                let kaiser = i0(beta * (1.0 - r * r).sqrt()) / i0_beta;
                2.0 * cutoff * sinc * kaiser * up as f64
            })
            .collect();
        StreamResampler { up, down, half, kernel, buf: Vec::new(), base: 0, n_in: 0, k_out: 0 }
    }

    pub fn reset(&mut self) {
        self.buf.clear();
        self.base = 0;
        self.n_in = 0;
        self.k_out = 0;
    }

    fn ceil_div(a: i64, b: i64) -> i64 {
        -((-a).div_euclid(b))
    }

    pub fn push(&mut self, x: &[f32]) -> Vec<f32> {
        self.buf.extend(x.iter().map(|&v| v as f64));
        self.n_in += x.len();
        // output k is final once every input within half a kernel of it is in
        let k_end = Self::ceil_div((self.n_in * self.up) as i64 - self.half as i64, self.down as i64);
        self.emit(k_end)
    }

    /// Output samples `flush` would still emit after `extra` more input samples.
    pub fn held(&self, extra: usize) -> usize {
        let total = Self::ceil_div(((self.n_in + extra) * self.up) as i64, self.down as i64);
        (total - self.k_out as i64).max(0) as usize
    }

    pub fn flush(&mut self) -> Vec<f32> {
        let k_end = Self::ceil_div((self.n_in * self.up) as i64, self.down as i64);
        let out = self.emit(k_end);
        self.reset();
        out
    }

    fn emit(&mut self, k_end: i64) -> Vec<f32> {
        let (up, down, half) = (self.up as i64, self.down as i64, self.half as i64);
        let k_out = self.k_out as i64;
        if k_end <= k_out || self.buf.is_empty() {
            let n = (k_end - k_out).max(0) as usize;
            self.k_out = self.k_out.max(k_end.max(0) as usize);
            return vec![0.0; n];
        }
        let span = half / up + 1;
        let mut row = vec![0.0f64; (2 * span + 1) as usize];
        let mut out = Vec::with_capacity((k_end - k_out) as usize);
        for k in k_out..k_end {
            let t = k * down; // center, upsampled
            for (slot, tap) in row.iter_mut().zip(-span..=span) {
                let ns = t.div_euclid(up) + tap;
                let idx = t - ns * up + half;
                let ok = (0..=2 * half).contains(&idx) && ns >= 0 && (ns as usize) < self.n_in;
                *slot = if ok {
                    let at = (ns - self.base as i64).clamp(0, self.buf.len() as i64 - 1) as usize;
                    self.buf[at] * self.kernel[idx as usize]
                } else {
                    0.0
                };
            }
            out.push(pairwise_sum(&row) as f32);
        }
        self.k_out = k_end as usize;
        // the first input the next output needs
        let keep = (self.base as i64).max(Self::ceil_div(k_end * down - half, up)) as usize;
        self.buf.drain(..(keep - self.base).min(self.buf.len()));
        self.base = keep;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i0_matches_numpy() {
        // np.i0(0), np.i0(1), np.i0(10)
        assert_eq!(i0(0.0), 1.0);
        assert!((i0(1.0) - 1.266_065_877_752_008_4).abs() < 1e-15);
        assert!((i0(10.0) - 2_815.716_628_466_254).abs() < 1e-9);
    }

    #[test]
    fn the_output_does_not_depend_on_how_the_input_was_split() {
        let x: Vec<f32> = (0..2400).map(|i| ((i as f32) * 0.01).sin()).collect();
        let mut whole = StreamResampler::new(24000, 16000);
        let mut a = whole.push(&x);
        a.extend(whole.flush());
        let mut pieces = StreamResampler::new(24000, 16000);
        let mut b = Vec::new();
        for chunk in x.chunks(97) {
            b.extend(pieces.push(chunk));
        }
        b.extend(pieces.flush());
        assert_eq!(a.len(), 1600);
        assert_eq!(a, b);
    }
}
