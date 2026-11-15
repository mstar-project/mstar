//! Audio on a `/v1/realtime` socket, in OpenAI's formats: PCM16 at a rate in
//! [`REALTIME_RATES`], or G.711 (8 kHz telephony), as `serving_realtime`'s
//! `AudioFormat` and `media_io`'s G.711 and PCM16 codecs.

use serde_json::{json, Map, Value};

use super::excerpt;

/// OpenAI's PCM16 rate, and G.711's.
pub const OPENAI_PCM_RATE: usize = 24000;
pub const G711_RATE: usize = 8000;
/// The PCM rates a socket takes. A closed set bounds what resampling one
/// append costs, which a bound on the kernel's size alone does not.
pub const REALTIME_RATES: [usize; 8] = [8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000];

/// A socket's audio on the wire: `audio/pcm` (PCM16, little endian, mono),
/// `audio/pcmu` or `audio/pcma` (G.711), at `rate` Hz.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioFormat {
    pub codec: Codec,
    pub rate: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    Pcm,
    Pcmu,
    Pcma,
}

impl Codec {
    fn name(self) -> &'static str {
        match self {
            Codec::Pcm => "audio/pcm",
            Codec::Pcmu => "audio/pcmu",
            Codec::Pcma => "audio/pcma",
        }
    }
}

impl AudioFormat {
    pub const fn pcm(rate: usize) -> AudioFormat {
        AudioFormat { codec: Codec::Pcm, rate }
    }

    pub fn view(&self) -> Value {
        match self.codec {
            Codec::Pcm => json!({"type": "audio/pcm", "rate": self.rate}),
            other => json!({"type": other.name()}),
        }
    }

    /// The transcription intent's beta name for this format, if it has one.
    /// `"pcm16"` is 16 kHz, as this endpoint has always read it (OpenAI reads
    /// it as 24 kHz).
    pub fn beta_name(&self) -> Option<&'static str> {
        match (self.codec, self.rate) {
            (Codec::Pcm, 16000) => Some("pcm16"),
            (Codec::Pcmu, G711_RATE) => Some("g711_ulaw"),
            (Codec::Pcma, G711_RATE) => Some("g711_alaw"),
            _ => None,
        }
    }

    pub fn from_beta_name(name: &str) -> Option<AudioFormat> {
        match name {
            "pcm16" => Some(AudioFormat::pcm(16000)),
            "g711_ulaw" => Some(AudioFormat { codec: Codec::Pcmu, rate: G711_RATE }),
            "g711_alaw" => Some(AudioFormat { codec: Codec::Pcma, rate: G711_RATE }),
            _ => None,
        }
    }
}

/// `_audio_format`: the GA format object `fmt`, `{"type": "audio/pcm", "rate": N}`
/// (24 kHz when left out) or `{"type": "audio/pcmu" | "audio/pcma"}` (8 kHz).
/// `fmt` is already known to be an object or null.
pub fn audio_format(fmt: Option<&Map<String, Value>>, name: &str) -> Result<AudioFormat, String> {
    let empty = Map::new();
    let fmt = fmt.unwrap_or(&empty);
    let codec = match fmt.get("type").and_then(Value::as_str) {
        Some("audio/pcm") => Codec::Pcm,
        Some("audio/pcmu") => Codec::Pcmu,
        Some("audio/pcma") => Codec::Pcma,
        _ => {
            let got = fmt.get("type").cloned().unwrap_or(Value::Null);
            return Err(format!("{name}.type must be \"audio/pcm\", \"audio/pcmu\" or \"audio/pcma\"; got {}", excerpt(&got)));
        }
    };
    let mut extra: Vec<&String> = fmt.keys().filter(|k| *k != "type" && *k != "rate").collect();
    if !extra.is_empty() {
        // channels, encoding and the like would change how the bytes read
        extra.sort();
        let list = Value::Array(extra.iter().map(|k| Value::from(k.as_str())).collect());
        return Err(format!("{name} takes only type and rate; got {}", excerpt(&list)));
    }
    let default = if codec == Codec::Pcm { OPENAI_PCM_RATE } else { G711_RATE };
    let rate = fmt.get("rate").cloned().unwrap_or(Value::from(default));
    // Python's ==: 8000.0 equals 8000, a bool equals nothing here
    let equals = |want: usize| !rate.is_boolean() && rate.as_f64() == Some(want as f64);
    if codec != Codec::Pcm && !equals(G711_RATE) {
        return Err(format!("{name}: {} is {G711_RATE} Hz; got rate {}", codec.name(), excerpt(&rate)));
    }
    match rate.as_u64().map(|r| r as usize) {
        Some(r) if (rate.is_u64() || rate.is_i64()) && REALTIME_RATES.contains(&r) => Ok(AudioFormat { codec, rate: r }),
        _ => Err(format!("{name}.rate {} Hz is not supported; use one of {REALTIME_RATES:?}", excerpt(&rate))),
    }
}

/// `decode_base64`: an append's payload, strictly. Whitespace is allowed
/// (base64.encodebytes wraps lines); anything else outside the alphabet is an
/// error, as is a payload that is not a string.
fn decode_base64(payload: &Value) -> Result<Vec<u8>, ()> {
    use base64::Engine;
    let s = payload.as_str().ok_or(())?;
    if !s.is_ascii() {
        return Err(());
    }
    let compact: String = s.chars().filter(|c| !" \t\n\r\x0b\x0c".contains(*c)).collect();
    base64::engine::general_purpose::STANDARD.decode(compact.as_bytes()).map_err(|_| ())
}

/// `decode_append`: a base64 append in `codec` as float32 in [-1, 1), and the
/// odd trailing byte. `carry` is the previous append's odd byte (PCM16 only; a
/// G.711 sample is one byte), so a sample split across two appends is joined.
pub fn decode_append(payload: &Value, carry: Option<u8>, codec: Codec) -> Result<(Vec<f32>, Option<u8>), ()> {
    let mut raw: Vec<u8> = carry.into_iter().collect();
    raw.extend(decode_base64(payload)?);
    if codec != Codec::Pcm {
        let table = g711_table(codec);
        return Ok((raw.iter().map(|&b| table[b as usize] as f32 / 32768.0).collect(), None));
    }
    let odd = (raw.len() % 2 == 1).then(|| raw[raw.len() - 1]);
    let samples = raw.as_chunks::<2>().0.iter().map(|b| i16::from_le_bytes(*b) as f32 / 32768.0).collect();
    Ok((samples, odd))
}

/// `to_pcm16`: float audio as int16, rounded half to even and clipped, the
/// inverse of reading PCM16 as `int16 / 32768`.
pub fn to_pcm16(audio: &[f32]) -> Vec<i16> {
    audio.iter().map(|&v| (v as f64 * 32768.0).round_ties_even().clamp(-32768.0, 32767.0) as i16).collect()
}

/// `encode_audio`: float audio as base64 in `codec`.
pub fn encode_audio(audio: &[f32], codec: Codec) -> String {
    use base64::Engine;
    let pcm = to_pcm16(audio);
    let bytes: Vec<u8> = match codec {
        Codec::Pcm => pcm.iter().flat_map(|s| s.to_le_bytes()).collect(),
        law => pcm.iter().map(|&s| g711_encode(s, law)).collect(),
    };
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

// G.711, coded as CPython's audioop and Sun's g711.c code it: mu-law from
// 14-bit magnitudes, A-law from 13-bit ones.
const ULAW_SEG_END: [i32; 8] = [0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF];
const ALAW_SEG_END: [i32; 8] = [0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF];

fn g711_table(law: Codec) -> &'static [i16; 256] {
    static ULAW: std::sync::OnceLock<[i16; 256]> = std::sync::OnceLock::new();
    static ALAW: std::sync::OnceLock<[i16; 256]> = std::sync::OnceLock::new();
    match law {
        Codec::Pcmu => ULAW.get_or_init(|| {
            std::array::from_fn(|b| {
                let u = !(b as i32) & 0xFF;
                let (exponent, mantissa) = ((u >> 4) & 7, u & 0xF);
                let magnitude = (((mantissa << 3) + 0x84) << exponent) - 0x84;
                (if u & 0x80 != 0 { -magnitude } else { magnitude }) as i16
            })
        }),
        _ => ALAW.get_or_init(|| {
            std::array::from_fn(|b| {
                let a = b as i32 ^ 0x55;
                let (exponent, mantissa) = ((a >> 4) & 7, a & 0xF);
                let magnitude = if exponent == 0 { (mantissa << 4) + 8 } else { ((mantissa << 4) + 0x108) << (exponent - 1) };
                (if a & 0x80 != 0 { magnitude } else { -magnitude }) as i16
            })
        }),
    }
}

/// np.searchsorted (side left): the first segment whose end is at least `x`.
fn segment(ends: &[i32; 8], x: i32) -> usize {
    ends.iter().position(|&end| end >= x).unwrap_or(8)
}

/// `g711_encode`, one sample.
fn g711_encode(pcm: i16, law: Codec) -> u8 {
    let pcm = pcm as i32;
    let (code, mask) = if law == Codec::Pcmu {
        let x = pcm >> 2;
        let mask = if x < 0 { 0x7F } else { 0xFF };
        let x = x.abs().min(8159) + (0x84 >> 2); // audioop's 14-bit clip
        let seg = segment(&ULAW_SEG_END, x);
        let s = seg.min(7) as i32;
        (if seg >= 8 { 0x7F } else { (s << 4) | ((x >> (s + 1)) & 0xF) }, mask)
    } else {
        let x = pcm >> 3;
        let mask = if x >= 0 { 0xD5 } else { 0x55 };
        let x = if x >= 0 { x } else { -x - 1 };
        let seg = segment(&ALAW_SEG_END, x);
        let s = seg.min(7) as i32;
        (if seg >= 8 { 0x7F } else { (s << 4) | ((x >> (seg as i32).clamp(1, 7)) & 0xF) }, mask)
    };
    ((code ^ mask) & 0xFF) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm16_keeps_an_odd_byte_for_the_next_append() {
        let (a, odd) = decode_append(&json!("AAEC"), None, Codec::Pcm).unwrap(); // 00 01 02
        assert_eq!((a, odd), (vec![256.0 / 32768.0], Some(2)));
        assert!(decode_append(&json!("A!=="), None, Codec::Pcm).is_err());
        assert!(decode_append(&Value::Null, None, Codec::Pcm).is_err());
    }

    #[test]
    fn g711_decodes_and_encodes_as_audioop() {
        // audioop.ulaw2lin / alaw2lin of 0x00, 0x7F, 0x80, 0xFF, and lin2ulaw /
        // lin2alaw of their output, which round-trips
        assert_eq!(g711_table(Codec::Pcmu)[0x00], -32124);
        assert_eq!(g711_table(Codec::Pcmu)[0xFF], 0);
        assert_eq!(g711_table(Codec::Pcmu)[0x80], 32124);
        assert_eq!(g711_table(Codec::Pcma)[0x55], -8);
        assert_eq!(g711_table(Codec::Pcma)[0xD5], 8);
        for law in [Codec::Pcmu, Codec::Pcma] {
            for b in 0..=255u8 {
                let back = g711_encode(g711_table(law)[b as usize], law);
                assert_eq!(g711_table(law)[back as usize], g711_table(law)[b as usize], "{law:?} {b:#x}");
            }
        }
    }

    fn fnv(bytes: impl IntoIterator<Item = u8>) -> u64 {
        bytes.into_iter().fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
    }

    #[test]
    fn g711_matches_media_io_on_every_sample() {
        // FNV-1a of media_io's g711_encode over every int16, and of its decode
        // tables as little-endian int16
        for (law, encode, decode) in [
            (Codec::Pcmu, 0xd5ee_23f7_d0ff_ae91u64, 0xb91e_65d6_8481_a1a8u64),
            (Codec::Pcma, 0xf030_666e_b780_cf25, 0x693a_40ca_62a3_d2a5),
        ] {
            assert_eq!(fnv((i16::MIN..=i16::MAX).map(|s| g711_encode(s, law))), encode, "{law:?}");
            assert_eq!(fnv(g711_table(law).iter().flat_map(|s| s.to_le_bytes())), decode, "{law:?}");
        }
    }
}
