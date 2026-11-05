//! Waveform processing that matches the Python OmniVoice package, which uses
//! pydub for silence handling.

const FRAME_MS: usize = 10;
const SILENCE_DB: f64 = -50.0;
const MAX_AMPLITUDE: f64 = 32768.0;

/// 16-bit audio addressed in milliseconds, with pydub's rounding.
struct Segment<'a> {
    samples: &'a [i16],
    sample_rate: usize,
}

impl Segment<'_> {
    fn len_ms(&self) -> usize {
        (1000.0 * self.samples.len() as f64 / self.sample_rate as f64).round_ties_even() as usize
    }

    fn frame(&self, ms: usize) -> usize {
        (ms as f64 * self.sample_rate as f64 / 1000.0) as usize
    }

    /// The samples in a range, and how many samples it was short by because
    /// the length in milliseconds was rounded up.
    fn slice(&self, start: usize, end: usize) -> (&[i16], usize) {
        let len = self.len_ms();
        let (start, end) = (self.frame(start.min(len)), self.frame(end.min(len)));
        let available = end.min(self.samples.len());
        let begin = start.min(available);
        (
            &self.samples[begin..available],
            end - start - (available - begin),
        )
    }

    fn rms(&self, start: usize, end: usize) -> u32 {
        let (samples, missing) = self.slice(start, end);
        let count = samples.len() + missing;
        if count == 0 {
            return 0;
        }
        let sum: f64 = samples
            .iter()
            .map(|sample| f64::from(*sample).powi(2))
            .sum();
        (sum / count as f64).sqrt() as u32
    }

    fn cut(&self, start: usize, end: usize) -> Vec<i16> {
        let (samples, missing) = self.slice(start, end);
        let mut out = samples.to_vec();
        out.resize(samples.len() + missing, 0);
        out
    }
}

fn db_to_amplitude(db: f64) -> f64 {
    10f64.powf(db / 20.0) * MAX_AMPLITUDE
}

fn detect_silence(
    segment: &Segment,
    min_silence_ms: usize,
    threshold_db: f64,
    seek_ms: usize,
) -> Vec<(usize, usize)> {
    let len = segment.len_ms();
    if len < min_silence_ms {
        return Vec::new();
    }
    let threshold = db_to_amplitude(threshold_db);
    let last_start = len - min_silence_ms;
    let mut starts: Vec<usize> = (0..=last_start).step_by(seek_ms).collect();
    if last_start % seek_ms != 0 {
        starts.push(last_start);
    }
    starts.retain(|start| f64::from(segment.rms(*start, start + min_silence_ms)) <= threshold);

    let Some((&first, rest)) = starts.split_first() else {
        return Vec::new();
    };
    let mut ranges = Vec::new();
    let (mut range_start, mut previous) = (first, first);
    for &start in rest {
        let continuous = start == previous + seek_ms;
        if !continuous && start > previous + min_silence_ms {
            ranges.push((range_start, previous + min_silence_ms));
            range_start = start;
        }
        previous = start;
    }
    ranges.push((range_start, previous + min_silence_ms));
    ranges
}

fn detect_nonsilent(
    segment: &Segment,
    min_silence_ms: usize,
    threshold_db: f64,
    seek_ms: usize,
) -> Vec<(usize, usize)> {
    let silent = detect_silence(segment, min_silence_ms, threshold_db, seek_ms);
    let len = segment.len_ms();
    let Some(&(_, last_end)) = silent.last() else {
        return vec![(0, len)];
    };
    if silent[0] == (0, len) {
        return Vec::new();
    }
    let mut ranges = Vec::new();
    let mut previous_end = 0;
    for &(start, end) in &silent {
        ranges.push((previous_end, start));
        previous_end = end;
    }
    if last_end != len {
        ranges.push((previous_end, len));
    }
    if ranges[0] == (0, 0) {
        ranges.remove(0);
    }
    ranges
}

/// Drops long silences, leaving `keep_ms` either side of each piece of sound.
fn split_on_silence(segment: &Segment, min_silence_ms: usize, keep_ms: usize) -> Vec<i16> {
    let mut ranges: Vec<(i64, i64)> =
        detect_nonsilent(segment, min_silence_ms, SILENCE_DB, FRAME_MS)
            .into_iter()
            .map(|(start, end)| (start as i64 - keep_ms as i64, (end + keep_ms) as i64))
            .collect();
    for index in 1..ranges.len() {
        if ranges[index].0 < ranges[index - 1].1 {
            let middle = (ranges[index - 1].1 + ranges[index].0).div_euclid(2);
            ranges[index - 1].1 = middle;
            ranges[index].0 = middle;
        }
    }
    let len = segment.len_ms() as i64;
    let mut out = Vec::new();
    for (start, end) in ranges {
        out.extend(segment.cut(start.max(0) as usize, end.min(len) as usize));
    }
    out
}

fn leading_silence_ms(segment: &Segment) -> usize {
    let len = segment.len_ms();
    let threshold = 10f64.powf(SILENCE_DB / 20.0);
    let mut trim = 0;
    while trim < len {
        let rms = segment.rms(trim, trim + FRAME_MS);
        if rms != 0 && f64::from(rms) / MAX_AMPLITUDE >= threshold {
            break;
        }
        trim += FRAME_MS;
    }
    trim.min(len)
}

fn trim_edges(samples: Vec<i16>, sample_rate: usize, lead_ms: usize, trail_ms: usize) -> Vec<i16> {
    let segment = Segment {
        samples: &samples,
        sample_rate,
    };
    let start = leading_silence_ms(&segment).saturating_sub(lead_ms);
    let mut trimmed = segment.cut(start, segment.len_ms());
    trimmed.reverse();

    let segment = Segment {
        samples: &trimmed,
        sample_rate,
    };
    let start = leading_silence_ms(&segment).saturating_sub(trail_ms);
    let mut out = segment.cut(start, segment.len_ms());
    out.reverse();
    out
}

fn to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|sample| (sample * 32768.0).clamp(-32768.0, 32767.0) as i16)
        .collect()
}

fn to_f32(samples: &[i16]) -> Vec<f32> {
    samples
        .iter()
        .map(|sample| f32::from(*sample) / 32768.0)
        .collect()
}

/// Shortens silences over `mid_ms` and trims the edges to `lead_ms` and `trail_ms`.
pub fn remove_silence(
    samples: &[f32],
    sample_rate: u32,
    mid_ms: usize,
    lead_ms: usize,
    trail_ms: usize,
) -> Vec<f32> {
    let sample_rate = sample_rate as usize;
    let pcm = to_i16(samples);
    let segment = Segment {
        samples: &pcm,
        sample_rate,
    };
    let split = split_on_silence(&segment, mid_ms, mid_ms);
    to_f32(&trim_edges(split, sample_rate, lead_ms, trail_ms))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(milliseconds: usize, level: f32) -> Vec<f32> {
        (0..milliseconds * 24)
            .map(|index| level * (index as f32 * 0.1).sin())
            .collect()
    }

    #[test]
    fn long_pauses_shorten_and_edges_trim() {
        let mut audio = tone(1000, 0.0);
        audio.extend(tone(500, 0.5));
        audio.extend(tone(2000, 0.0));
        audio.extend(tone(500, 0.5));
        audio.extend(tone(1000, 0.0));
        let out = remove_silence(&audio, 24_000, 500, 100, 100);
        let seconds = out.len() as f32 / 24_000.0;
        // Two tones, a one-second gap kept from both sides, and 0.1s at each edge.
        assert!((seconds - 2.2).abs() < 0.03, "{seconds}");
    }

    #[test]
    fn all_silence_empties() {
        assert!(remove_silence(&tone(1000, 0.0), 24_000, 200, 100, 200).is_empty());
    }
}
