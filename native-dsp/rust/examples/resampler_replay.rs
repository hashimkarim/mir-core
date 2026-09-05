use mir_dsp::Resampler;
use std::{env, fs, path::PathBuf};
fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 7 {
        return Err(
            "usage: replay library input-rate output-rate chunk-samples input.f32 output.f32"
                .into(),
        );
    }
    let mut stream = Resampler::load(
        PathBuf::from(&args[1]),
        args[2].parse::<u32>().map_err(|e| e.to_string())?,
        args[3].parse::<u32>().map_err(|e| e.to_string())?,
    )?;
    let chunk = args[4].parse::<usize>().map_err(|e| e.to_string())?;
    if chunk == 0 {
        return Err("chunk must be positive".into());
    }
    let bytes = fs::read(&args[5]).map_err(|e| e.to_string())?;
    if bytes.len() % 4 != 0 {
        return Err("truncated float audio".into());
    }
    let samples: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let mut output = Vec::new();
    for block in samples.chunks(chunk) {
        output.extend(stream.process(block)?);
    }
    output.extend(stream.finish()?);
    let bytes: Vec<u8> = output.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(&args[6], bytes).map_err(|e| e.to_string())?;
    // Reset must reproduce the entire sequence, and rejected input must not
    // alter the state. These are lifecycle checks, not a reference substitute.
    if stream.process(&[0.0]).is_ok() {
        return Err("accepted input after finish".into());
    }
    stream.reset()?;
    if stream.process(&[f32::NAN]).is_ok() {
        return Err("accepted nonfinite audio".into());
    }
    let mut again = stream.process(&samples)?;
    again.extend(stream.finish()?);
    if output != again {
        return Err("reset/chunk invariance failed".into());
    }
    Ok(())
}
