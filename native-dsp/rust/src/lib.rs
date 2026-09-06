//! Safe ownership interface to the shared libsoxr HQ implementation.
//! The supplied library is trusted native code; ABI and contract are checked
//! before handles are created. All unsafe operations are confined here.
use libloading::Library;
use std::{
    ffi::{CStr, c_char, c_void},
    path::Path,
    ptr::NonNull,
};

type Create = unsafe extern "C" fn(u32, u32) -> *mut c_void;
type Process =
    unsafe extern "C" fn(*mut c_void, *const f32, usize, i32, *mut *const f32, *mut usize) -> i32;
type Reset = unsafe extern "C" fn(*mut c_void) -> i32;
type Delay = unsafe extern "C" fn(*const c_void) -> f64;
type Destroy = unsafe extern "C" fn(*mut c_void);
type ErrorFn = unsafe extern "C" fn() -> *const c_char;

/// A mono f32 stream with Python-SoXR HQ delay, flush and reset semantics.
pub struct Resampler {
    handle: NonNull<c_void>,
    process_fn: Process,
    reset_fn: Reset,
    delay_fn: Delay,
    destroy_fn: Destroy,
    error_fn: ErrorFn,
    _library: Library,
}

impl Resampler {
    /// Load the operator-selected trusted MIR DSP library and check its ABI.
    pub fn load(
        path: impl AsRef<Path>,
        source_rate: u32,
        target_rate: u32,
    ) -> Result<Self, String> {
        unsafe {
            // SAFETY: loading executable code is an explicit caller choice. All
            // signatures below match mir_dsp.h and remain live through _library.
            let library = Library::new(path.as_ref()).map_err(|e| e.to_string())?;
            let version = library
                .get::<unsafe extern "C" fn() -> u32>(b"mir_dsp_abi_version\0")
                .map_err(|e| e.to_string())?;
            let contract = library
                .get::<ErrorFn>(b"mir_dsp_resampling_contract\0")
                .map_err(|e| e.to_string())?;
            if version() != 1 || read_string(contract())? != "mir.soxr-hq/mono-f32-v1" {
                return Err("MIR resampler ABI/contract mismatch".into());
            }
            let create = *library
                .get::<Create>(b"mir_resampler_create\0")
                .map_err(|e| e.to_string())?;
            let process_fn = *library
                .get::<Process>(b"mir_resampler_process\0")
                .map_err(|e| e.to_string())?;
            let reset_fn = *library
                .get::<Reset>(b"mir_resampler_reset\0")
                .map_err(|e| e.to_string())?;
            let delay_fn = *library
                .get::<Delay>(b"mir_resampler_delay\0")
                .map_err(|e| e.to_string())?;
            let destroy_fn = *library
                .get::<Destroy>(b"mir_resampler_destroy\0")
                .map_err(|e| e.to_string())?;
            let error_fn = *library
                .get::<ErrorFn>(b"mir_dsp_last_error\0")
                .map_err(|e| e.to_string())?;
            let handle = NonNull::new(create(source_rate, target_rate))
                .ok_or_else(|| read_string(error_fn()).unwrap_or_else(|e| e))?;
            Ok(Self {
                handle,
                process_fn,
                reset_fn,
                delay_fn,
                destroy_fn,
                error_fn,
                _library: library,
            })
        }
    }

    /// Process a chunk, retaining delayed samples. Empty input does not flush.
    pub fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>, String> {
        self.run(samples, false)
    }
    /// Drain the filter; subsequent input requires reset().
    pub fn finish(&mut self) -> Result<Vec<f32>, String> {
        self.run(&[], true)
    }
    fn run(&mut self, samples: &[f32], final_block: bool) -> Result<Vec<f32>, String> {
        let mut output = std::ptr::null();
        let mut count = 0usize;
        unsafe {
            // SAFETY: handle is uniquely owned; input and output locations are
            // live for the call. The native buffer is copied before next use.
            let status = (self.process_fn)(
                self.handle.as_ptr(),
                samples.as_ptr(),
                samples.len(),
                i32::from(final_block),
                &mut output,
                &mut count,
            );
            if status != 0 {
                return Err(read_string((self.error_fn)())?);
            }
            if count == 0 {
                return Ok(Vec::new());
            }
            if output.is_null() || count > 32 * 1024 * 1024 {
                return Err("invalid native output buffer".into());
            }
            Ok(std::slice::from_raw_parts(output, count).to_vec())
        }
    }
    /// Discard buffered samples and restart the stream at time zero.
    pub fn reset(&mut self) -> Result<(), String> {
        unsafe {
            // SAFETY: exclusive handle ownership, library symbols remain live.
            if (self.reset_fn)(self.handle.as_ptr()) != 0 {
                return Err(read_string((self.error_fn)())?);
            }
        }
        Ok(())
    }
    /// Filter latency as the number of output samples buffered internally.
    pub fn delay_samples(&self) -> f64 {
        // SAFETY: shared read of the live handle, no concurrent mutation allowed.
        unsafe { (self.delay_fn)(self.handle.as_ptr()) }
    }
}

impl Drop for Resampler {
    fn drop(&mut self) {
        // SAFETY: exactly one handle owner; library outlives this destructor.
        unsafe { (self.destroy_fn)(self.handle.as_ptr()) }
    }
}
unsafe fn read_string(ptr: *const c_char) -> Result<String, String> {
    if ptr.is_null() {
        return Err("null native error/contract string".into());
    }
    // SAFETY: the checked ABI returns a NUL-terminated string valid for this call.
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map(str::to_owned)
        .map_err(|e| e.to_string())
}
