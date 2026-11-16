//! AMD GPUs through burn's ROCm backend.
//!
//! candle has no AMD backend, so with the `rocm` feature the models that support
//! it run their heaviest part on burn instead (for OmniVoice, the transformer
//! backbone) and the rest on candle's CPU device.

use burn::tensor::{Device, Tensor};
use std::panic::{catch_unwind, AssertUnwindSafe};
use tracing::info;

/// burn's ROCm device `ordinal`, once a tensor has round-tripped through it, so
/// a missing GPU, driver or ROCm install is an error here rather than a panic
/// in the middle of generation.
pub fn rocm_device(ordinal: usize) -> Result<Device, String> {
    prepare_rocm();
    let probe = catch_unwind(AssertUnwindSafe(|| {
        let device = Device::rocm(ordinal);
        let data = Tensor::<1>::zeros([1], &device).try_into_data();
        (device, data)
    }));
    match probe {
        Ok((device, Ok(_))) => Ok(device),
        Ok((_, Err(error))) => Err(format!("ROCm device {ordinal} is not usable: {error:?}")),
        Err(panic) => Err(format!("ROCm device {ordinal} is not usable: {}", panic_message(&*panic))),
    }
}

/// Call before burn first touches the GPU, from anything that runs on burn's
/// ROCm backend (it's idempotent). It turns on CubeCL's cache of compiled
/// kernels, in `~/.cache/cubecl`, keeping any settings from a `cubecl.toml` in
/// the working directory or its parents. Without the cache, every run compiles
/// each kernel again with LLVM, which took 15 s for the first pass at a new
/// length on an RX 6600.
pub fn prepare_rocm() {
    use burn::cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
        config.compilation.cache = true;
        if CubeClRuntimeConfig::try_set(config) {
            info!("CubeCL caches compiled kernels");
        } else {
            info!("CubeCL's configuration was already set; its kernel cache is as that set it");
        }
    });
}

/// Run `f`, turning a burn panic into an error. burn reports device and shape
/// problems by panicking.
pub fn guard<T>(what: &str, f: impl FnOnce() -> T) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|panic| format!("burn failed while {what}: {}", panic_message(&*panic)))
}

fn panic_message(panic: &(dyn std::any::Any + Send)) -> &str {
    panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("unknown panic")
}
