//! Matrix multiplication through AMD's hipBLAS, for burn's ROCm backend.
//!
//! burn's own matmul kernels (cubek's register-tiled kernel, on GPUs without
//! matrix cores such as RDNA2) reach about 3 to 4 TFLOP/s on an RX 6600 at
//! OmniVoice's shapes; hipBLAS reaches 6 to 8 in F32 and 9 to 12 with F16
//! inputs, at the same shapes. [`matmul`] runs a matmul through hipBLAS when it
//! can and through burn when it can't (not ROCm, an unsupported layout or
//! dtype, hipBLAS missing, or `ANY_TTS_HIPBLAS=0`), with the same result
//! either way up to rounding.
//!
//! The call is a burn backend extension, so it takes its place in burn's
//! queue of lazily fused operations, and hipBLAS queues its work on the same
//! HIP stream as burn's kernels (which a patched cubecl-hip hands out with a
//! buffer's address), so it runs in order with them with no waiting on
//! either side. An earlier version ran hipBLAS on HIP's default stream with a
//! barrier on each side; the GPU then sat idle for a third of each step while
//! the CPU caught up.
//!
//! rocBLAS, under hipBLAS, ships kernels for gfx1030 but not the rest of
//! RDNA2 (gfx1031 to gfx1036, such as the RX 6600's gfx1032), which run them
//! with `HSA_OVERRIDE_GFX_VERSION=10.3.0`; see [`apply_gfx_override`].

use burn::backend::tensor::FloatTensor;
use burn::backend::{backend_extension, Backend};
use burn::tensor::{DType, Tensor};
use burn_cubecl::cubecl::hip::compute::HipServer;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::CubeBackend;
use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

/// `lhs` · `rhs`, as [`Tensor::matmul`], through hipBLAS when it can.
pub fn matmul<const D: usize>(lhs: Tensor<D>, rhs: Tensor<D>) -> Tensor<D> {
    if !enabled() {
        return lhs.matmul(rhs);
    }
    let out = <burn::backend::Dispatch as BlasMatmul>::blas_matmul(lhs.into_dispatch(), rhs.into_dispatch());
    Tensor::from_dispatch(out)
}

/// Whether [`matmul`] tries hipBLAS: unless `ANY_TTS_HIPBLAS` is `0`, and
/// only if the library loads.
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ANY_TTS_HIPBLAS").map_or(true, |v| v != "0") && library().is_some())
}

/// Every GPU's `gfx_target_version` (gfx1032 is 100302), from the kernel's
/// topology, without starting the HIP runtime.
fn gfx_target_versions() -> Vec<u32> {
    let Ok(nodes) = std::fs::read_dir("/sys/class/kfd/kfd/topology/nodes") else {
        return Vec::new();
    };
    nodes
        .flatten()
        .filter_map(|node| std::fs::read_to_string(node.path().join("properties")).ok())
        .filter_map(|properties| {
            properties
                .lines()
                .find_map(|line| line.strip_prefix("gfx_target_version "))
                .and_then(|v| v.trim().parse().ok())
        })
        .filter(|&v| v != 0) // CPU nodes
        .collect()
}

/// Sets `HSA_OVERRIDE_GFX_VERSION=10.3.0` when hipBLAS will be used and every
/// GPU is an RDNA2 chip that rocBLAS has no kernels for but that runs
/// gfx1030's (gfx1031 to gfx1036; the instruction set is the same), and the
/// variable isn't already set. Returns a line saying so when it sets it.
///
/// HIP reads the variable once, when it starts, so call this first thing in
/// `main`, before anything touches the GPU and before other threads start.
/// It changes what HIP reports for the GPU, so burn compiles and tunes its
/// own kernels for gfx1030 too (once, into its cache).
///
/// # Safety
///
/// Sets an environment variable, so no other thread may be running.
pub unsafe fn apply_gfx_override() -> Option<String> {
    if std::env::var_os("HSA_OVERRIDE_GFX_VERSION").is_some()
        || std::env::var("ANY_TTS_HIPBLAS").is_ok_and(|v| v == "0")
    {
        return None;
    }
    let versions = gfx_target_versions();
    let needs = |v: u32| (100301..=100306).contains(&v);
    if versions.is_empty() || !versions.iter().all(|&v| needs(v)) {
        return None;
    }
    // Set before loading hipBLAS, whose `hipblasCreate` starts HIP. If hipBLAS
    // doesn't load, nothing has started HIP yet (no HIP call is made unless
    // both libraries load), so the variable can be taken back.
    unsafe { std::env::set_var("HSA_OVERRIDE_GFX_VERSION", "10.3.0") };
    if library().is_none() {
        unsafe { std::env::remove_var("HSA_OVERRIDE_GFX_VERSION") };
        return None;
    }
    let chips: Vec<String> = versions.iter().map(|v| format!("gfx{}{}{:x}", v / 10000, v / 100 % 100, v % 100)).collect();
    Some(format!(
        "GPU {}: using gfx1030's rocBLAS kernels (HSA_OVERRIDE_GFX_VERSION=10.3.0)",
        chips.join(", ")
    ))
}

#[backend_extension(Cube, Fusion)]
pub trait BlasMatmul: Backend {
    /// lhs (…, m, k) · rhs (…, k, n), as `float_matmul`: batch dimensions of
    /// size 1 broadcast.
    #[fusion(dtype = lhs, shape = {
        let mut shape = lhs.clone();
        let d = shape.num_dims();
        for i in 0..d - 2 {
            shape[i] = shape[i].max(rhs[i]);
        }
        shape[d - 1] = rhs[d - 1];
        shape
    })]
    fn blas_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self>;
}

impl BlasMatmul for CubeBackend {
    fn blas_matmul(lhs: CubeTensor, rhs: CubeTensor) -> CubeTensor {
        match hip_matmul(&lhs, &rhs) {
            Some(out) => out,
            None => <CubeBackend as burn::backend::ops::FloatTensorOps<CubeBackend>>::float_matmul(lhs, rhs),
        }
    }
}

// hipBLAS's C API, loaded at run time so the library is optional.
type Handle = *mut c_void;
const OP_N: c_int = 111;
const OP_T: c_int = 112;
const R_32F: c_int = 0;
const R_16F: c_int = 2;
const COMPUTE_32F: c_int = 2;
const GEMM_DEFAULT: c_int = 160;

type GemmStridedBatchedEx = unsafe extern "C" fn(
    Handle, c_int, c_int, c_int, c_int, c_int,
    *const c_void, *const c_void, c_int, c_int, i64,
    *const c_void, c_int, c_int, i64,
    *const c_void, *mut c_void, c_int, c_int, i64,
    c_int, c_int, c_int,
) -> c_int;

struct Library {
    handle: Handle,
    gemm: GemmStridedBatchedEx,
    set_stream: unsafe extern "C" fn(Handle, *mut c_void) -> c_int,
    _libraries: (libloading::Library, libloading::Library),
}

// The handle is used from whichever thread runs burn's queue, one call at a
// time (see `CALL`).
unsafe impl Send for Library {}
unsafe impl Sync for Library {}

static CALL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// How many matmuls ran through hipBLAS, for tests.
static RAN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn library() -> Option<&'static Library> {
    static LIBRARY: OnceLock<Option<Library>> = OnceLock::new();
    LIBRARY
        .get_or_init(|| match unsafe { load() } {
            Ok(library) => Some(library),
            Err(error) => {
                tracing::info!("hipBLAS isn't available, so burn's matmul is used: {error}");
                None
            }
        })
        .as_ref()
}

unsafe fn load() -> Result<Library, String> {
    let open = |names: &[&str]| -> Result<libloading::Library, String> {
        let mut errors = Vec::new();
        for name in names {
            match unsafe { libloading::Library::new(name) } {
                Ok(library) => return Ok(library),
                Err(error) => errors.push(error.to_string()),
            }
        }
        Err(errors.join("; "))
    };
    // The same runtime burn's HIP backend loads (by these names, through the
    // linker's search path), so both share one HIP runtime.
    let hip = open(&["libamdhip64.so", "libamdhip64.so.7"])?;
    let blas = open(&["libhipblas.so", "libhipblas.so.3"])?;
    unsafe {
        let create: libloading::Symbol<unsafe extern "C" fn(*mut Handle) -> c_int> =
            blas.get(b"hipblasCreate\0").map_err(|e| e.to_string())?;
        let gemm: libloading::Symbol<GemmStridedBatchedEx> =
            blas.get(b"hipblasGemmStridedBatchedEx\0").map_err(|e| e.to_string())?;
        let set_stream: libloading::Symbol<unsafe extern "C" fn(Handle, *mut c_void) -> c_int> =
            blas.get(b"hipblasSetStream\0").map_err(|e| e.to_string())?;
        let mut handle: Handle = std::ptr::null_mut();
        let status = create(&mut handle);
        if status != 0 {
            return Err(format!("hipblasCreate failed with status {status}"));
        }
        Ok(Library {
            handle,
            gemm: *gemm,
            set_stream: *set_stream,
            _libraries: (hip, blas),
        })
    }
}

/// How one operand's (rows, cols) matrices lie in memory, as BLAS takes them.
struct Operand {
    /// Whether consecutive columns are adjacent (row-major).
    row_major: bool,
    /// The stride between rows (row-major) or columns.
    ld: usize,
    /// The stride between matrices in the batch, 0 to broadcast one.
    batch_stride: usize,
}

/// `t`'s last two dimensions as a BLAS matrix, and its batch dimensions as
/// one stride, if they allow it.
fn operand(t: &CubeTensor, batch: usize) -> Option<Operand> {
    if t.meta.is_tiled() {
        return None;
    }
    let shape = t.meta.shape();
    let strides = t.meta.strides();
    let d = shape.num_dims();
    let (rows, cols) = (shape[d - 2], shape[d - 1]);
    let (rs, cs) = (strides[d - 2], strides[d - 1]);
    let (row_major, ld) = if cs == 1 || cols == 1 {
        (true, rs.max(cols))
    } else if rs == 1 || rows == 1 {
        (false, cs.max(rows))
    } else {
        return None;
    };
    // The batch dimensions must collapse to one stride (or be broadcast).
    let mut batch_stride = None;
    let mut expected = 0usize;
    for i in (0..d - 2).rev() {
        if shape[i] == 1 {
            continue;
        }
        match batch_stride {
            None => {
                batch_stride = Some(strides[i]);
                expected = strides[i] * shape[i];
            }
            Some(_) if strides[i] == expected => expected *= shape[i],
            Some(_) => return None,
        }
    }
    let batch_stride = batch_stride.unwrap_or(0);
    let count: usize = (0..d - 2).map(|i| shape[i]).product();
    if count != batch && count != 1 {
        return None;
    }
    Some(Operand {
        row_major,
        ld,
        batch_stride: if count == 1 { 0 } else { batch_stride },
    })
}

fn hip_matmul(lhs: &CubeTensor, rhs: &CubeTensor) -> Option<CubeTensor> {
    let library = library()?;
    let type_ = match (lhs.dtype, rhs.dtype) {
        (DType::F32, DType::F32) => R_32F,
        (DType::F16, DType::F16) => R_16F,
        _ => return None,
    };
    let (ls, rs) = (lhs.meta.shape(), rhs.meta.shape());
    let d = ls.num_dims();
    if d < 2 || rs.num_dims() != d || ls[d - 1] != rs[d - 2] {
        return None;
    }
    let (m, k, n) = (ls[d - 2], ls[d - 1], rs[d - 1]);
    // The output's batch dimensions: lhs's, with rhs broadcast across them.
    let mut out_shape = ls.clone();
    out_shape[d - 1] = n;
    for i in 0..d - 2 {
        if ls[i] != rs[i] && rs[i] != 1 {
            return None;
        }
    }
    let batch: usize = (0..d - 2).map(|i| ls[i]).product();
    let a = operand(lhs, batch)?;
    let b = operand(rhs, batch)?;
    if [m, n, k, batch, a.ld, b.ld].iter().any(|&v| v > i32::MAX as usize) || m * n * k == 0 {
        return None;
    }

    let out = burn_cubecl::ops::numeric::empty_device_contiguous_dtype(
        lhs.client.clone(),
        lhs.device.clone(),
        out_shape,
        lhs.dtype,
    );
    let resource = |t: &CubeTensor| t.client.get_resource::<HipServer>(t.handle.clone()).ok();
    let (lr, rr, or) = (resource(lhs)?, resource(rhs)?, resource(&out)?);
    let stream = or.resource().stream;
    if stream.is_null() || lr.resource().stream != stream || rr.resource().stream != stream {
        return None;
    }

    // Row-major C (m, n) is column-major Cᵀ (n, m) = Bᵀ Aᵀ: BLAS's A is rhs
    // and its B is lhs. A row-major operand is already the transpose BLAS
    // wants; a column-major one needs transposing.
    let op = |o: &Operand| if o.row_major { OP_N } else { OP_T };
    let (alpha, beta) = (1.0f32, 0.0f32);
    let _guard = CALL.lock().ok()?;
    if unsafe { (library.set_stream)(library.handle, stream.cast()) } != 0 {
        return None;
    }
    let status = unsafe {
        (library.gemm)(
            library.handle,
            op(&b),
            op(&a),
            n as c_int,
            m as c_int,
            k as c_int,
            (&alpha as *const f32).cast(),
            rr.resource().ptr as *const c_void,
            type_,
            b.ld as c_int,
            b.batch_stride as i64,
            lr.resource().ptr as *const c_void,
            type_,
            a.ld as c_int,
            a.batch_stride as i64,
            (&beta as *const f32).cast(),
            or.resource().ptr as *mut c_void,
            type_,
            n as c_int,
            (m * n) as i64,
            batch as c_int,
            COMPUTE_32F,
            GEMM_DEFAULT,
        )
    };
    if status != 0 {
        tracing::warn!("hipBLAS matmul failed (status {status}); using burn's");
        return None;
    }
    drop((lr, rr, or));
    RAN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    /// hipBLAS against burn's matmul for the layouts OmniVoice's backbone
    /// uses: a row-major input times a transposed (column-major) weight,
    /// broadcast over the batch; batched attention with a transposed right
    /// side; and both in F16. Needs a ROCm GPU and hipBLAS (and on gfx1031 to
    /// gfx1036, `HSA_OVERRIDE_GFX_VERSION=10.3.0`); skipped without them.
    #[test]
    fn matches_burns_matmul() {
        let Ok(device) = crate::burn_device::rocm_device(0) else {
            eprintln!("no ROCm device; skipping");
            return;
        };
        if !enabled() {
            eprintln!("no hipBLAS; skipping");
            return;
        }
        let values = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|i| (((i + seed) * 2654435761) % 1000) as f32 / 500.0 - 1.0).collect()
        };
        let t = |shape: [usize; 4], seed: usize| {
            Tensor::<4>::from_data(TensorData::new(values(shape.iter().product(), seed), shape), &device)
        };
        let check = |name: &str, got: Tensor<4>, want: Tensor<4>, tolerance: f32| {
            assert_eq!(got.dims(), want.dims(), "{name}");
            let got: Vec<f32> = got.cast(DType::F32).into_data().try_to_vec().unwrap();
            let want: Vec<f32> = want.cast(DType::F32).into_data().try_to_vec().unwrap();
            let diff = got.iter().zip(&want).fold(0.0f32, |d, (a, b)| d.max((a - b).abs()));
            assert!(diff < tolerance, "{name}: max |diff| {diff}");
        };
        let ran = || RAN.load(std::sync::atomic::Ordering::Relaxed);
        // A projection: (1, 1, 37, 64) · (1, 1, 64, 48), the weight stored (48, 64).
        let x = t([1, 1, 37, 64], 1);
        let w = t([1, 1, 48, 64], 2).swap_dims(2, 3);
        // Other tests may use hipBLAS at the same time, so each check only
        // asserts that the count went up.
        let mut before = ran();
        let mut went_up = |what: &str| {
            let now = ran();
            assert!(now > before, "{what} didn't run through hipBLAS");
            before = now;
        };
        check("projection", matmul(x.clone(), w.clone()), x.clone().matmul(w.clone()), 1e-4);
        went_up("the projection");
        // Batched attention: (2, 3, 40, 16) · (2, 3, 40, 16)ᵀ, then · (2, 3, 40, 20).
        let q = t([2, 3, 40, 16], 3);
        let k = t([2, 3, 40, 16], 4);
        let v = t([2, 3, 40, 20], 5);
        check("scores", matmul(q.clone(), k.clone().swap_dims(2, 3)), q.clone().matmul(k.clone().swap_dims(2, 3)), 1e-4);
        let scores = q.clone().matmul(k.swap_dims(2, 3));
        check("weighted", matmul(scores.clone(), v.clone()), scores.matmul(v), 1e-3);
        went_up("attention");
        // F16 inputs.
        let half = |t: Tensor<4>| t.cast(DType::F16);
        check("projection f16", matmul(half(x.clone()), half(w.clone())), half(x).matmul(half(w)), 2e-2);
        went_up("the F16 projection");
        // A non-contiguous batch (heads sliced) falls back to burn and still matches.
        let q2 = q.clone().narrow(1, 1, 2);
        let k2 = t([2, 3, 40, 16], 8).narrow(1, 0, 2);
        check("sliced", matmul(q2.clone(), k2.clone().swap_dims(2, 3)), q2.matmul(k2.swap_dims(2, 3)), 1e-4);
    }
}
