//! SiLU-gated MLP (feed-forward network).
//!
//! The standard Qwen / Llama MLP: gate_proj and up_proj are applied in
//! parallel, gated with SiLU, then fused through down_proj.
//!
//! ```text
//! output = down_proj( silu(gate_proj(x)) * up_proj(x) )
//! ```

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::tensor_utils::{silu, KernelMode};

/// SiLU-gated feed-forward network.
pub struct SiluMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    kernels: KernelMode,
}

impl SiluMlp {
    /// Load MLP weights from a VarBuilder.
    ///
    /// Expected weight names:
    /// - `gate_proj.weight` — (intermediate_size, hidden_size)
    /// - `up_proj.weight`   — (intermediate_size, hidden_size)
    /// - `down_proj.weight` — (hidden_size, intermediate_size)
    pub fn load(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj =
            candle_nn::linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = candle_nn::linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj =
            candle_nn::linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            kernels: KernelMode::Composite,
        })
    }

    /// Choose the kernels [`Self::forward`] runs. [`KernelMode::Fused`] runs
    /// SiLU as one kernel rather than a sigmoid and a product.
    pub fn with_kernels(mut self, kernels: KernelMode) -> Self {
        self.kernels = kernels;
        self
    }

    /// Forward pass: `down_proj( silu(gate_proj(x)) * up_proj(x) )`
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let gate = match self.kernels {
            KernelMode::Composite => silu(&gate)?,
            KernelMode::Fused => gate.silu()?,
        };
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

impl std::fmt::Debug for SiluMlp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SiluMlp").finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::collections::HashMap;

    #[test]
    fn composite_mlp_runs_sigmoid_silu_and_fused_matches_it() -> Result<()> {
        let device = Device::Cpu;
        let mut weights = HashMap::new();
        for (index, (name, shape)) in [
            ("gate_proj.weight", (12, 8)),
            ("up_proj.weight", (12, 8)),
            ("down_proj.weight", (8, 12)),
        ]
        .into_iter()
        .enumerate()
        {
            let values: Vec<f32> = (0..96)
                .map(|i| ((i + 5 * index) as f32 * 0.21).sin())
                .collect();
            weights.insert(name.to_string(), Tensor::from_vec(values, shape, &device)?);
        }
        let x = Tensor::arange(0f32, 24., &device)?
            .reshape((3, 8))?
            .affine(0.2, -2.)?;
        let product = |name: &str, x: &Tensor| x.matmul(&weights[name].t()?);
        let expected = product(
            "down_proj.weight",
            &(silu(&product("gate_proj.weight", &x)?)? * product("up_proj.weight", &x)?)?,
        )?;
        let vb = VarBuilder::from_tensors(weights.clone(), DType::F32, &device);
        let composite = SiluMlp::load(8, 12, vb.clone())?.forward(&x)?;
        assert_eq!(
            composite.flatten_all()?.to_vec1::<f32>()?,
            expected.flatten_all()?.to_vec1::<f32>()?
        );
        let fused = SiluMlp::load(8, 12, vb)?
            .with_kernels(KernelMode::Fused)
            .forward(&x)?;
        let difference = (fused - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(difference < 1e-5, "fused SiLU error {difference}");
        Ok(())
    }
}
