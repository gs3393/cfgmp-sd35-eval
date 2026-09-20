"""CFG-MP / CFG-MP+ as extra `method`s on the re-evaluation harness's SD3.5-M pipeline.

The sampling loop is a port of the authors' `CFG-MP_SD/utils_SD.py`
(https://github.com/LeonSuZhengYi/CFG-MP, MIT License, Cai, Liu, Su, Wang, ICML 2026).
Changes from the authors' code, none of which alter the update rule:
  * batched (`num_images_per_prompt`) instead of batch size 1;
  * the fine-grained sigma grid is built here instead of swapping the scheduler,
    so one loaded model serves every method;
  * latents come from the same `prepare_latents` + generator as the harness methods,
    so all methods start from identical noise for a given (prompt, seed);
  * transformer forwards are counted and returned (`last_nfe`).

`method="cfgmp"`  -> Picard fixed-point iteration (paper: CFG-MP)
`method="cfgmpp"` -> Anderson-accelerated iteration AA(m, beta) (paper: CFG-MP+)
Any other method is delegated unchanged to the harness pipeline.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from pipelines.sd35.pipeline_stable_diffusion_3_methods import StableDiffusion3MethodsPipeline

CFGMP_METHODS = ("cfgmp", "cfgmpp")


def fine_grained_sigmas(num_inference_steps: int, shift: float = 3.0) -> np.ndarray:
    """Authors' grid: 2N shifted sigmas in [1, 1e-4] plus a trailing 0 (t_curr, t_mid, t_next per step)."""
    sigmas = np.linspace(1.0, 1e-4, num_inference_steps * 2)
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    return np.append(sigmas, 0.0)


def planned_nfe(num_inference_steps: int, max_aa_iter: int, time_threshold: float, shift: float = 3.0) -> int:
    """Transformer forwards per image if no fixed-point iteration stops early (cond and uncond count separately)."""
    sigmas = fine_grained_sigmas(num_inference_steps, shift)
    projected = sum(1 for i in range(num_inference_steps) if sigmas[2 * i] > time_threshold)
    return 2 * num_inference_steps + 2 * max_aa_iter * projected


def solve_anderson_mixing(
    z_history: List[torch.Tensor], f_history: List[torch.Tensor], beta: float = 1.0, ridge: float = 1e-4
) -> torch.Tensor:
    """x_{k+1} = (1-beta) sum_i alpha_i x_{k-i} + beta sum_i alpha_i g(x_{k-i}), alphas from a ridge least squares."""
    m = len(f_history) - 1
    if m < 1:  # not enough history: Picard step
        return z_history[-1] + beta * f_history[-1]

    batch = f_history[0].shape[0]
    device = f_history[0].device
    orig_dtype = f_history[0].dtype

    z_flat = torch.stack([z.reshape(batch, -1).to(torch.float32) for z in z_history], dim=1)
    f_flat = torch.stack([f.reshape(batch, -1).to(torch.float32) for f in f_history], dim=1)
    g_flat = z_flat + f_flat

    f_k = f_flat[:, -1:, :]
    delta_f = f_flat[:, -1:, :] - f_flat[:, :-1, :]
    input_matrix = delta_f.transpose(1, 2)  # [B, D, m]
    target_matrix = f_k.transpose(1, 2)  # [B, D, 1]
    if ridge > 0:
        eye = torch.eye(m, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch, 1, 1) * ridge
        input_matrix = torch.cat([input_matrix, eye], dim=1)
        target_matrix = torch.cat(
            [target_matrix, torch.zeros(batch, m, 1, device=device, dtype=torch.float32)], dim=1
        )

    gamma = torch.linalg.lstsq(input_matrix, target_matrix).solution  # [B, m, 1]
    alphas = torch.cat([gamma.squeeze(-1), 1.0 - gamma.sum(dim=1)], dim=1)  # [B, m+1], sums to 1

    x_avg = torch.bmm(alphas.unsqueeze(1), z_flat).squeeze(1)
    g_avg = torch.bmm(alphas.unsqueeze(1), g_flat).squeeze(1)
    z_next = (1.0 - beta) * x_avg + beta * g_avg
    return z_next.to(orig_dtype).view_as(z_history[0])


class CFGMPMethodsPipeline(StableDiffusion3MethodsPipeline):
    last_nfe: Optional[int] = None

    def _velocity(self, latents, sigma, embeds, pooled) -> torch.Tensor:
        timestep = (sigma * 1000.0).to(device=latents.device, dtype=latents.dtype).expand(latents.shape[0])
        self.last_nfe += 1
        return self.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=embeds,
            pooled_projections=pooled,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False,
        )[0]

    def _fixed_point_step(
        self, z_k, t_curr, t_mid, cond, uncond, z_history, f_history,
        use_aa: bool, aa_window_size: int, aa_damping: float, aa_ridge: float, op_type: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dt_half = (t_mid - t_curr).to(z_k.dtype)
        if op_type == "H":  # cond forward half-step, then uncond backward half-step
            z_temp = z_k + self._velocity(z_k, t_curr, *cond) * dt_half
            g_z_k = z_temp - self._velocity(z_temp, t_curr, *uncond) * dt_half
        else:  # "G": uncond backward half-step, then cond forward half-step
            z_temp = z_k - self._velocity(z_k, t_curr, *uncond) * dt_half
            g_z_k = z_temp + self._velocity(z_temp, t_curr, *cond) * dt_half
        f_k = g_z_k - z_k

        if not use_aa:
            return g_z_k, f_k

        z_history.append(z_k)
        f_history.append(f_k)
        if len(f_history) > aa_window_size + 1:
            z_history.pop(0)
            f_history.pop(0)
        return solve_anderson_mixing(z_history, f_history, beta=aa_damping, ridge=aa_ridge), f_k

    @torch.no_grad()
    def __call__(self, prompt: Union[str, List[str]] = None, method: str = "cfg", **kwargs):
        method = method.lower()
        if method not in CFGMP_METHODS:
            return super().__call__(prompt=prompt, method=method, **kwargs)
        return self._call_cfgmp(prompt, use_aa=(method == "cfgmpp"), **kwargs)

    def _call_cfgmp(
        self,
        prompt,
        use_aa: bool,
        num_inference_steps: int = 10,
        guidance_scale: float = 4.0,
        max_aa_iter: int = 3,
        aa_window_size: int = 1,
        aa_tol: float = 1e-6,
        aa_damping: float = 1.0,
        aa_ridge: float = 1e-4,
        time_threshold: float = 0.6,
        op_switch_time: float = 0.95,
        shift: Optional[float] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_images_per_prompt: int = 1,
        generator=None,
        max_sequence_length: int = 256,
        output_type: str = "pil",
        joint_attention_kwargs: Optional[Dict] = None,
        **_unused,
    ):
        from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        self._guidance_scale = guidance_scale
        self._clip_skip = None
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        self.last_nfe = 0
        device = self._execution_device
        batch_size = 1 if isinstance(prompt, str) else len(prompt)

        p_embeds, n_embeds, p_pooled, n_pooled = self.encode_prompt(
            prompt=prompt, prompt_2=None, prompt_3=None, device=device,
            do_classifier_free_guidance=True, num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        cond, uncond = (p_embeds, p_pooled), (n_embeds, n_pooled)

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt, self.transformer.config.in_channels,
            height, width, p_embeds.dtype, device, generator, None,
        )

        if shift is None:
            shift = float(self.scheduler.config.get("shift", 3.0))
        sigmas = torch.from_numpy(fine_grained_sigmas(num_inference_steps, shift)).to(
            device=device, dtype=torch.float32
        )

        with self.progress_bar(total=num_inference_steps) as bar:
            for i in range(num_inference_steps):
                t_curr, t_mid, t_next = sigmas[2 * i], sigmas[2 * i + 1], sigmas[2 * i + 2]

                # manifold projection phase (early, high-noise steps only)
                if t_curr <= time_threshold:
                    z_star = latents
                else:
                    z_k = latents.clone()
                    z_history, f_history = [], []
                    op_type = "H" if t_curr >= op_switch_time else "G"
                    for _ in range(max_aa_iter):
                        z_k, f_k = self._fixed_point_step(
                            z_k, t_curr, t_mid, cond, uncond, z_history, f_history,
                            use_aa, aa_window_size, aa_damping, aa_ridge, op_type,
                        )
                        if f_k.abs().mean() < aa_tol:
                            break
                    z_star = z_k

                # CFG sampling phase: Euler step from the projected point
                v_cond = self._velocity(z_star, t_curr, *cond)
                v_uncond = self._velocity(z_star, t_curr, *uncond)
                v_final = v_uncond + guidance_scale * (v_cond - v_uncond)
                latents = z_star + v_final * (t_next - t_curr).to(z_star.dtype)
                bar.update()

        if output_type == "latent":
            image = latents
        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        self.maybe_free_model_hooks()
        return StableDiffusion3PipelineOutput(images=image)
