import typing
from bisect import bisect_left, bisect_right
from inspect import cleandoc
from comfy_api.latest import ComfyExtension, io
import comfy.samplers
import comfy.model_sampling
import torch

if typing.TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher


class CFGPP(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="cfgpp_cui-ggf",
            display_name="CFG++",
            category="advanced/guidance",
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input(
                    "print_debug",
                    tooltip="Print this node's calculations to console at each sampling step.",
                    default=False,
                ),
            ],
            outputs=[io.Model.Output()],
            description=(
                "Implements CFG++ (2406.08070) as a model patch.\n"
                "Disables the speed optimization when `cfg=1`, as the negative term no longer cancels out.\n"
            ),
        )

    @classmethod
    def execute(cls, model: "ModelPatcher", print_debug: bool) -> io.NodeOutput:
        m = model.clone()

        def cfgpp(args):
            cond = args["cond_denoised"]
            uncond = args["uncond_denoised"]
            cond_scale = args["cond_scale"]
            x_t = args["input"]
            sigma = args["sigma"]
            sigmas = args["model_options"]["transformer_options"]["sample_sigmas"]

            i_next = torch.searchsorted(-sigmas, -sigma, side="right")
            sigma_curr = sigmas[i_next - 1]  # compat with multistage samplers
            sigma_next = sigmas[i_next]

            model_type = model.model.model_sampling
            if isinstance(model_type, comfy.model_sampling.CONST):  # assume RF
                scale = sigma_curr * (1 - sigma_next) / (sigma_curr - sigma_next)
            else:  # VE
                scale = sigma_curr / (sigma_curr - sigma_next)
            phi_t = scale * cond_scale

            if print_debug:
                if phi_t.numel() == 1:
                    sigma_display = sigma.item()
                    scale_display = scale.item()
                    phi_t_display = phi_t.item()
                else:
                    sigma_display = sigma.squeeze()
                    scale_display = scale.squeeze()
                    phi_t_display = phi_t.squeeze()
                print(
                    "",
                    f"[cui-ggf] sigma = {sigma_display}",
                    f"[cui-ggf] scale = {scale_display}",
                    f"[cui-ggf] effective_cfg = {phi_t_display}",
                    sep="\n",
                )

            # just in case sigmas are different across batches somehow
            shape = (phi_t.shape[0],) + (1,) * (x_t.ndim - 1)
            return x_t - (uncond + (cond - uncond) * phi_t.view(shape))

        m.set_model_sampler_cfg_function(
            cfgpp,
            disable_cfg1_optimization=True,
        )
        return io.NodeOutput(m)
