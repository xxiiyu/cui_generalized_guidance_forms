import typing
import math
from inspect import cleandoc
from comfy_api.latest import ComfyExtension, io
import comfy.model_sampling

if typing.TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher

import torch


class PowerLawCFG(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="powerlaw-cfg_cui-ggf",
            display_name="PowerLawCFG",
            category="advanced/guidance",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input(
                    "alpha",
                    tooltip=(
                        "The exponent for the 'power law,' which finds a number to multiply `cfg` by at each step.\n"
                        "Values > 0 are recommended by the paper authors.\n"
                        "  0: disable\n"
                        "  <0: guidance which speeds up convergence to the target at early times\n"
                        "  >0: dampen guidance if positive and negative are similar; amplify guidance if positive and negative are dissimilar.\n"
                    ),
                    default=0.9,
                    min=-0.99,
                    max=100.0,
                    step=0.01,
                    display_mode=io.NumberDisplay.number,
                ),
                io.Combo.Input(
                    "parameterization",
                    ["score", "x0", "eps", "v", "flow"],
                    default="score",
                    tooltip=(
                        "The space in which to calculate the scale. You do NOT have to match this with your model type, they can be different.\n"
                        "In practice, `score` seems to be the most stable.\n"
                        "  score: The default implementation.\n"
                        "  x0: 'Rescaled Power-law CFG' in the paper.\n"
                        "  eps/v/flow: Other parameterizations for completeness's sake.\n"
                    ),
                ),
                io.Boolean.Input(
                    "print_debug",
                    tooltip="Print this node's calculations to console at each sampling step.",
                    default=False,
                ),
            ],
            outputs=[io.Model.Output()],
            description=(
                "Implements 'Power-law CFG' as found in the same paper (2502.07849).\n"
                "Disables the speed optimization when `cfg=1`, as the negative term no longer cancels out.\n"
            ),
        )

    @classmethod
    def execute(
        cls,
        model: "ModelPatcher",
        alpha: float,
        parameterization: str,
        print_debug: bool,
    ) -> io.NodeOutput:
        m = model.clone()

        def power_law_cfg(args):
            cond = args["cond_denoised"]
            uncond = args["uncond_denoised"]
            cond_scale = args["cond_scale"]
            x_t = args["input"]
            sigma = args["sigma"]

            # just in case sigmas are different across batches somehow
            shape = (x_t.shape[0],) + (1,) * (x_t.ndim - 1)
            sigma = sigma.view(shape)

            d = cond - uncond
            is_rf = isinstance(model.model.model_sampling, comfy.model_sampling.CONST)
            if parameterization == "score":
                # s = (alpha_t / sigma_t^2) * x0
                # add 1 to denom as sigma approaches 0 to avoid numerical errors.
                if is_rf:
                    d *= (1 - sigma) / (sigma**2 + 1)
                else:  # VE
                    d *= 1 / (sigma**2 + 1)
            elif parameterization in ["eps", "v", "flow"]:
                # eps = (x_t - alpha_t*x0) / sigma_t
                # v = alpha_t*eps - sigma_t*x0
                # u = (x_t - x_0) / sigma_t
                d *= 1 / (sigma + 1)
            else:  # x0
                pass

            l2 = torch.norm(d, p=2, dim=list(range(1, d.ndim)), keepdim=True)
            scale = torch.pow(l2 + 1e-6, alpha)
            # pow_scale = pow_scale.clamp(0.1, 2)  # prevent crazy scales
            phi_t = (scale * cond_scale).clamp(min=1.0)

            if print_debug:
                if phi_t.numel() == 1:
                    l2_display = l2.item()
                    scale_display = scale.item()
                    phi_t_display = phi_t.item()
                else:
                    l2_display = l2.squeeze()
                    scale_display = scale.squeeze()
                    phi_t_display = phi_t.squeeze()
                print(
                    "",
                    f"[cui-ggf] param = {parameterization}",
                    f"[cui-ggf] l2 = {l2_display}",
                    f"[cui-ggf] scale = {scale_display}",
                    f"[cui-ggf] effective_cfg = {phi_t_display}",
                    sep="\n",
                )

            return x_t - (uncond + (cond - uncond) * phi_t)

        m.set_model_sampler_cfg_function(
            power_law_cfg,
            disable_cfg1_optimization=math.isclose(alpha, 0.0),
        )
        return io.NodeOutput(m)
