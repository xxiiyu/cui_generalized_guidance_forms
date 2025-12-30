# CUI-GeneralizedGuidanceForms

Unofficial implementation of '[Classifier-Free Guidance: From High-Dimensional Analysis to Generalized Guidance Forms](https://arxiv.org/abs/2502.07849)' for ComfyUI.

## Installation

- Through [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager): Search for custom node named "`CUI-GeneralizedGuidanceForms`".
- Manually: `git clone https://github.com/xxiiyu/cui_generalized_guidance_forms.git` into `ComfyUI/custom_nodes/`

After installation, restart ComfyUI.

## Features

- **Power Law CFG:** an implementation of the Power-law cfg in the same paper.

Most nodes & parameters should have hover tooltips that you can read for more information.

## Technical Details

This extension implements various "generalized guidance"s. Paraphrasing the paper, that is guidances which take the following form:

$$
x_{t,denoised}=x_{t,neg}+(x_{t,pos}-x_{t,neg})\phi_t(|s_{t,pos}-s_{t,neg}|_2)
$$

where:
- $x_{t,pos}, x_{t,neg}, x_{t,denoised}:$ The model's positive prompt prediction, negative prompt prediction, and the final denoised result respectively, at a specific timestep, in data parameterization $x_0.$
  - $x_{t,pos}, x_{t,neg}$ may sometimes be written as $x_{t,cond}, x_{t,uncond}$
- $s_{t,pos}, s_{t,neg}:$ The model's positive and negative predictions in score parameterization $s,$ namely $\nabla_x \log p(x).$
- $\phi_t(\cdot):$ Any arbitrary function that depends on time and the L2 norm of the differences between the score predictions, namely $|s_{t,pos}-s_{t,neg}|_2,$ satisfying $\lim_{s\to0}[s\phi(s)]=0.$
- The above technically differs from the paper, as the latter bases on $x_{t,pos}$ but comfy opts for basing on $x_{t,neg}.$ I follow comfy's convention in this extension.

### In Relation to Other CFG Modifications

Many other alternate CFG methods can also be expressed through this framework by defining $\phi$ as follows:

| Guidance Method                                          | $\phi_t(\cdot)$ Definition             | Notes |
| :------------------------------------------------------- | :------------------------------------- | :---- |
| CFG                                                      | $\omega$                               |
| Scheduled CFG                                            | $\omega_t$                             |
| [Limited Interval CFG](https://arxiv.org/abs/2404.07724) | $(\omega-1)\cdot\mathbb I_{[t1,t2)}+1$ | *1    |
| [CFG++](https://arxiv.org/abs/2406.08070)                | -                                      | *2    |

**Notes:**
1. $\mathbb I_{[t1, t2)}$ equals 1 if time is between $[t1, t2),$ and 0 otherwise. In essense, Limited Interval CFG turns on CFG only if the timestep is within this interval.
2. While CFG++ says it could be achieved with a CFG schedule (page 10), the proposed schedule doesn't actually make CFG into CFG++.

### On Empirical Effects of Non-Linear CFG

In one of their [talks](https://www.youtube.com/watch?v=94mXzub4JRc&t=2081), they note that enhancements from these generalized guidances seem more pronounced in class-conditioned models (i.e. image with only 1 label like bee or flower but not both at once), and less effective on general text-to-image models (that would be stuff like sd, flux, qwen, etc.).
