from comfy_api.latest import ComfyExtension, io
from .nodes_plcfg import PowerLawCFG
from .nodes_cfgpp import CFGPP


class CUIGeneralizedGuidanceForms(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            PowerLawCFG,
            CFGPP,
        ]


async def comfy_entrypoint() -> ComfyExtension:
    return CUIGeneralizedGuidanceForms()
