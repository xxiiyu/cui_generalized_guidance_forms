from comfy_api.latest import ComfyExtension, io
from .nodes_plcfg import PowerLawCFG


class CUIGeneralizedGuidanceForms(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            PowerLawCFG,
        ]


async def comfy_entrypoint() -> ComfyExtension:
    return CUIGeneralizedGuidanceForms()
