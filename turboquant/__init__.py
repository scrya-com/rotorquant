from .turboquant import TurboQuantMSE, TurboQuantProd, TurboQuantKVCache
from .lloyd_max import LloydMaxCodebook, solve_lloyd_max
from .compressors import TurboQuantCompressorV2, TurboQuantCompressorMSE
from .cuda_backend import is_cuda_available, QJLSketch, QJLKeyQuantizer
from .isoquant import IsoQuantMSE, IsoQuantProd
from .planarquant import PlanarQuantMSE, PlanarQuantProd
from .rotorquant import RotorQuantMSE, RotorQuantProd, RotorQuantKVCache
from .literatiquant import (
    LiteratiQuantMSE, LiteratiQuantRotated, LiteratiQuantLinear,
    LiteratiQuantEmbedding, LiteratiQuantKVCache,
    literati_replace, export_literati_to_gguf_tensors,
)
from .clifford import geometric_product, make_random_rotor, rotor_sandwich

# IsoQuant is the recommended default (5.8x faster, same quality)
QuantMSE = IsoQuantMSE
QuantProd = IsoQuantProd

# Triton kernels (optional, requires triton >= 3.0 and an active GPU driver)
try:
    from .triton_planarquant import (
        triton_planar2_fused,
        triton_planar2_quantize,
        triton_planar2_dequantize,
    )
    from .fused_planar_attention import (
        triton_fused_planar_quantize_attend,
        triton_planar_cached_attention,
        pre_rotate_query_planar,
        PlanarQuantCompressedCache,
    )
    from .triton_isoquant import (
        triton_iso_full_fused,
        triton_iso_fast_fused,
    )
    from .triton_literatiquant import (
        triton_literati_fused,
        triton_literati_quantize,
        triton_literati_dequantize,
    )
    from .triton_kernels import (
        triton_rotor_sandwich,
        triton_rotor_full_fused,
        triton_rotor_inverse_sandwich,
        triton_fused_attention,
        pack_rotors_for_triton,
    )
    _triton_available = True
except (ImportError, RuntimeError):
    _triton_available = False
