from .llama import biLlamaForMaskedLM, biLlamaForMaskedNTP
from .gemma3 import biGemma3ForMaskedLM, biGemma3ForMaskedNTP


MODELS_MAPPING = {
    'llama': {
        'mlm': biLlamaForMaskedLM,
        'mntp': biLlamaForMaskedNTP
    },
    'gemma3': {
        'mlm': biGemma3ForMaskedLM,
        'mntp': biGemma3ForMaskedNTP
    }
}