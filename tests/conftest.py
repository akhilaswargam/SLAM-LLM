import os
import sys
import pytest

# Ensure src/ is in sys.path so tests can import slam_llm consistently
_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

try:
    from transformers import LlamaTokenizer
except ImportError:
    LlamaTokenizer = None


@pytest.fixture
def setup_tokenizer():
    def _helper(tokenizer):
        if LlamaTokenizer is None:
            pytest.skip("transformers is not installed")
        # Align with Llama 2 tokenizer
        tokenizer.from_pretrained.return_value = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
        tokenizer.from_pretrained.return_value.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>'})
        tokenizer.from_pretrained.return_value.bos_token_id = 1
        tokenizer.from_pretrained.return_value.eos_token_id = 2

    return _helper

