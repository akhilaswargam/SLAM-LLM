import os
import sys
import pytest

# Ensure src/ is in sys.path
_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)


def test_package_import():
    """Verify clean import of slam_llm package without syntax or runtime error."""
    import slam_llm
    assert slam_llm is not None


def test_deepspeed_checkpoint_handler_lazy_import():
    """Verify that checkpoint_handler can be imported without deepspeed installed,
    and calling save_model_checkpoint_deepspeed raises a clear ImportError."""
    try:
        import torch
        import psutil
    except ImportError:
        pytest.skip("torch and/or psutil not installed")

    from slam_llm.utils.checkpoint_handler import (
        get_date_of_run,
        save_model_checkpoint_deepspeed,
    )
    assert callable(get_date_of_run)

    # When deepspeed is not installed, calling save_model_checkpoint_deepspeed raises ImportError
    class DummyModel:
        pass
    class DummyConfig:
        output_dir = "/tmp"

    try:
        import deepspeed
        # If deepspeed is present in test environment, this won't raise ImportError
    except ImportError:
        with pytest.raises(ImportError, match="DeepSpeed is required"):
            save_model_checkpoint_deepspeed(DummyModel(), DummyConfig())


def test_compute_accuracy_zero_tokens():
    """Verify compute_accuracy handles zero valid target tokens gracefully."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch is not installed")

    from slam_llm.utils.metric import compute_accuracy

    # Case 1: All targets are ignore_label (zero denominator)
    outputs = torch.tensor([[10, 20], [30, 40]])
    targets = torch.tensor([[-100, -100], [-100, -100]])
    acc = compute_accuracy(outputs, targets, ignore_label=-100)

    assert not torch.isnan(acc)
    assert not torch.isinf(acc)
    assert acc.item() == 0.0

    # Case 2: Normal computation
    targets2 = torch.tensor([[10, 20], [30, 99]])
    acc2 = compute_accuracy(outputs, targets2, ignore_label=-100)
    assert acc2.item() == 0.75


def test_mutable_defaults_safety():
    """Verify that files with previous mutable defaults have None defaults via AST analysis."""
    import ast

    def get_init_defaults(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=filepath)
        init_defaults = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                # Match default values
                args = node.args
                diff = len(args.args) - len(args.defaults)
                for i, default in enumerate(args.defaults):
                    param_name = args.args[diff + i].arg
                    init_defaults[param_name] = type(default).__name__
        return init_defaults

    # Verify HTSAT_Swin_Transformer
    htsat_path = os.path.join(_src_dir, "slam_llm", "models", "CLAP", "htsat.py")
    defaults = get_init_defaults(htsat_path)
    assert defaults.get("depths") == "Constant"  # None is a Constant in AST
    assert defaults.get("num_heads") == "Constant"

    # Verify MusicFM25Hz
    musicfm_path = os.path.join(_src_dir, "slam_llm", "models", "musicfm", "model", "musicfm_25hz.py")
    defaults = get_init_defaults(musicfm_path)
    assert defaults.get("features") == "Constant"

    # Verify Conv2dSubsampling
    conv_path = os.path.join(_src_dir, "slam_llm", "models", "musicfm", "modules", "conv.py")
    defaults = get_init_defaults(conv_path)
    assert defaults.get("strides") == "Constant"

    # Verify EncodecDecoderLstm
    vallex_path = os.path.join(_src_dir, "slam_llm", "models", "vallex", "vallex_model.py")
    defaults = get_init_defaults(vallex_path)
    assert defaults.get("activation_param") == "Constant"
