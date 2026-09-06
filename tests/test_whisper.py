import os
import pytest

try:
    import torch
    import torch.nn.functional as F
    import whisper
    import types
except ImportError:
    torch = None
    F = None
    whisper = None
    types = None


def test_whisper_extract_features():
    if torch is None or whisper is None:
        pytest.skip("Whisper dependency/checkpoint/test audio unavailable: torch or whisper not installed")

    # Check for available checkpoint
    model_path = os.environ.get("WHISPER_CHECKPOINT_PATH")
    if not model_path or not os.path.exists(model_path):
        pytest.skip("Whisper dependency/checkpoint/test audio unavailable: checkpoint not found")

    audio_path = os.environ.get("WHISPER_TEST_AUDIO")
    if not audio_path or not os.path.exists(audio_path):
        pytest.skip("Whisper dependency/checkpoint/test audio unavailable: test audio not found")

    encoder = whisper.load_model(model_path).encoder
    audio = whisper.load_audio(audio_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mel = whisper.log_mel_spectrogram(audio).to(device)

    def extract_features(self, x: torch.Tensor):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)
        for block in self.blocks:
            x = block(x)
        x = self.ln_post(x)
        return x

    encoder.extract_features = types.MethodType(extract_features, encoder)
    mel = mel.unsqueeze(0)
    encoder_out = encoder.extract_features(mel)
    assert encoder_out is not None