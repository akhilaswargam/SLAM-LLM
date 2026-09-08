from pathlib import Path

from omegaconf import OmegaConf

from slam_llm.enhancement.noise_enhancer import NoiseEnhancer
from slam_llm.models.smartslam_router import SmartSLAMRouter
from slam_llm.pipeline.smartslam_dispatcher import SmartSLAMDispatcher


SPEECH_AUDIO = Path("examples/s2s/audio_prompt/en/prompt_1.wav")
NON_SPEECH_AUDIO = Path("examples/s2s/audio_prompt/en/prompt_6.wav")


def make_base_configs():
    model_config = OmegaConf.create({
        "file": None,
        "encoder_name": None,
        "encoder_dim": 1280,
        "encoder_projector": "linear",
        "encoder_projector_ds_rate": 5,
    })

    dataset_config = OmegaConf.create({
        "input_type": "raw",
        "mel_size": 80,
    })

    return model_config, dataset_config


def test_speech_audio_end_to_end_dispatch():
    enhancer = NoiseEnhancer()
    router = SmartSLAMRouter()

    enhanced_audio, enhancement_result = enhancer.enhance(
        str(SPEECH_AUDIO)
    )

    import torch
    routing = router.route(torch.from_numpy(enhanced_audio))

    assert routing["modality"] == "speech"
    assert routing["encoder"] == "whisper"
    assert enhanced_audio.dtype.name == "float32"
    assert enhancement_result.sample_rate > 0

    model_config, dataset_config = make_base_configs()
    dispatcher = SmartSLAMDispatcher(routing)
    result = dispatcher.dispatch(model_config, dataset_config)

    assert result["encoder"] == "whisper"
    assert result["model_config"].encoder_name == "whisper"
    assert result["model_config"].encoder_dim == 1280
    assert result["model_config"].encoder_projector == "linear"
    assert result["model_config"].encoder_projector_ds_rate == 5
    assert result["dataset_config"].input_type == "mel"
    assert result["dataset_config"].mel_size == 128


def test_non_speech_audio_end_to_end_dispatch():
    enhancer = NoiseEnhancer()
    router = SmartSLAMRouter()

    enhanced_audio, enhancement_result = enhancer.enhance(
        str(NON_SPEECH_AUDIO)
    )

    import torch
    routing = router.route(torch.from_numpy(enhanced_audio))

    assert routing["modality"] == "non_speech"
    assert routing["encoder"] == "eat"
    assert enhanced_audio.dtype.name == "float32"
    assert enhancement_result.sample_rate > 0

    model_config, dataset_config = make_base_configs()
    dispatcher = SmartSLAMDispatcher(routing)
    result = dispatcher.dispatch(model_config, dataset_config)

    assert result["encoder"] == "eat"
    assert result["model_config"].encoder_name == "eat"
    assert result["model_config"].encoder_dim == 768
    assert result["model_config"].encoder_projector == "linear"
    assert result["model_config"].encoder_projector_ds_rate == 5
    assert result["dataset_config"].input_type == "mel"
    assert result["dataset_config"].mel_size == 80
    assert result["dataset_config"].model_name == "eat"
    assert result["dataset_config"].fbank_mean == -4.268
    assert result["dataset_config"].fbank_std == 4.569
    assert result["dataset_config"].target_length == 1024


def test_smartslam_pipeline_prepare():
    from slam_llm.pipeline.smartslam_pipeline import SmartSLAMPipeline

    model_config, dataset_config = make_base_configs()
    pipeline = SmartSLAMPipeline(rag_directory="rag_data")

    result = pipeline.prepare(
        SPEECH_AUDIO,
        "How does RAG help SmartSLAM?",
        model_config,
        dataset_config,
        top_k=3,
    )

    try:
        assert Path(result["input_wav"]).resolve() == SPEECH_AUDIO.resolve()
        assert Path(result["enhanced_wav"]).is_file()

        assert result["enhancement"].sample_rate > 0

        assert result["routing"]["modality"] == "speech"
        assert result["routing"]["encoder"] == "whisper"
        assert result["routing"]["confidence"] > 0

        assert result["dispatch"]["encoder"] == "whisper"
        assert result["dispatch"]["model_config"].encoder_name == "whisper"
        assert result["dispatch"]["model_config"].encoder_dim == 1280

        assert len(result["retrieved_results"]) > 0
        assert result["rag_confidence"] > 0
        assert isinstance(result["rag_explanation"], str)
        assert len(result["rag_explanation"]) > 0
        assert isinstance(result["augmented_prompt"], str)
        assert len(result["augmented_prompt"]) > 0
    finally:
        pipeline.cleanup(result)

    assert not Path(result["enhanced_wav"]).exists()
