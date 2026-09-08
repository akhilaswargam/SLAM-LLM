from pathlib import Path
import tempfile

import soundfile as sf

from slam_llm.enhancement.noise_enhancer import NoiseEnhancer
from slam_llm.models.smartslam_router import SmartSLAMRouter
from slam_llm.pipeline.smartslam_dispatcher import SmartSLAMDispatcher
from slam_llm.rag.rag_pipeline import RAGRetriever


class SmartSLAMPipeline:
    """Orchestrate SmartSLAM preprocessing, routing, dispatch, and RAG."""

    def __init__(self, rag_directory="rag_data"):
        self.enhancer = NoiseEnhancer()
        self.router = SmartSLAMRouter()

        self.rag = RAGRetriever()
        self.rag.load_documents(rag_directory)
        self.rag.build_index()

    def prepare_audio(self, wav_path):
        """Enhance input audio and save the processed waveform as a WAV."""
        enhanced_audio, enhancement_result = self.enhancer.enhance(str(wav_path))

        temp_file = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        )
        temp_file.close()

        enhanced_wav_path = Path(temp_file.name)

        sf.write(
            str(enhanced_wav_path),
            enhanced_audio,
            enhancement_result.sample_rate,
        )

        return enhanced_wav_path, enhancement_result

    def route_audio(self, audio_path):
        """Route audio using the existing SmartSLAM router."""
        return self.router.route(audio_path)

    def retrieve_context(self, prompt, top_k=3):
        """Retrieve RAG evidence and construct the augmented prompt."""
        retrieved_results = self.rag.retrieve(
            prompt,
            top_k=top_k,
        )

        augmented_prompt = self.rag.build_augmented_prompt(
            prompt,
            retrieved_results,
        )

        return retrieved_results, augmented_prompt

    def prepare(self, wav_path, prompt, model_config, dataset_config, top_k=3):
        """
        Run the complete SmartSLAM preparation flow.

        This method deliberately stops before heavyweight SLAM-LLM
        model construction/inference.
        """
        wav_path = Path(wav_path)

        if not wav_path.is_file():
            raise FileNotFoundError(
                f"Audio file not found: {wav_path}"
            )

        enhanced_wav_path, enhancement_result = self.prepare_audio(
            wav_path
        )

        routing = self.route_audio(enhanced_wav_path)

        dispatcher = SmartSLAMDispatcher(routing)

        dispatch_result = dispatcher.dispatch(
            model_config,
            dataset_config,
        )

        retrieved_results, augmented_prompt = self.retrieve_context(
            prompt,
            top_k=top_k,
        )

        return {
            "input_wav": str(wav_path),
            "enhanced_wav": str(enhanced_wav_path),
            "enhancement": enhancement_result,
            "routing": routing,
            "dispatch": dispatch_result,
            "retrieved_results": retrieved_results,
            "augmented_prompt": augmented_prompt,
            "rag_confidence": self.rag.confidence(
                retrieved_results
            ),
            "rag_explanation": self.rag.explain(
                retrieved_results
            ),
        }

    def cleanup(self, result):
        """Remove the temporary enhanced WAV created by prepare()."""
        enhanced_wav = result.get("enhanced_wav")

        if enhanced_wav:
            path = Path(enhanced_wav)

            if path.exists():
                path.unlink()
