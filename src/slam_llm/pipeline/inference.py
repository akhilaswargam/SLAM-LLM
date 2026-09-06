import logging
import random
import torch

from slam_llm.models.slam_model import slam_model
from slam_llm.rag import RAGRetriever
from slam_llm.utils.model_utils import get_custom_model_factory

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf


logger = logging.getLogger(__name__)


@hydra.main(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):

    def to_plain_list(cfg_item):
        if isinstance(cfg_item, ListConfig):
            return OmegaConf.to_container(cfg_item, resolve=True)
        elif isinstance(cfg_item, DictConfig):
            return {
                k: to_plain_list(v)
                for k, v in cfg_item.items()
            }
        else:
            return cfg_item

    kwargs = cfg

    log_level = getattr(
        logging,
        kwargs.get("log_level", "INFO").upper()
    )

    logging.basicConfig(level=log_level)

    if kwargs.get("debug", False):
        import pdb
        pdb.set_trace()

    main(kwargs)


def main(kwargs: DictConfig):

    # Get configuration sections
    train_config = kwargs.train_config
    fsdp_config = kwargs.fsdp_config
    model_config = kwargs.model_config
    log_config = kwargs.log_config
    dataset_config = kwargs.dataset_config

    # Remove configuration sections from kwargs
    del kwargs.train_config
    del kwargs.fsdp_config
    del kwargs.model_config
    del kwargs.log_config
    del kwargs.dataset_config

    # Set seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    # Create model
    model_factory = get_custom_model_factory(
        model_config,
        logger
    )

    model, tokenizer = model_factory(
        train_config,
        model_config,
        **kwargs
    )

    # Select device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Move model to device
    model.to(device)
    model.eval()

    # ============================================================
    # SMARTSLAM RAG INITIALIZATION
    # ============================================================

    rag = RAGRetriever()

    print("\nLoading SmartSLAM RAG knowledge base...")

    rag.load_documents("rag_data")
    rag.build_index()

    print("SmartSLAM RAG initialized successfully.")
    print(f"Loaded {len(rag.chunks)} document chunks.")

    # ============================================================
    # INTERACTIVE INFERENCE
    # ============================================================

    while True:

        print("\n==============================")
        print("       SMARTSLAM INFERENCE")
        print("==============================")

        wav_path = input("Your Wav Path:\n")
        prompt = input("Your Prompt:\n")

        try:

            # ----------------------------------------------------
            # 1. RAG RETRIEVAL
            # ----------------------------------------------------

            retrieved_results = rag.retrieve(
                prompt,
                top_k=3
            )

            # ----------------------------------------------------
            # 2. BUILD AUGMENTED PROMPT
            # ----------------------------------------------------

            augmented_prompt = rag.build_augmented_prompt(
                prompt,
                retrieved_results
            )

            # ----------------------------------------------------
            # 3. GENERATE ANSWER USING SLAM-LLM
            # ----------------------------------------------------

            model_outputs = model.inference(
                wav_path,
                augmented_prompt
            )

            output_text = model.tokenizer.batch_decode(
                model_outputs,
                add_special_tokens=False,
                skip_special_tokens=True
            )

            # ----------------------------------------------------
            # 4. DISPLAY ANSWER
            # ----------------------------------------------------

            print("\n=== SMARTSLAM ANSWER ===")
            print(output_text)

            # ----------------------------------------------------
            # 5. EXPLAINABLE AI - RETRIEVED EVIDENCE
            # ----------------------------------------------------

            print("\n=== RETRIEVED EVIDENCE ===")

            if retrieved_results:

                for result in retrieved_results:

                    print(
                        f"Source: {result['source']} | "
                        f"Chunk: {result['chunk_id']} | "
                        f"Similarity: "
                        f"{result['similarity']:.4f}"
                    )

                    print(
                        f"Evidence: {result['text']}\n"
                    )

            else:

                print(
                    "No relevant evidence retrieved."
                )

            # ----------------------------------------------------
            # 6. CONFIDENCE ESTIMATION
            # ----------------------------------------------------

            confidence = rag.confidence(
                retrieved_results
            )

            print("=== CONFIDENCE ===")
            print(f"{confidence:.4f}")

            # ----------------------------------------------------
            # 7. XAI EXPLANATION
            # ----------------------------------------------------

            print("\n=== XAI EXPLANATION ===")

            print(
                rag.explain(
                    retrieved_results
                )
            )

            # ----------------------------------------------------
            # 8. SHOW AUGMENTED PROMPT
            # ----------------------------------------------------

            print("\n=== AUGMENTED PROMPT ===")
            print(augmented_prompt)

        except Exception as e:

            print(
                f"\nRAG inference error: {e}"
            )

            continue


if __name__ == "__main__":
    main_hydra()