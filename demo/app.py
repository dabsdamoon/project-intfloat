#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio Demo App for comparing Original vs Finetuned Embeddings
"""

import gradio as gr
from pathlib import Path
import sys
from typing import Tuple, Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.retriever import RAGRetriever
from utils.evaluator import evaluate_models


class RAGDemo:
    """Gradio demo for RAG comparison"""

    def __init__(
        self,
        db_path: str = "./chroma_db",
        finetuned_model_path: str = "./logs/tensorboard/run_20251103_083449/model",
        original_collection: str = "original_embeddings",
        finetuned_collection: str = "finetuned_embeddings",
        data_type: str = "KorQuAD"
    ):
        """
        Initialize demo

        Args:
            db_path: Path to ChromaDB
            finetuned_model_path: Path to finetuned model
            original_collection: Name of original embeddings collection
            finetuned_collection: Name of finetuned embeddings collection
            data_type: Type of data being searched ("KorQuAD" or "Wiki")
        """
        print("Initializing RAG Demo...")
        print("This may take a moment to load models...")

        self.data_type = data_type
        self.retriever = RAGRetriever(
            db_path=db_path,
            finetuned_model_path=finetuned_model_path,
            original_collection_name=original_collection,
            finetuned_collection_name=finetuned_collection
        )

        # Store current search results for evaluation
        self.current_query = ""
        self.current_original_results = []
        self.current_finetuned_results = []

        print("✅ Demo initialized successfully!")

    def search(
        self,
        query: str,
        top_k: int,
        model_type: str
    ) -> Tuple[str, str, str]:
        """
        Perform search and return formatted results

        Args:
            query: Query text
            top_k: Number of results
            model_type: "Original", "Finetuned", or "Both"

        Returns:
            Tuple of (original_results, finetuned_results, evaluate_button_visibility) as formatted strings
        """
        if not query.strip():
            return "⚠️ Please enter a query", "⚠️ Please enter a query", gr.update(visible=False)

        # Convert model type to lowercase for API
        model_type_lower = model_type.lower()

        # Perform search
        results = self.retriever.search(
            query=query,
            top_k=top_k,
            model_type=model_type_lower
        )

        # Store results for evaluation
        self.current_query = query
        self.current_original_results = results.get("original", [])
        self.current_finetuned_results = results.get("finetuned", [])

        # Format results
        original_text = self._format_results(
            results.get("original", []),
            "Original Model"
        ) if model_type_lower in ["original", "both"] else "ℹ️ Original model not selected"

        finetuned_text = self._format_results(
            results.get("finetuned", []),
            "Finetuned Model"
        ) if model_type_lower in ["finetuned", "both"] else "ℹ️ Finetuned model not selected"

        # Show evaluate button only when "Both" is selected
        evaluate_visible = model_type_lower == "both"

        return original_text, finetuned_text, gr.update(visible=evaluate_visible)

    def _format_results(self, results: list, title: str) -> str:
        """Format results as markdown string"""
        if not results:
            return f"### {title}\n\nNo results found."

        output = f"### {title}\n\n"

        for r in results:
            output += f"#### 🔍 Rank {r['rank']} (Score: {r['score']:.4f})\n\n"

            # Use different labels based on data type
            if self.data_type == "Wiki":
                output += f"**Section:** {r['query']}\n\n"
                output += f"**Content:** {r['answer']}\n\n"
            else:  # KorQuAD or default
                output += f"**Query:** {r['query']}\n\n"
                output += f"**Answer:** {r['answer']}\n\n"

            output += "---\n\n"

        return output

    def evaluate(self, progress=gr.Progress()) -> str:
        """
        Evaluate retrieval quality using OpenAI embeddings with progress tracking.

        Args:
            progress: Gradio Progress tracker

        Returns:
            Formatted evaluation results as markdown
        """
        if not self.current_query:
            return "### ⚠️ No Search Results\n\nPlease perform a search first before evaluating."

        if not self.current_original_results or not self.current_finetuned_results:
            return "### ⚠️ Incomplete Results\n\nBoth models must have results for evaluation. Please search with 'Both' selected."

        try:
            # Create progress callback
            def progress_callback(value, desc):
                progress(value, desc=desc)

            progress(0, desc="Starting evaluation...")
            return evaluate_models(
                self.current_query,
                self.current_original_results,
                self.current_finetuned_results,
                progress_callback
            )
        except Exception as e:
            return f"### ❌ Evaluation Error\n\n{str(e)}\n\nPlease check your OPENAI_API_KEY in .env file."

    def launch(self, share: bool = False):
        """Launch Gradio interface"""

        # Define Gradio interface
        with gr.Blocks(title="RAG Embedding Comparison", theme=gr.themes.Soft()) as demo:
            gr.Markdown(
                """
                # 🔬 RAG Embedding Model Comparison

                Compare retrieval results between **Original** and **Finetuned** embedding models.

                - **Original Model**: `intfloat/multilingual-e5-small`
                - **Finetuned Model**: Fine-tuned on KorQuAD with LoRA
                - **Dataset**: 나무위키 - 볼드모트 (Harry Potter Wiki)

                Enter a query in Korean to retrieve relevant content from the wiki document.
                """
            )

            with gr.Row():
                with gr.Column():
                    query_input = gr.Textbox(
                        label="🔍 Enter Your Query",
                        placeholder="예: 볼드모트는 누구인가?",
                        lines=2
                    )

                    with gr.Row():
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Top K Results"
                        )

                        model_type_radio = gr.Radio(
                            choices=["Both", "Original", "Finetuned"],
                            value="Both",
                            label="Model Selection"
                        )

                    search_button = gr.Button("🚀 Search", variant="primary")

            gr.Markdown("## 📊 Results")

            with gr.Row():
                with gr.Column():
                    original_output = gr.Markdown(label="Original Model Results")

                with gr.Column():
                    finetuned_output = gr.Markdown(label="Finetuned Model Results")

            # Evaluate button (only visible when Both is selected)
            evaluate_button = gr.Button(
                "⚖️ Evaluate with OpenAI Embeddings",
                variant="secondary",
                visible=False
            )

            # Evaluation results
            evaluation_output = gr.Markdown(label="Evaluation Results", visible=True)

            # Example queries
            gr.Markdown("### 💡 Example Queries")
            gr.Examples(
                examples=[
                    ["볼드모트는 누구인가?", 5, "Both"],
                    ["볼드모트의 본명은?", 3, "Both"],
                    ["볼드모트의 능력은?", 5, "Both"],
                    ["해리 포터와 볼드모트의 관계는?", 5, "Both"],
                    ["죽음을 먹는 자들이란?", 3, "Both"],
                ],
                inputs=[query_input, top_k_slider, model_type_radio],
            )

            # Connect search button
            search_button.click(
                fn=self.search,
                inputs=[query_input, top_k_slider, model_type_radio],
                outputs=[original_output, finetuned_output, evaluate_button]
            )

            # Connect evaluate button
            evaluate_button.click(
                fn=self.evaluate,
                inputs=[],
                outputs=[evaluation_output]
            )

            # Add footer
            gr.Markdown(
                """
                ---

                **Dataset**: 나무위키 - 볼드모트 문서 (261 text chunks)

                **Training Details**: LoRA finetuning with contrastive learning (InfoNCE loss)

                **Source**: Namu Wiki - Harry Potter Voldemort article

                **Evaluation**: OpenAI `text-embedding-3-small` (neutral benchmark for quality assessment)

                💡 **Tip**: Select "Both" models and click "Evaluate" to compare retrieval quality using OpenAI embeddings

                🤖 Built with [Claude Code](https://claude.com/claude-code)
                """
            )

        # Launch the app
        print("\n" + "=" * 60)
        print("🚀 Launching Gradio Demo...")
        print("=" * 60 + "\n")

        demo.launch(share=share, server_name="0.0.0.0", server_port=7860)


def main():
    """Main function to launch demo"""
    import argparse

    parser = argparse.ArgumentParser(description="Launch RAG comparison demo")
    parser.add_argument(
        "--db-path",
        type=str,
        default="./chroma_db",
        help="Path to ChromaDB"
    )
    parser.add_argument(
        "--finetuned-model",
        type=str,
        default="./logs/tensorboard/run_20251103_083449/model",
        help="Path to finetuned model"
    )
    parser.add_argument(
        "--original-collection",
        type=str,
        default="original_embeddings",
        help="Name of original embeddings collection"
    )
    parser.add_argument(
        "--finetuned-collection",
        type=str,
        default="finetuned_embeddings",
        help="Name of finetuned embeddings collection"
    )
    parser.add_argument(
        "--data-type",
        type=str,
        choices=["KorQuAD", "Wiki"],
        default="KorQuAD",
        help="Type of data in the collections"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public share link"
    )
    args = parser.parse_args()

    # Create and launch demo
    demo_app = RAGDemo(
        db_path=args.db_path,
        finetuned_model_path=args.finetuned_model,
        original_collection=args.original_collection,
        finetuned_collection=args.finetuned_collection,
        data_type=args.data_type
    )

    demo_app.launch(share=args.share)


if __name__ == "__main__":
    main()
