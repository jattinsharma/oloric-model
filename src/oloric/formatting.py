"""
Prompt formatting and tokenization for OLORIC.
"""
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer

from .config import config
from .schemas.model_input import OloricModelInput
from .schemas.model_output import Diagnosis, Memory, OloricModelOutput, UnderstandingCheck


class OloricFormatter:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.dataset_config = config.get_dataset_config()
        self.preprocessing_config = self.dataset_config.get("preprocessing", {})
        self.template_format = self.preprocessing_config.get("template_format", "chatml")
        self.max_sequence_length = self.preprocessing_config.get("max_sequence_length", 2048)

    # ------------------------------------------------------------------
    # Prompt formatters
    # ------------------------------------------------------------------

    def format_input(self, model_input: OloricModelInput) -> str:
        if self.template_format == "chatml":
            return self._format_chatml(model_input)
        elif self.template_format == "alpaca":
            return self._format_alpaca(model_input)
        elif self.template_format == "llama2":
            return self._format_llama2(model_input)
        else:
            return self._format_simple(model_input)

    def _format_chatml(self, model_input: OloricModelInput) -> str:
        messages = []
        system_content = self._build_system_message(model_input)
        messages.append({"role": "system", "content": system_content})
        for turn in model_input.conversation_context:
            role = "user" if turn.role == "student" else ("assistant" if turn.role in ("tutor", "model") else turn.role)
            messages.append({"role": role, "content": turn.content})
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted = self._format_simple(model_input)
        return formatted

    def _format_alpaca(self, model_input: OloricModelInput) -> str:
        system_content = self._build_system_message(model_input)
        history = ""
        for turn in model_input.conversation_context:
            if turn.role == "student":
                history += f"\n\n### Student:\n{turn.content}"
            else:
                history += f"\n\n### Tutor:\n{turn.content}"
        prompt = (
            "Below is an instruction that describes a task, paired with input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            f"### Instruction:\n{model_input.task}\n\n"
            f"### Input:\n{system_content}{history}\n\n"
            "### Response:\n"
        )
        return prompt

    def _format_llama2(self, model_input: OloricModelInput) -> str:
        system_content = self._build_system_message(model_input)
        history = ""
        for turn in model_input.conversation_context:
            if turn.role == "student":
                history += (
                    f"\n\n### Student:\n{turn.content}\n"
                )
            else:
                history += (
                    f"### Tutor:\n{turn.content}\n"
                )
        prompt = (
            f"SYSTEM:\n{system_content}\n\n"
            f"{history}"
            "ASSISTANT:\n"
        )
        return prompt

    def _format_simple(self, model_input: OloricModelInput) -> str:
        system_content = self._build_system_message(model_input)
        history_parts = []
        for turn in model_input.conversation_context:
            history_parts.append(f"{turn.role.capitalize()}: {turn.content}")
        history = "\n".join(history_parts)
        prompt = (
            f"Task: {model_input.task}\n\n"
            f"Context:\n{system_content}\n\n"
            f"Conversation History:\n{history}\n\n"
            "Response:"
        )
        return prompt

    def _build_system_message(self, model_input: OloricModelInput) -> str:
        doc_ctx = model_input.document_context
        learner_state = model_input.learner_state
        known_pre = (
            ", ".join(learner_state.known_prerequisites)
            if learner_state.known_prerequisites else "None"
        )
        weak_pre = (
            ", ".join(learner_state.weak_prerequisites)
            if learner_state.weak_prerequisites else "None"
        )
        known_mis = (
            ", ".join(learner_state.known_misconceptions)
            if learner_state.known_misconceptions else "None"
        )
        system_parts = [
            "You are OLORIC, an adaptive teaching model. Your goal is to help "
            "the student understand the concept by providing appropriate "
            "explanations and adapting your teaching strategy based on their responses.",
            "\nDocument Context:",
            f"- Document: {doc_ctx.title} (Page {doc_ctx.page}, Section: {doc_ctx.section})",
            f"- Selected Text: \"{doc_ctx.selected_text}\"",
            f"- Surrounding Context: {doc_ctx.surrounding_context[:200]}...",
            "\nLearner State:",
            f"- Level: {learner_state.level}",
            f"- Concept: {learner_state.concept}",
            f"- Mastery: {learner_state.mastery:.2f}",
            f"- Known Prerequisites: {known_pre}",
            f"- Weak Prerequisites: {weak_pre}",
            f"- Known Misconceptions: {known_mis}",
            f"\nCurrent Goal: {model_input.current_goal}",
            "\nBased on the conversation history and learner state, choose the "
            "appropriate tutoring action and strategy.",
        ]
        if doc_ctx.retrieved_evidence:
            system_parts.append("\nRetrieved Evidence:")
            for evidence in doc_ctx.retrieved_evidence[:3]:
                system_parts.append(f"- {evidence}")
        return "\n".join(system_parts)

    def format_output(self, model_output: OloricModelOutput) -> str:
        return model_output.json()

    def parse_output(self, generated_text: str) -> OloricModelOutput:
        try:
            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return OloricModelOutput.parse_raw(json_str)
        except Exception:
            pass
        return OloricModelOutput(
            action="explain",
            strategy="simple_example",
            difficulty="beginner",
            response=generated_text.strip(),
            understanding_check=UnderstandingCheck(required=False, question=None),
            diagnosis=Diagnosis(confusion_type="unknown", misconception=None, missing_prerequisite=None),
            memory=Memory(candidate=False, title=None, content=None, anchor_concept=None),
        )

    # ------------------------------------------------------------------
    # Tokenization (NEW — fixes TrainingExample collator crash)
    # ------------------------------------------------------------------

    def build_prompt_and_target(self, example_dict: Dict[str, Any]) -> Tuple[str, str]:
        """
        Convert a raw TrainingExample dict (from train.jsonl) into
        (prompt_text, target_text).

        The prompt is produced by ``format_input`` via OloricModelInput.
        The target is the full JSON serialisation of the ``target`` field.
        """
        ctx = example_dict.get("context", {})
        model_input = OloricModelInput(
            task=ctx.get("task", example_dict.get("instruction", "Help the student.")),
            document_context=ctx.get("document_context", {}),
            learner_state=ctx.get("learner_state", {}),
            conversation_context=ctx.get("conversation_context", []),
            current_goal=ctx.get("current_goal", ""),
        )
        prompt_text = self.format_input(model_input)
        target_dict = example_dict.get("target", {})
        target_text = json.dumps(target_dict, ensure_ascii=False)
        return prompt_text, target_text

    def tokenize_example(
        self,
        example_dict: Dict[str, Any],
        max_length: int = 2048,
    ) -> Dict[str, List[int]]:
        """
        Tokenize a single raw TrainingExample dict into HuggingFace model features.

        Supervision is applied ONLY to the response (target) tokens.
        Prompt tokens and padding positions are masked to -100 in ``labels``
        so the causal-LM cross-entropy loss is computed only on the target JSON.

        Returns:
            {
                'input_ids':      List[int],
                'attention_mask': List[int],
                'labels':         List[int],  # -100 at prompt & padding positions
            }
        """
        prompt_text, target_text = self.build_prompt_and_target(example_dict)

        # Append EOS to mark end of sequence
        eos = self.tokenizer.eos_token or ""
        full_text = prompt_text + target_text + eos

        # Measure prompt length before padding/truncation for label masking
        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        prompt_len = len(prompt_ids)

        # Tokenize full sequence
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,   # plain Python lists — no torch dependency here
            add_special_tokens=False,
        )

        input_ids      = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels         = list(input_ids)   # mutable copy

        pad_id = self.tokenizer.pad_token_id
        for i in range(len(labels)):
            if i < prompt_len or labels[i] == pad_id:
                labels[i] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }

    def tokenize_function(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """
        HuggingFace Dataset.map()-compatible batch tokenizer.

        ``examples`` must contain key ``'example_dict'`` whose value is a list
        of raw TrainingExample dicts.  Returns standard model-feature dict.
        """
        all_input_ids       = []
        all_attention_masks = []
        all_labels          = []

        for ex_dict in examples.get("example_dict", []):
            result = self.tokenize_example(ex_dict, max_length=self.max_sequence_length)
            all_input_ids.append(result["input_ids"])
            all_attention_masks.append(result["attention_mask"])
            all_labels.append(result["labels"])

        return {
            "input_ids":      all_input_ids,
            "attention_mask": all_attention_masks,
            "labels":         all_labels,
        }
