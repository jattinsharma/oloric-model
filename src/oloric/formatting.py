"""
Prompt formatting and tokenization for OLORIC.
"""
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer

from .config import config
from .schemas.model_input import (
    ConversationTurn,
    DocumentContext,
    LearnerState,
    OloricModelInput,
)
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
        inst = model_input.instruction or model_input.task
        prompt = (
            "Below is an instruction that describes a task, paired with input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            f"### Instruction:\n{inst}\n\n"
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
        inst_line = f"Instruction: {model_input.instruction}\n" if model_input.instruction else ""
        prompt = (
            f"Task: {model_input.task}\n\n"
            f"{inst_line}"
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
        ]
        if model_input.task:
            system_parts.append(f"- Tutoring Task: {model_input.task}")
        if model_input.instruction:
            system_parts.append(f"- Instruction: {model_input.instruction}")
        system_parts.append(
            "\nBased on the conversation history and learner state, choose the "
            "appropriate tutoring action and strategy."
        )
        if doc_ctx.retrieved_evidence:
            system_parts.append("\nRetrieved Evidence:")
            for evidence in doc_ctx.retrieved_evidence[:3]:
                system_parts.append(f"- {evidence}")
        return "\n".join(system_parts)


    def format_output(self, model_output: OloricModelOutput) -> str:
        return model_output.json()

    def parse_output(self, generated_text: str) -> OloricModelOutput:
        """
        Parse model-generated text into OloricModelOutput.

        Raises:
            pydantic.ValidationError: if the extracted JSON fails schema
                validation (e.g. a required field like ``severity`` is absent).
                The caller is responsible for catching this and recording the
                failure rather than silently discarding the example.
        """
        import pydantic

        json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                parsed = None

            if parsed is not None:
                # Let ValidationError propagate so callers can record it.
                return OloricModelOutput.parse_obj(parsed)

        # No JSON block found at all — return a clearly-marked fallback.
        # All required fields must be present to avoid a second crash here.
        return OloricModelOutput(
            action="explain",
            strategy="simple_example",
            difficulty="beginner",
            response=generated_text.strip(),
            understanding_check=UnderstandingCheck(required=False, question=None),
            diagnosis=Diagnosis(
                confusion_type="unknown",
                severity="unknown",
                misconception_addressed=None,
            ),
            memory=Memory(candidate=False, title=None, content=None, anchor_concept=None),
        )

    # ------------------------------------------------------------------
    # Tokenization (NEW — fixes TrainingExample collator crash)
    # ------------------------------------------------------------------

    def example_to_model_input(self, example_dict: Dict[str, Any]) -> OloricModelInput:
        """Helper method delegating to example_to_model_input."""
        return example_to_model_input(example_dict)

    def build_prompt_and_target(self, example_dict: Dict[str, Any]) -> Tuple[str, str]:
        """
        Convert a raw TrainingExample dict (from train.jsonl) into
        (prompt_text, target_text).

        The prompt is produced by ``format_input`` via OloricModelInput.
        The target is the full JSON serialisation of the ``target`` field.
        """
        model_input = self.example_to_model_input(example_dict)
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


def example_to_model_input(example_dict: Dict[str, Any]) -> OloricModelInput:
    """
    Convert an example dictionary (from benchmark, training split, or dataset)
    into a fully populated OloricModelInput object, preserving task, instruction,
    document context, learner state, and conversation turns.
    """
    ctx = example_dict.get("context", {})
    if not ctx and ("document_context" in example_dict or "learner_state" in example_dict):
        ctx = example_dict

    instruction = example_dict.get("instruction") or ctx.get("instruction")

    task = example_dict.get("task")
    if not task or task == "resolve_confusion":
        task = (
            example_dict.get("category")
            or (ctx.get("task") if ctx.get("task") != "resolve_confusion" else None)
            or task
            or ctx.get("task")
            or instruction
            or "resolve_confusion"
        )

    doc_ctx_raw = ctx.get("document_context", {})
    if isinstance(doc_ctx_raw, DocumentContext):
        doc_ctx = doc_ctx_raw
    elif isinstance(doc_ctx_raw, dict):
        doc_ctx = DocumentContext(
            document_id=doc_ctx_raw.get("document_id", "doc_default"),
            title=doc_ctx_raw.get("title", "Study Document"),
            page=doc_ctx_raw.get("page", 1),
            section=doc_ctx_raw.get("section", "General"),
            selected_text=doc_ctx_raw.get("selected_text", ""),
            surrounding_context=doc_ctx_raw.get("surrounding_context", ""),
            retrieved_evidence=doc_ctx_raw.get("retrieved_evidence", [])
        )
    else:
        doc_ctx = DocumentContext(
            document_id="doc_default", title="Study Document", page=1,
            section="General", selected_text="", surrounding_context="",
            retrieved_evidence=[]
        )

    ls_raw = ctx.get("learner_state", {})
    if isinstance(ls_raw, LearnerState):
        learner_state = ls_raw
    elif isinstance(ls_raw, dict):
        learner_state = LearnerState(
            level=ls_raw.get("level", "intermediate"),
            concept=ls_raw.get("concept", "general_topic"),
            mastery=float(ls_raw.get("mastery", 0.5)),
            known_prerequisites=ls_raw.get("known_prerequisites", []),
            weak_prerequisites=ls_raw.get("weak_prerequisites", []),
            known_misconceptions=ls_raw.get("known_misconceptions", [])
        )
    else:
        learner_state = LearnerState(
            level="intermediate", concept="general_topic", mastery=0.5,
            known_prerequisites=[], weak_prerequisites=[], known_misconceptions=[]
        )

    raw_turns = ctx.get("conversation_context") or ctx.get("conversation_turns", [])
    turns = []
    for t in raw_turns:
        if isinstance(t, ConversationTurn):
            turns.append(t)
        elif isinstance(t, dict):
            turns.append(ConversationTurn(role=t.get("role", "student"), content=t.get("content", "")))

    concept_name = learner_state.concept if learner_state else "the concept"
    current_goal = ctx.get("current_goal") or f"Understand {concept_name} well enough to apply it"

    return OloricModelInput(
        task=str(task),
        document_context=doc_ctx,
        learner_state=learner_state,
        conversation_context=turns,
        current_goal=str(current_goal),
        instruction=str(instruction) if instruction else None
    )
