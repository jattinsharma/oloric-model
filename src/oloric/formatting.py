"""
Prompt formatting and tokenization for OLORIC.
"""
from typing import Dict, Any, List, Optional
from transformers import PreTrainedTokenizer
from .schemas.model_input import OloricModelInput
from .schemas.model_output import OloricModelOutput, UnderstandingCheck, Diagnosis, Memory
from .config import config
import re

class OloricFormatter:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.dataset_config = config.get_dataset_config()
        self.preprocessing_config = self.dataset_config.get("preprocessing", {})
        self.template_format = self.preprocessing_config.get("template_format", "chatml")
        self.max_sequence_length = self.preprocessing_config.get("max_sequence_length", 2048)
    
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
            messages.append({"role": turn.role, "content": turn.content})
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
        prompt = f"""Below is an instruction that describes a task, paired with input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{model_input.task}

### Input:
{system_content}{history}

### Response:
"""
        return prompt
    
    def _format_llama2(self, model_input: OloricModelInput) -> str:
        system_content = self._build_system_message(model_input)
        history = ""
        for turn in model_input.conversation_context:
            if turn.role == "student":
                history += f"\n\n<|start_header_id|>student<|end_header_id|>\n\n{turn.content}<|eot_id|>"
            else:
                history += f"<|start_header_id|>tutor<|end_header_id|>\n\n{turn.content}<|eot_id|>"
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_content}<|eot_id|>{history}<|start_header_id|>assistant<|end_header_id|>
"""
        return prompt
    
    def _format_simple(self, model_input: OloricModelInput) -> str:
        system_content = self._build_system_message(model_input)
        history_parts = []
        for turn in model_input.conversation_context:
            history_parts.append(f"{turn.role.capitalize()}: {turn.content}")
        history = "\n".join(history_parts)
        prompt = f"""Task: {model_input.task}

Context:
{system_content}

Conversation History:
{history}

Response:"""
        return prompt
    
    def _build_system_message(self, model_input: OloricModelInput) -> str:
        doc_ctx = model_input.document_context
        learner_state = model_input.learner_state
        system_parts = [
            "You are OLORIC, an adaptive teaching model. Your goal is to help the student understand the concept by providing appropriate explanations and adapting your teaching strategy based on their responses.",
            f"\nDocument Context:",
            f"- Document: {doc_ctx.title} (Page {doc_ctx.page}, Section: {doc_ctx.section})",
            f"- Selected Text: \"{doc_ctx.selected_text}\"",
            f"- Surrounding Context: {doc_ctx.surrounding_context[:200]}...",
            f"\nLearner State:",
            f"- Level: {learner_state.level}",
            f"- Concept: {learner_state.concept}",
            f"- Mastery: {learner_state.mastery:.2f}",
            f"- Known Prerequisites: {', '.join(learner_state.known_prerequisites) if learner_state.known_prerequisites else 'None'}",
            f"- Weak Prerequisites: {', '.join(learner_state.weak_prerequisites) if learner_state.weak_prerequisites else 'None'}",
            f"- Known Misconceptions: {', '.join(learner_state.known_misconceptions) if learner_state.known_misconceptions else 'None'}",
            f"\nCurrent Goal: {model_input.current_goal}",
            f"\nBased on the conversation history and learner state, choose the appropriate tutoring action and strategy."
        ]
        if doc_ctx.retrieved_evidence:
            system_parts.append(f"\nRetrieved Evidence:")
            for i, evidence in enumerate(doc_ctx.retrieved_evidence[:3]):
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
            memory=Memory(candidate=False, title=None, content=None, anchor_concept=None)
        )
    
    def tokenize_function(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
