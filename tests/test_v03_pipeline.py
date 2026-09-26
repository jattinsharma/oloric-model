#!/usr/bin/env python3
"""
Regression and pipeline tests for OLORIC v0.3.
Verifies that task and instruction information are preserved end-to-end
across model input schemas, prompt formatting, tokenization, and evaluation.
"""

import json
import os
import sys
import pytest
from transformers import AutoTokenizer

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_dir = os.path.join(_repo_root, "src")
for _p in (_src_dir, _repo_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from oloric.schemas.model_input import (
    DocumentContext,
    LearnerState,
    ConversationTurn,
    OloricModelInput,
)
from oloric.schemas.model_output import OloricModelOutput, UnderstandingCheck, Diagnosis, Memory
from oloric.formatting import OloricFormatter, example_to_model_input
from oloric.evaluation import OloricEvaluator, EvaluationResult


CHATML_TEMPLATE = (
    "{% for m in messages %}"
    "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.chat_template = CHATML_TEMPLATE
    return tok


@pytest.fixture
def formatter(tokenizer):
    return OloricFormatter(tokenizer)


@pytest.fixture
def base_sample_dict():
    """A standard benchmark-like or training-like example."""
    return {
        "id": "test_example_001",
        "task": "follow_up_questions",
        "instruction": "Ask 2 targeted follow-up questions to check understanding of Photosynthesis.",
        "category": "follow_up_questions",
        "domain": "biology",
        "context": {
            "document_context": {
                "document_id": "bio_doc_01",
                "title": "Photosynthesis Fundamentals",
                "page": 42,
                "section": "Light Reactions",
                "selected_text": "Chlorophyll absorbs photons in the thylakoid membrane.",
                "surrounding_context": "Photosynthesis occurs in chloroplasts. Chlorophyll absorbs photons in the thylakoid membrane.",
                "retrieved_evidence": ["Thylakoid membranes contain light-harvesting complexes."]
            },
            "learner_state": {
                "level": "intermediate",
                "concept": "photosynthesis_light_reaction",
                "mastery": 0.45,
                "known_prerequisites": ["cell_structure"],
                "weak_prerequisites": ["electron_transport"],
                "known_misconceptions": ["plants_breathe_only_co2"]
            },
            "conversation_context": [
                {"role": "student", "content": "I understand chlorophyll absorbs light, but what does the electron do?"}
            ],
            "current_goal": "Understand the role of electrons in the light-dependent reactions"
        },
        "target": {
            "action": "question",
            "strategy": "guided_inquiry",
            "difficulty": "intermediate",
            "response": "What molecule donates an electron when chlorophyll is excited?",
            "understanding_check": {"required": True, "question": "Where does the replacement electron come from?"},
            "diagnosis": {"confusion_type": "mechanism", "severity": "moderate", "misconception_addressed": None},
            "memory": {"candidate": False, "title": None, "content": None, "anchor_concept": None}
        }
    }


class TestTaskAndInstructionPreservation:
    """Test suite verifying end-to-end task and instruction preservation."""

    def test_task_follow_up_questions_in_prompt(self, formatter, base_sample_dict):
        """Test A: A sample with task='follow_up_questions' produces a prompt containing the task."""
        sample = dict(base_sample_dict)
        sample["task"] = "follow_up_questions"
        model_input = formatter.example_to_model_input(sample)
        prompt = formatter.format_input(model_input)

        assert "Tutoring Task: follow_up_questions" in prompt
        assert model_input.task == "follow_up_questions"

    def test_task_practice_questions_in_prompt(self, formatter, base_sample_dict):
        """Test B: A sample with task='practice_questions' produces a prompt containing the task."""
        sample = dict(base_sample_dict)
        sample["task"] = "practice_questions"
        sample["instruction"] = "Provide 2 practice problems on electron transfer."
        model_input = formatter.example_to_model_input(sample)
        prompt = formatter.format_input(model_input)

        assert "Tutoring Task: practice_questions" in prompt
        assert model_input.task == "practice_questions"

    def test_different_instructions_produce_different_prompts(self, formatter, base_sample_dict):
        """Test C: Samples with different instructions but identical conversation produce different prompts."""
        sample_1 = dict(base_sample_dict)
        sample_1["instruction"] = "Ask 2 targeted follow-up questions to check understanding."

        sample_2 = dict(base_sample_dict)
        sample_2["instruction"] = "Provide a step-by-step numerical analogy."

        inp_1 = formatter.example_to_model_input(sample_1)
        inp_2 = formatter.example_to_model_input(sample_2)

        prompt_1 = formatter.format_input(inp_1)
        prompt_2 = formatter.format_input(inp_2)

        assert prompt_1 != prompt_2
        assert "Instruction: Ask 2 targeted follow-up questions to check understanding." in prompt_1
        assert "Instruction: Provide a step-by-step numerical analogy." in prompt_2

    def test_root_example_instruction_reaches_model_prompt(self, formatter, base_sample_dict):
        """Test D: Root example instruction reaches the final model prompt."""
        sample = dict(base_sample_dict)
        sample["instruction"] = "Specific unique tutor directive: highlight ATP synthase operation."
        # Ensure context does NOT have instruction, proving root-level is picked up
        sample["context"].pop("instruction", None)

        model_input = example_to_model_input(sample)
        assert model_input.instruction == "Specific unique tutor directive: highlight ATP synthase operation."

        prompt = formatter.format_input(model_input)
        assert "- Instruction: Specific unique tutor directive: highlight ATP synthase operation." in prompt

    def test_eval_and_train_path_prompt_alignment(self, formatter, base_sample_dict):
        """Test E: Evaluation path uses the same instruction as training path and produces identical prompt."""
        # 1. Training path
        train_prompt, _ = formatter.build_prompt_and_target(base_sample_dict)

        # 2. Evaluation path (via example_to_model_input as in evaluate_example)
        eval_input = example_to_model_input(base_sample_dict)
        eval_prompt = formatter.format_input(eval_input)

        assert train_prompt == eval_prompt, "Training prompt and evaluation prompt must be bit-for-bit identical"

    def test_record_a_vs_record_b_distinct_prompts(self, formatter, base_sample_dict):
        """
        Test F: Focused test proving Record A (task='follow_up_questions') and
        Record B (task='practice_questions') with identical conversation history
        do NOT produce identical model prompts (resolves the v0.2 root cause).
        """
        # Both records have identical conversation context & document context
        rec_a = dict(base_sample_dict)
        rec_a["task"] = "follow_up_questions"
        rec_a["instruction"] = "Generate follow-up questions to test grasp."

        rec_b = dict(base_sample_dict)
        rec_b["task"] = "practice_questions"
        rec_b["instruction"] = "Create practice exercises for the student."

        inp_a = example_to_model_input(rec_a)
        inp_b = example_to_model_input(rec_b)

        prompt_a = formatter.format_input(inp_a)
        prompt_b = formatter.format_input(inp_b)

        # Prompts must differ and contain their respective tasks
        assert prompt_a != prompt_b
        assert "Tutoring Task: follow_up_questions" in prompt_a
        assert "Tutoring Task: practice_questions" in prompt_b
        assert "Instruction: Generate follow-up questions to test grasp." in prompt_a
        assert "Instruction: Create practice exercises for the student." in prompt_b

    def test_backward_compatibility_legacy_record(self, formatter):
        """Legacy records without root instruction or with task='resolve_confusion' parse cleanly."""
        legacy_example = {
            "id": "legacy_001",
            "context": {
                "document_context": {
                    "document_id": "legacy_doc",
                    "title": "Legacy Doc",
                    "page": 1,
                    "section": "Intro",
                    "selected_text": "Sample text",
                    "surrounding_context": "Sample context",
                    "retrieved_evidence": []
                },
                "learner_state": {
                    "level": "beginner",
                    "concept": "fractions",
                    "mastery": 0.2,
                    "known_prerequisites": [],
                    "weak_prerequisites": [],
                    "known_misconceptions": []
                },
                "conversation_context": [
                    {"role": "student", "content": "I don't understand fractions."}
                ],
                "current_goal": "Understand 1/2"
            },
            "target": {
                "action": "explain",
                "strategy": "analogy",
                "difficulty": "beginner",
                "response": "Imagine a pizza cut in half.",
                "understanding_check": {"required": False, "question": None},
                "diagnosis": {"confusion_type": "concept", "severity": "mild", "misconception_addressed": None},
                "memory": {"candidate": False, "title": None, "content": None, "anchor_concept": None}
            }
        }
        model_input = example_to_model_input(legacy_example)
        assert model_input.task == "resolve_confusion"
        assert model_input.instruction is None

        prompt = formatter.format_input(model_input)
        assert "Tutoring Task: resolve_confusion" in prompt
        assert "Instruction:" not in prompt
        assert "Legacy Doc" in prompt

    def test_all_templates_preserve_task_and_instruction(self, tokenizer, base_sample_dict):
        """Verify task and instruction appear across all template formats (chatml, alpaca, simple)."""
        model_input = example_to_model_input(base_sample_dict)

        for tmpl in ["chatml", "alpaca", "simple", "llama2"]:
            fmt = OloricFormatter(tokenizer)
            fmt.template_format = tmpl
            prompt = fmt.format_input(model_input)

            assert "follow_up_questions" in prompt, f"Task missing in template {tmpl}"
            assert "Photosynthesis" in prompt, f"Instruction missing in template {tmpl}"
