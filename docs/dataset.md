# OLORIC Dataset Design

## Overview

The OLORIC dataset is designed to teach the model HOW TO TEACH, not what specific content to teach. It focuses on tutoring behaviors, strategies, and adaptive teaching techniques that transfer across domains.

## Design Philosophy

### Behavior-Focused, Not Content-Focused
- Examples demonstrate tutoring techniques rather than subject mastery
- Model learns to diagnose confusion, select strategies, and adapt explanations
- Content varies widely to prevent memorization and encourage generalization

### Multi-Turn Emphasis
- Prefers rich, multi-turn tutoring sequences over single Q&A pairs
- Captures the dynamic nature of real tutoring interactions
- Models strategy switching based on student responses

### Domain Diversity
- Examples span multiple academic disciplines
- Ensures tutoring skills transfer across subjects
- Prevents overfitting to specific terminology or concepts

## Dataset Structure

### JSONL Format
Each line is a valid JSON object representing a complete training example:

```jsonl
{
  "id": "example_001",
  "category": "strategy_switch",
  "domain": "economics",
  "instruction": "Resolve student confusion about MPC",
  "context": {/* OloricModelInput */},
  "target": {/* OloricModelOutput */}
}
```

### Categories (Types of Tutoring Behavior)

1. **simple_explanation** - Straightforward, clear explanations
2. **simplification** - Breaking down complex ideas into simpler terms
3. **analogy** - Using familiar concepts to explain unfamiliar ones
4. **concrete_example** - Specific, tangible instances
5. **numerical_example** - Quantitative examples with numbers
6. **prerequisite_detection** - Identifying missing foundational knowledge
7. **misconception_detection** - Spotting and addressing wrong beliefs
8. **follow_up_questions** - Asking probing questions to understand confusion
9. **multi_turn_tutoring** - Extended tutoring sequences
10. **repeated_confusion** - Handling "I still don't understand" responses
11. **strategy_switching** - Changing approach when initial explanation fails
12. **diagnostic_questions** - Questions designed to pinpoint confusion type
13. **hint_based_teaching** - Providing clues rather than direct answers
14. **practice_questions** - Give students opportunities to apply knowledge
15. **error_correction** - Identifying and correcting mistakes
16. **partial_understanding** - Building on what the student already knows
17. **understanding_confirmation** - Verifying true comprehension
18. **memory_generation** - Creating summary notes for future reference
19. **document_grounded** - Tightly coupled to specific document passages
20. **context_retention** - Maintaining and using earlier conversation information

### Domains (Academic Fields)

1. **economics** - Macroeconomics, microeconomics, econometrics
2. **accountancy** - Financial accounting, managerial accounting, auditing
3. **mathematics** - Algebra, calculus, statistics, geometry
4. **science** - Physics, chemistry, biology, earth science
5. **nutrition_food_science** - Human nutrition, food chemistry, dietetics
6. **general_academic** - Cross-disciplinary study skills, research methods

## Example Structure Details

### Context (Input)
The model receives a structured input containing:
- **task**: Usually "resolve_confusion"
- **document_context**: 
  - document_id, title, page, section
  - selected_text (what student is confused about)
  - surrounding_context (broader passage)
  - retrieved_evidence (from RAG system)
- **learner_state**:
  - level (beginner/intermediate/advanced)
  - concept (what they're trying to learn)
  - mastery (0.0-1.0 score)
  - known_prerequisites, weak_prerequisites, known_misconceptions
- **conversation_context**: Array of {role: "student"/"tutor", content: "..."}
- **current_goal**: What the student wants to achieve

### Target (Expected Output)
The model should produce:
- **action**: What type of tutoring action to take
- **strategy**: Specific technique to use for that action
- **difficulty**: Appropriate difficulty level
- **response**: The actual tutoring response to the student
- **understanding_check**: Whether and how to check understanding
- **diagnosis**: Assessment of the confusion (type, misconceptions, missing prerequisites)
- **memory**: Whether this interaction should be saved as a memory candidate

## Quality Characteristics

### High-Value Examples
- Multi-turn sequences showing adaptation
- Clear demonstration of strategy switching
- Realistic student confusions and tutor responses
- Accurate diagnosis of learning issues
- Appropriate understanding checks that reveal true comprehension
- Memory candidates that capture key insights

### Avoidance of Pitfalls
- No memorization of specific facts (focus on process)
- No contradictory examples (same input → different correct outputs)
- No incomplete or ambiguous examples
- No examples where tutor gives up or provides wrong information
- No examples that promote misconceptions

## Data Generation Approach

### Seed Creation
- Manually crafted high-quality examples (300-500)
- Created by educators and subject matter experts
- Reviewed for pedagogical soundness
- Spanning multiple domains and tutoring categories

### Expansion Strategies
- Template-based generation from seed examples
- Controlled variation of concepts, domains, and difficulties
- Human-in-the-loop validation of generated examples
- Ongoing addition of edge cases and challenging scenarios

### Validation Checks
- Schema validation (all required fields present)
- JSON validity
- No empty targets or responses
- Valid enum values (actions, strategies, difficulties, etc.)
- Logical conversation order (alternating turns, starts with student)
- No duplicate examples
- No contradictory schema (e.g., understanding check required but no question)
- Valid document anchors (page numbers positive, etc.)
- No fabricated source references

## Statistics Tracked

### Per Category
- Count of examples
- Average conversation length
- Token length distribution

### Per Domain
- Distribution of examples
- Concept difficulty levels
- Prerequisite complexity

### Overall
- Train/validation/test split ratios
- Average turns per conversation
- Percentage of multi-turn examples
- Strategy switching frequency
- Memory candidate generation rate

## Usage Guidelines

### Training
- Shuffle examples before batching
- Maintain category balance in batches when possible
- Monitor for overfitting to specific domains or concepts
- Validate on held-out examples regularly

### Evaluation
- Use separate validation and test sets
- Ensure evaluation examples don't appear in training
- Test transfer to unseen domains and concepts
- Measure both immediate responses and long-term adaptation

### Augmentation
- Back-paraphrasing of responses (preserving meaning)
- Concept substitution within same domain
- Difficulty level adjustment
- Learner state variation (different mastery levels)

## Ethical Considerations

### Bias Mitigation
- Balanced representation across genders, ethnicities, cultures in examples
- Avoid stereotypes in analogies and examples
- Inclusive language in all generated content
- Regular auditing for unintended biases

### Educational Soundness
- All examples reviewed by educational professionals
- Alignment with established learning science principles
- Promotion of growth mindset and productive struggle
- Avoidance of oversimplification that leads to misconceptions

### Privacy
- No real student data used in examples
- All examples are synthetic or anonymized
- No personally identifiable information in any field

