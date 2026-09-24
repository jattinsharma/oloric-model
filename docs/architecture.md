# OLORIC Architecture

## Overview

OLORIC is designed as a specialized tutoring model that learns HOW TO TEACH, separate from WHAT to teach. The architecture follows a modular design that separates concerns and enables easy integration with future applications.

## Core Components

### 1. Model Base
- **Foundation Model**: Open-weight instruction-tuned Transformer (Qwen3-4B-Instruct)
- **Adaptation Method**: LoRA/QLoRA adapters for efficient fine-tuning
- **Key Feature**: Frozen base model weights, only adapter parameters trained

### 2. Input Processing Module
- **Input Schema Validator**: Ensures all inputs conform to the expected structure
- **Context Builder**: Constructs system messages from document context, learner state, and conversation history
- **Template Formatter**: Supports multiple prompt formats (ChatML, Alpaca, Llama2, Simple)

### 3. Inference Engine
- **Model Loader**: Handles base model and adapter loading with quantization support
- **Response Generator**: Generates responses using controlled sampling parameters
- **Output Parser**: Converts generated text back to structured OloricModelOutput

### 4. Training Pipeline
- **Dataset Processor**: Converts structured examples to model-ready format
- **QLoRA Trainer**: Implements efficient fine-tuning with memory optimization
- **Checkpoint Manager**: Saves and loads model states during training

### 5. Evaluation System
- **Benchmark Runner**: Executes evaluation scenarios against the model
- **Dimension Scorer**: Assesses performance across multiple tutoring dimensions
- **Report Generator**: Creates detailed evaluation reports

### 6. Data Management
- **Data Loader**: Handles loading/saving of JSONL datasets
- **Validator**: Ensures dataset quality and schema compliance
- **Preprocessor**: Converts between structured examples and model formats

## Information Flow

```
[Application] 
     ↓ (Provides context, learner state, conversation)
[Input Processor] 
     ↓ (Builds prompt, formats input)
[Inference Engine] 
     ↓ (Generates response)
[Output Parser] 
     ↓ (Converts to structured output)
[Application] 
     ← (Receives tutoring action, strategy, response, etc.)
```

## Key Design Decisions

### Separation of Concerns
- **WHAT to teach** is handled by external systems (document parser, retrieval, RAG)
- **HOW to teach** is learned by OLORIC through specialized tutoring data

### Modularity
- Each component can be replaced or upgraded independently
- Clear interfaces between modules via well-defined schemas
- Configuration-driven behavior for easy experimentation

### Efficiency
- QLoRA enables training on consumer hardware (L4 24GB)
- 4-bit quantization reduces memory footprint
- Gradient checkpointing trades compute for memory

### Extensibility
- Input/output schemas can be extended without breaking changes
- New teaching strategies can be added via configuration
- Evaluation dimensions can be expanded as needed

## Integration Points

### With Document Processing Systems
- Receives: document_id, page, section, selected_text, surrounding_context, retrieved_evidence
- Provides: tutoring response tailored to the specific document context

### With Learner Modeling Systems
- Receives: learner_state (level, concept, mastery, prerequisites, misconceptions)
- Provides: updated diagnosis, understanding checks, memory candidates

### With Conversation Systems
- Receives: conversation_history (alternating student/tutor turns)
- Provides: contextually appropriate responses that continue the dialogue

### With Application Frontend
- Receives: current_learning_goal, task_specification
- Provides: actionable tutoring response with optional understanding checks

## Data Flow Example

1. **Application** detects student confusion about "MPC" on page 37 of economics textbook
2. **Application** gathers context: 
   - Document info (title, page, section)
   - Learner state (beginner level, low MPC mastery)
   - Conversation history ("I don't understand MPC.")
   - Current goal ("Understand MPC well enough to apply it")
3. **Application** sends structured input to OLORIC model
4. **Model** processes input:
   - Builds system message with all context
   - Formats prompt according to template
   - Generates response using LoRA-adapted weights
   - Parses output into structured format
5. **Model** returns:
   - Action: "explain"
   - Strategy: "simple_example" 
   - Response: "MPC is..."
   - Understanding check: "If you get $100 extra..."
   - Diagnosis: "conceptual confusion"
   - Memory candidate: Not yet (understanding not demonstrated)
6. **Application** presents response to student
7. Loop continues until understanding is demonstrated, then memory candidate is generated

## Scalability Considerations

### Horizontal Scaling
- Stateless inference enables load balancing
- Multiple model instances can serve different users
- Shared adapter weights reduce storage overhead

### Vertical Scaling
- Model size can be increased (7B, 14B variants)
- Sequence length can be adjusted for longer contexts
- Batch size can be increased with more VRAM

### Future Extensions
- Multimodal inputs (diagrams, formulas as images)
- Different base models (domain-specific variants)
- Additional evaluation dimensions
- New teaching strategies based on educational research

