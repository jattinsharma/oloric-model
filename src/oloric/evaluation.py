"""
Evaluation suite for OLORIC model.
"""
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from .schemas.model_input import OloricModelInput
from .schemas.model_output import OloricModelOutput
from .inference import inference_engine
from .config import config
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluating a single example."""
    example_id: str
    dimension_scores: Dict[str, float]
    overall_score: float
    feedback: str
    model_output: OloricModelOutput
    expected_output: OloricModelOutput


class OloricEvaluator:
    """Handles evaluation of OLORIC model."""
    
    def __init__(self):
        """Initialize OLORIC evaluator."""
        self.eval_config = config.get_evaluation_config()
        self.dimensions = self.eval_config.get("dimensions", {})
        self.benchmark_path = self.eval_config.get("benchmark", {}).get("path", "./evaluation/benchmark.jsonl")
        
    def load_benchmark(self) -> List[Dict[str, Any]]:
        """
        Load evaluation benchmark.
        
        Returns:
            List of evaluation examples
        """
        try:
            with open(self.benchmark_path, 'r') as f:
                examples = []
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            examples.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse benchmark line: {e}")
                return examples
        except FileNotFoundError:
            logger.error(f"Benchmark file not found: {self.benchmark_path}")
            return []
    
    def evaluate_example(self, example: Dict[str, Any]) -> EvaluationResult:
        """
        Evaluate a single example.
        
        Args:
            example: Evaluation example dictionary
            
        Returns:
            EvaluationResult object
        """
        # This is a simplified evaluation - in practice, we would use rubrics
        # and potentially human evaluation for many dimensions
        
        example_id = example.get("id", "unknown")
        
        # Convert context and target to proper objects
        context = OloricModelInput(**example["context"])
        expected_output = OloricModelOutput(**example["target"])
        
        # Generate model output
        model_output = inference_engine.generate_response(context)
        
        # Score each dimension (placeholder implementation)
        dimension_scores = {}
        for dimension_name in self.dimensions.keys():
            # Placeholder scoring - in reality, this would use rubrics
            dimension_scores[dimension_name] = self._score_dimension(
                dimension_name, context, model_output, expected_output
            )
        
        # Calculate overall score (weighted average)
        overall_score = self._calculate_overall_score(dimension_scores)
        
        # Generate feedback
        feedback = self._generate_feedback(dimension_scores, model_output, expected_output)
        
        return EvaluationResult(
            example_id=example_id,
            dimension_scores=dimension_scores,
            overall_score=overall_score,
            feedback=feedback,
            model_output=model_output,
            expected_output=expected_output
        )
    
    def _score_dimension(self, dimension_name: str, context: OloricModelInput,
                        model_output: OloricModelOutput, 
                        expected_output: OloricModelOutput) -> float:
        """
        Score a specific evaluation dimension.
        
        Args:
            dimension_name: Name of dimension to score
            context: Input context
            model_output: Model's output
            expected_output: Expected output
            
        Returns:
            Score from 0.0 to 4.0
        """
        # Placeholder implementation - returns fixed scores for now
        # In a real implementation, each dimension would have specific scoring logic
        return 3.0  # Placeholder score
    
    def _calculate_overall_score(self, dimension_scores: Dict[str, float]) -> float:
        """
        Calculate overall score from dimension scores.
        
        Args:
            dimension_scores: Dictionary of dimension scores
            
        Returns:
            Overall score
        """
        if not dimension_scores:
            return 0.0
        
        # Get weights from config
        weights = {}
        total_weight = 0.0
        for name in dimension_scores.keys():
            weight = config.get(f"evaluation.dimensions.{name}.weight", 0.0)
            weights[name] = weight
            total_weight += weight
        
        # Normalize weights if they don't sum to 1
        if total_weight > 0:
            weights = {name: weight/total_weight for name, weight in weights.items()}
        else:
            # Equal weighting if no weights specified
            weights = {name: 1.0/len(dimension_scores) for name in dimension_scores.keys()}
        
        # Calculate weighted average
        overall_score = sum(
            score * weights[name] for name, score in dimension_scores.items()
        )
        
        return round(overall_score, 2)
    
    def _generate_feedback(self, dimension_scores: Dict[str, float],
                          model_output: OloricModelOutput,
                          expected_output: OloricModelOutput) -> str:
        """
        Generate feedback for the evaluation.
        
        Args:
            dimension_scores: Dictionary of dimension scores
            model_output: Model's output
            expected_output: Expected output
            
        Returns:
            Feedback string
        """
        # Find lowest scoring dimensions
        sorted_dims = sorted(dimension_scores.items(), key=lambda x: x[1])
        lowest_dims = sorted_dims[:2] if len(sorted_dims) >= 2 else sorted_dims
        
        feedback_parts = []
        if lowest_dims:
            feedback_parts.append(f"Areas for improvement: {', '.join([dim for dim, score in lowest_dims])}")
        
        # Check if response is too short/long
        response_length = len(model_output.response.split())
        if response_length < 10:
            feedback_parts.append("Response may be too brief")
        elif response_length > 200:
            feedback_parts.append("Response may be overly verbose")
        
        # Check if understanding check is appropriate
        if model_output.understanding_check.required and not model_output.understanding_check.question:
            feedback_parts.append("Understanding check required but no question provided")
        
        if not feedback_parts:
            feedback_parts.append("Performance is satisfactory across dimensions")
        
        return "; ".join(feedback_parts)
    
    def evaluate_benchmark(self, limit: Optional[int] = None) -> List[EvaluationResult]:
        """
        evaluate the entire benchmark.

        Args:
            limit: Maximum number of examples to evaluate (None for all)

        Returns:
            List of EvaluationResult objects
        """
        examples = self.load_benchmark()
        if limit:
            examples = examples[:limit]

        results = []
        for example in examples:
            try:
                result = self.evaluate_example(example)
                results.append(result)
                logger.info(f"Evaluated example {result.example_id}: score {result.overall_score}")
            except Exception as e:
                logger.error(f"Failed to evaluate example {example.get('id', 'unknown')}: {e}")
        return results

    def generate_report(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Generate a report from evaluation results.

        Args:
            results: List of EvaluationResult objects

        Returns:
            Dictionary containing the report
        """
        if not results:
            return {
                "num_examples": 0,
                "overall_score": {"mean": 0.0, "min": 0.0, "max": 0.0},
                "dimension_scores": {}
            }

        # Calculate overall score statistics
        overall_scores = [r.overall_score for r in results]

        # Calculate dimension score statistics
        dimension_scores = {}
        if results:
            # Get all dimension names from the first result
            first_result = results[0]
            for dim_name in first_result.dimension_scores.keys():
                dim_values = [r.dimension_scores.get(dim_name, 0.0) for r in results]
                dimension_scores[dim_name] = {
                    "mean": sum(dim_values) / len(dim_values),
                    "min": min(dim_values),
                    "max": max(dim_values)
                }

        report = {
            "num_examples": len(results),
            "overall_score": {
                "mean": sum(overall_scores) / len(overall_scores),
                "min": min(overall_scores),
                "max": max(overall_scores)
            },
            "dimension_scores": dimension_scores
        }

        return report
