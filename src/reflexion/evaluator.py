import json
import re
import time
from typing import List, Optional

from ..config.settings import settings
from ..core.interfaces import (
    Document,
    LLMInterface,
    ReflexionDecision,
    ReflexionEvaluation,
    ReflexionEvaluatorInterface,
)
from ..llm.github_llm import GitHubLLM


class SmartReflexionEvaluator(ReflexionEvaluatorInterface):
    """Smart reflexion evaluator with confidence scoring and decision making"""

    def __init__(self, evaluation_llm: Optional[LLMInterface] = None):
        # Use dedicated evaluation model
        self.evaluation_llm = evaluation_llm or GitHubLLM(
            model_override=settings.evaluation_model,
            temperature_override=settings.evaluation_temperature,
            max_tokens_override=settings.evaluation_max_tokens,
        )

        # Confidence indicators for analysis
        self.uncertainty_phrases = [
            "i'm not sure",
            "i don't know",
            "unclear",
            "uncertain",
            "might be",
            "could be",
            "possibly",
            "perhaps",
            "maybe",
            "it's difficult to say",
            "hard to determine",
            "not enough information",
            "insufficient data",
            "limited information",
            "unclear from the context",
        ]

    async def evaluate_response(
        self,
        query: str,
        partial_answer: str,
        retrieved_docs: List[Document],
        cycle_number: int,
    ) -> ReflexionEvaluation:
        """Evaluate if response sufficiently answers the query"""

        start_time = time.time()

        # Create evaluation prompt
        evaluation_prompt = self._create_evaluation_prompt(
            query, partial_answer, retrieved_docs, cycle_number
        )

        try:
            # Get evaluation from LLM
            response_chunks = []
            async for chunk in self.evaluation_llm.generate_stream(
                evaluation_prompt
            ):
                response_chunks.append(chunk.content)

            evaluation_response = "".join(response_chunks)

            # Parse evaluation response
            evaluation = self._parse_evaluation_response(
                evaluation_response, query, partial_answer
            )

            # Add processing metadata
            evaluation.metadata = {
                "evaluation_time": time.time() - start_time,
                "cycle_number": cycle_number,
                "docs_count": len(retrieved_docs),
                "answer_length": len(partial_answer),
                "evaluation_model": settings.evaluation_model,
            }

            return evaluation

        except Exception as e:
            # Fallback evaluation on error
            return ReflexionEvaluation(
                confidence_score=0.5,
                decision=ReflexionDecision.CONTINUE,
                reasoning=f"Evaluation failed: {e}",
                follow_up_queries=[],
                covered_aspects=[],
                missing_aspects=["evaluation_error"],
                uncertainty_phrases=[],
                metadata={"error": str(e)},
            )

    async def generate_follow_up_queries(
        self,
        original_query: str,
        partial_answer: str,
        missing_aspects: List[str],
    ) -> List[str]:
        """Generate follow-up queries based on missing aspects"""

        follow_up_prompt = self._create_follow_up_prompt(
            original_query, partial_answer, missing_aspects
        )

        try:
            response_chunks = []
            async for chunk in self.evaluation_llm.generate_stream(
                follow_up_prompt
            ):
                response_chunks.append(chunk.content)

            response = "".join(response_chunks)
            follow_up_queries = self._parse_follow_up_queries(response)

            # Limit to 2 follow-up queries max
            return follow_up_queries[:2]

        except Exception:
            # Fallback: create simple follow-up based on missing aspects
            return [f"What about {aspect}?" for aspect in missing_aspects[:2]]

    def _create_evaluation_prompt(
        self,
        query: str,
        partial_answer: str,
        retrieved_docs: List[Document],
        cycle_number: int,
    ) -> str:
        """Create evaluation prompt for confidence assessment"""

        docs_summary = (
            f"Retrieved {len(retrieved_docs)} documents from knowledge base."
        )
        if retrieved_docs:
            docs_preview = "\n".join(
                [
                    f"- {doc.metadata.get('source', 'Unknown')}: {doc.content[:100]}..."
                    for doc in retrieved_docs[:3]
                ]
            )
            docs_summary += f"\n\nDocument previews:\n{docs_preview}"

        return f"""You are an expert evaluator assessing the quality and completeness of AI responses.

                EVALUATION TASK:
                Assess if the following response sufficiently answers the user's question.

                Original Question: {query}

                Current Response (Cycle {cycle_number}):
                {partial_answer}

                Available Context: {docs_summary}

                EVALUATION CRITERIA:
                1. Completeness: Does the response address all aspects of the question?
                2. Accuracy: Is the response supported by the available documents?
                3. Confidence: Does the response contain uncertain or vague language?
                4. Specificity: Are there specific sub-questions that need more detail?

                RESPONSE FORMAT (JSON):
                {{
                    "confidence_score": 0.35,
                    "decision": "continue|refine_query|complete|insufficient_data",
                    "reasoning": "Detailed explanation of the assessment",
                    "covered_aspects": ["aspect1", "aspect2"],
                    "missing_aspects": ["missing1", "missing2"],
                    "uncertainty_phrases": ["phrase1", "phrase2"],
                    "specific_gaps": ["What specific details are missing?"]
                }}

                DECISION GUIDELINES:
                - confidence_score: 0.0-1.0 (how well the question is answered)
                - "complete": confidence >= 0.8 and no major gaps
                - "continue": confidence < 0.8 but retrievable information exists
                - "refine_query": need more specific queries for missing aspects
                - "insufficient_data": fundamental information is missing from knowledge base

                INSTRUCTION:
                1. Be very strict in the process
                2. Always lower confidence on mistakes
                3. Ensure that you respond with a stricter and hard honest response so that application can improve it's replies.
                Provide your evaluation as valid JSON:"""

    def _create_follow_up_prompt(
        self,
        original_query: str,
        partial_answer: str,
        missing_aspects: List[str],
    ) -> str:
        """Create prompt for generating follow-up queries"""

        missing_text = ", ".join(missing_aspects)

        return f"""Generate 1-2 specific follow-up queries to address missing information.

                Original Question: {original_query}
                Current Answer: {partial_answer}
                Missing Aspects: {missing_text}

                Requirements:
                - Create specific, searchable queries
                - Focus on the most important missing information
                - Make queries standalone (no pronouns)
                - Prioritize factual, retrievable information

                Format as numbered list:
                1. [First follow-up query]
                2. [Second follow-up query]

                Follow-up queries:"""

    def _parse_evaluation_response(
        self, response: str, query: str, partial_answer: str
    ) -> ReflexionEvaluation:
        """Parse LLM evaluation response into structured format"""

        try:
            # Try to extract JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                eval_data = json.loads(json_str)

                # Map decision string to enum
                decision_map = {
                    "continue": ReflexionDecision.CONTINUE,
                    "refine_query": ReflexionDecision.REFINE_QUERY,
                    "complete": ReflexionDecision.COMPLETE,
                    "insufficient_data": ReflexionDecision.INSUFFICIENT_DATA,
                }

                decision = decision_map.get(
                    eval_data.get("decision", "continue").lower(),
                    ReflexionDecision.CONTINUE,
                )

                return ReflexionEvaluation(
                    confidence_score=float(
                        eval_data.get("confidence_score", 0.5)
                    ),
                    decision=decision,
                    reasoning=eval_data.get(
                        "reasoning", "No reasoning provided"
                    ),
                    follow_up_queries=eval_data.get("specific_gaps", []),
                    covered_aspects=eval_data.get("covered_aspects", []),
                    missing_aspects=eval_data.get("missing_aspects", []),
                    uncertainty_phrases=eval_data.get(
                        "uncertainty_phrases", []
                    ),
                )
        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback parsing if JSON fails
            pass

        # Fallback: analyze response with heuristics
        return self._heuristic_evaluation(response, query, partial_answer)

    def _heuristic_evaluation(
        self, response: str, query: str, partial_answer: str
    ) -> ReflexionEvaluation:
        """Fallback heuristic evaluation when JSON parsing fails"""

        answer_lower = partial_answer.lower()

        # Count uncertainty phrases
        uncertainty_count = sum(
            1 for phrase in self.uncertainty_phrases if phrase in answer_lower
        )

        # Basic confidence scoring
        confidence = 0.7  # Base confidence

        # Reduce confidence for uncertainty
        confidence -= min(uncertainty_count * 0.1, 0.3)

        # Reduce confidence for very short answers
        if len(partial_answer) < 100:
            confidence -= 0.2

        # Determine decision based on confidence
        if confidence >= 0.8:
            decision = ReflexionDecision.COMPLETE
        elif confidence >= 0.5:
            decision = ReflexionDecision.CONTINUE
        else:
            decision = ReflexionDecision.REFINE_QUERY

        return ReflexionEvaluation(
            confidence_score=max(0.0, min(1.0, confidence)),
            decision=decision,
            reasoning="Heuristic evaluation based on uncertainty analysis",
            follow_up_queries=[],
            covered_aspects=[],
            missing_aspects=[],
            uncertainty_phrases=[
                phrase
                for phrase in self.uncertainty_phrases
                if phrase in answer_lower
            ],
        )

    def _parse_follow_up_queries(self, response: str) -> List[str]:
        """Parse follow-up queries from LLM response"""

        lines = response.strip().split("\n")
        queries = []

        for line in lines:
            line = line.strip()
            # Match numbered lists
            match = re.match(r"^\d+\.\s*(.+)", line)
            if match:
                query = match.group(1).strip()
                if query and len(query) > 10:  # Minimum query length
                    queries.append(query)

        return queries
