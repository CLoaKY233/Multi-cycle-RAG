-- QA Cache lookup function for semantic similarity search
DEFINE FUNCTION OVERWRITE fn::qa_cache_lookup($query_embedding: array<float>, $threshold: float) -> any {
	RETURN (SELECT id, question, answer, vector::similarity::cosine(question_embedding, $query_embedding) AS score 
	        FROM qa_history 
	        WHERE vector::similarity::cosine(question_embedding, $query_embedding) >= $threshold
	        ORDER BY score DESC LIMIT 1);
} PERMISSIONS FULL
