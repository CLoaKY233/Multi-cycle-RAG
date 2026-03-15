-- QA History table for semantic caching
DEFINE TABLE IF NOT EXISTS qa_history SCHEMAFULL;
DEFINE FIELD question ON qa_history TYPE string;
DEFINE FIELD question_embedding ON qa_history TYPE array<float>;
DEFINE FIELD answer ON qa_history TYPE string;
DEFINE FIELD metadata ON qa_history FLEXIBLE TYPE object;
DEFINE FIELD created_at ON qa_history TYPE datetime DEFAULT time::now();
