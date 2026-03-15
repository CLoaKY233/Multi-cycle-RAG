DEFINE TABLE IF NOT EXISTS web_search SCHEMAFULL;
DEFINE FIELD content ON web_search TYPE string;
DEFINE FIELD metadata ON web_search FLEXIBLE TYPE object;
DEFINE FIELD embedding ON web_search TYPE array<float>;
