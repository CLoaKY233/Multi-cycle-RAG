-- Define documents table
DEFINE TABLE IF NOT EXISTS documents SCHEMAFULL;
DEFINE FIELD content ON documents TYPE string;
DEFINE FIELD metadata ON documents FLEXIBLE TYPE object;
DEFINE FIELD embedding ON documents TYPE array<float>;
