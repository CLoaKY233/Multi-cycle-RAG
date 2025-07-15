DEFINE FUNCTION OVERWRITE fn::countdocs() -> int {
	count(SELECT * FROM documents);
} PERMISSIONS FULL
