DEFINE FUNCTION OVERWRITE fn::count_web() -> int {
	count(SELECT * FROM web_search);
} PERMISSIONS FULL
