#pragma once

// Runs the deterministic job-system and coroutine stress suite.
// Returns zero on success; invariant failures abort with a diagnostic.
int runJobSystemStressTests();
