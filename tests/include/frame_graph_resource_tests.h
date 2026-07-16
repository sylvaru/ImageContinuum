#pragma once

// Proves the item-1 resource model: per-frame-slot / history instance selection
// math and the compiler's version-aware (read-previous / write-current)
// dependency and barrier handling. Returns zero on success; nonzero on failure.
int runFrameGraphResourceTests();
