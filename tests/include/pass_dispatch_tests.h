#pragma once

// Proves the item-3 pass-dispatch spine contract: the PassPayload variant maps
// exhaustively (compile-time, no catch-all) onto a per-payload recorder kind,
// mirroring the routing both backends' dispatchNode() std::visits onto their
// recordPassPayload overload sets. Returns zero on success; nonzero on failure.
int runPassDispatchTests();
