// ic/common/ic_common.h
#pragma once
#include <memory>

namespace ic
{
    
    template<typename T>
    using Scope = std::unique_ptr<T>;

    template<typename T>
    using Ref = std::shared_ptr<T>;

} 