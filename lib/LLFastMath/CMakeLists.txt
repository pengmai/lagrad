add_llvm_library(LLFastMath MODULE
  LLFastMath.cpp
  DEPENDS
  intrinsics_gen
  PLUGIN_TOOL
  opt
)

target_compile_features(LLFastMath PRIVATE cxx_range_for cxx_auto_type)

# LLVM is (typically) built with no C++ RTTI. We need to match that;
# otherwise, we'll get linker errors about missing RTTI data.
set_target_properties(LLFastMath PROPERTIES
  COMPILE_FLAGS "-fno-rtti"
)

# Get proper shared-library behavior (where symbols are not necessarily
# resolved when the shared library is linked) on OS X.
if(APPLE)
  set_target_properties(LLFastMath PROPERTIES
    LINK_FLAGS "-Wl,-flat_namespace -Wl,-undefined -Wl,suppress"
  )
endif(APPLE)
