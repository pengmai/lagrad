add_mlir_dialect_library(MLIRStandalone
        StandaloneDialect.cpp
        StandaloneOps.cpp
        StandaloneTransforms.cpp
        LowerToLLVM.cpp
        LowerGrad.cpp
        LinalgKnownLibraryCalls.cpp
        Bufferize.cpp
        TensorConstantFolding.cpp
        ElementwiseToAffine.cpp
        TriangularLoops.cpp
        TriangularPacking.cpp
        StaticAllocs.cpp
        ActivityAnalysis.cpp
        DynamicShapeAnalysis.cpp
        DeadCodeElimination.cpp
        LoopAllocHoisting.cpp
        LinalgCanonicalize.cpp
        PrimalCaching.cpp
        LinalgVectorJacobianProducts.cpp
        SCFVectorJacobianProducts.cpp
        Utils.cpp
        DebugNames.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Standalone

        DEPENDS
        MLIRStandaloneOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
