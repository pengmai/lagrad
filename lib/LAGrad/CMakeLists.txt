add_mlir_dialect_library(MLIRLAGrad
        LAGradDialect.cpp
        LAGradOps.cpp
        LAGradTransforms.cpp
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
        InsertExtractAnalysis.cpp
        DynamicShapeAnalysis.cpp
        DeadCodeElimination.cpp
        LoopAllocHoisting.cpp
        LinalgCanonicalize.cpp
        PrimalCaching.cpp
        LinalgVectorJacobianProducts.cpp
        SCFVectorJacobianProducts.cpp
        SparsePropagation.cpp
        LoopNestAnalysis.cpp
        Utils.cpp
        DebugNames.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/LAGrad

        DEPENDS
        MLIRLAGradOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
