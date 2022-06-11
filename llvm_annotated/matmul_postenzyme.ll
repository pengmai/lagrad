; ModuleID = '<stdin>'
source_filename = "-"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

@.memset_pattern = private unnamed_addr constant [2 x double] [double 1.000000e+00, double 1.000000e+00], align 16

; Function Attrs: nofree norecurse nounwind ssp uwtable
define dso_local void @ematmul(i64 %N, double* nocapture readonly %A, double* nocapture readonly %B, double* nocapture %out) local_unnamed_addr #0 {
entry:
  %cmp45 = icmp sgt i64 %N, 0
  br i1 %cmp45, label %for.cond5.preheader.lr.ph, label %for.cond.cleanup

for.cond5.preheader.lr.ph:                        ; preds = %entry, %for.cond.cleanup3
  %i.046 = phi i64 [ %inc21, %for.cond.cleanup3 ], [ 0, %entry ]
  %mul = mul nsw i64 %i.046, %N
  br label %for.body8.lr.ph

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3, %entry
  ret void

for.body8.lr.ph:                                  ; preds = %for.cond.cleanup7, %for.cond5.preheader.lr.ph
  %j.043 = phi i64 [ 0, %for.cond5.preheader.lr.ph ], [ %inc18, %for.cond.cleanup7 ]
  %add14 = add nsw i64 %j.043, %mul
  %arrayidx15 = getelementptr inbounds double, double* %out, i64 %add14
  %.pre = load double, double* %arrayidx15, align 8, !tbaa !3
  br label %for.body8

for.cond.cleanup3:                                ; preds = %for.cond.cleanup7
  %inc21 = add nuw nsw i64 %i.046, 1
  %exitcond49.not = icmp eq i64 %inc21, %N
  br i1 %exitcond49.not, label %for.cond.cleanup, label %for.cond5.preheader.lr.ph, !llvm.loop !7

for.cond.cleanup7:                                ; preds = %for.body8
  %inc18 = add nuw nsw i64 %j.043, 1
  %exitcond48.not = icmp eq i64 %inc18, %N
  br i1 %exitcond48.not, label %for.cond.cleanup3, label %for.body8.lr.ph, !llvm.loop !10

for.body8:                                        ; preds = %for.body8, %for.body8.lr.ph
  %0 = phi double [ %.pre, %for.body8.lr.ph ], [ %add16, %for.body8 ]
  %k.041 = phi i64 [ 0, %for.body8.lr.ph ], [ %inc, %for.body8 ]
  %add = add nsw i64 %k.041, %mul
  %arrayidx = getelementptr inbounds double, double* %A, i64 %add
  %1 = load double, double* %arrayidx, align 8, !tbaa !3
  %mul9 = mul nsw i64 %k.041, %N
  %add10 = add nsw i64 %mul9, %j.043
  %arrayidx11 = getelementptr inbounds double, double* %B, i64 %add10
  %2 = load double, double* %arrayidx11, align 8, !tbaa !3
  %mul12 = fmul double %1, %2
  %add16 = fadd double %0, %mul12
  store double %add16, double* %arrayidx15, align 8, !tbaa !3
  %inc = add nuw nsw i64 %k.041, 1
  %exitcond.not = icmp eq i64 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup7, label %for.body8, !llvm.loop !11
}

; Function Attrs: nounwind ssp uwtable
define dso_local noalias double* @enzyme_c_matmul(i64 %N, double* nocapture readnone %A, double* nocapture readonly %B) local_unnamed_addr #1 {
entry:
  %mul = mul i64 %N, %N
  %mul1 = shl i64 %mul, 3
  ;; %call is the gradient dA
  %call = tail call i8* @malloc(i64 %mul1) #6
  ;; %call7 is the gradient signal, 1.0
  %call7 = tail call i8* @malloc(i64 %mul1) #6
  %cmp30.not = icmp eq i64 %mul, 0
  br i1 %cmp30.not, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  ;; dA and g are initialized here.
  tail call void @llvm.memset.p0i8.i64(i8* align 8 %call, i8 0, i64 %mul1, i1 false)
  tail call void @memset_pattern16(i8* %call7, i8* bitcast ([2 x double]* @.memset_pattern to i8*), i64 %mul1) #7
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body.preheader, %entry
  %0 = bitcast i8* %call to double*
  %1 = bitcast i8* %call7 to double*
  %cmp45.i = icmp sgt i64 %N, 0
  br i1 %cmp45.i, label %for.cond5.preheader.lr.ph.preheader.i, label %diffeematmul.exit

for.cond5.preheader.lr.ph.preheader.i:            ; preds = %for.cond.cleanup
  ;; What is this?
  %2 = shl i64 %N, 3
  ;; N * N * N bytes
  %mallocsize.i = mul i64 %2, %mul
  %malloccall.i = tail call noalias nonnull i8* @malloc(i64 %mallocsize.i) #7
  %_malloccache.i = bitcast i8* %malloccall.i to double*
  br label %for.cond5.preheader.lr.ph.i

for.cond5.preheader.lr.ph.i:                      ; preds = %for.cond.cleanup3.i, %for.cond5.preheader.lr.ph.preheader.i
  %iv.i = phi i64 [ %iv.next.i, %for.cond.cleanup3.i ], [ 0, %for.cond5.preheader.lr.ph.preheader.i ]
  %3 = mul nuw nsw i64 %iv.i, %mul
  br label %for.body8.lr.ph.i

for.body8.lr.ph.i:                                ; preds = %for.cond.cleanup7.i, %for.cond5.preheader.lr.ph.i
  %iv1.i = phi i64 [ %iv.next2.i, %for.cond.cleanup7.i ], [ 0, %for.cond5.preheader.lr.ph.i ]
  %4 = mul nuw nsw i64 %iv1.i, %N
  %5 = add i64 %4, %3
  br label %for.body8.i

for.cond.cleanup3.i:                              ; preds = %for.cond.cleanup7.i
  %iv.next.i = add nuw nsw i64 %iv.i, 1
  %exitcond49.not.i = icmp eq i64 %iv.next.i, %N
  br i1 %exitcond49.not.i, label %invertfor.cond.cleanup3.i.preheader, label %for.cond5.preheader.lr.ph.i, !llvm.loop !7

invertfor.cond.cleanup3.i.preheader:              ; preds = %for.cond.cleanup3.i
  %"iv3'ac.0.i.peel" = add i64 %N, -1
  %6 = icmp eq i64 %"iv3'ac.0.i.peel", 0
  %min.iters.check = icmp ult i64 %"iv3'ac.0.i.peel", 4
  %n.vec = and i64 %"iv3'ac.0.i.peel", -4
  %ind.end = and i64 %"iv3'ac.0.i.peel", 3
  %cmp.n = icmp eq i64 %"iv3'ac.0.i.peel", %n.vec
  br label %invertfor.cond.cleanup3.i

for.cond.cleanup7.i:                              ; preds = %for.body8.i
  %iv.next2.i = add nuw nsw i64 %iv1.i, 1
  %exitcond48.not.i = icmp eq i64 %iv.next2.i, %N
  br i1 %exitcond48.not.i, label %for.cond.cleanup3.i, label %for.body8.lr.ph.i, !llvm.loop !10

for.body8.i:                                      ; preds = %for.body8.i, %for.body8.lr.ph.i
  ;; This loop goes from 0 to N
  %iv3.i = phi i64 [ %iv.next4.i, %for.body8.i ], [ 0, %for.body8.lr.ph.i ]
  %iv.next4.i = add nuw nsw i64 %iv3.i, 1
  %mul9.i = mul nsw i64 %iv3.i, %N
  %add10.i = add nsw i64 %mul9.i, %iv1.i
  %arrayidx11.i = getelementptr inbounds double, double* %B, i64 %add10.i
  %7 = load double, double* %arrayidx11.i, align 8, !tbaa !3
  %8 = add i64 %5, %iv3.i
  %9 = getelementptr inbounds double, double* %_malloccache.i, i64 %8
  store double %7, double* %9, align 8, !invariant.group !12
  %exitcond.not.i = icmp eq i64 %iv.next4.i, %N
  br i1 %exitcond.not.i, label %for.cond.cleanup7.i, label %for.body8.i, !llvm.loop !11

invertfor.cond5.preheader.lr.ph.preheader.i:      ; preds = %invertfor.cond5.preheader.lr.ph.i
  tail call void @free(i8* nonnull %malloccall.i) #7
  br label %diffeematmul.exit

invertfor.cond5.preheader.lr.ph.i:                ; preds = %invertfor.body8.lr.ph.i
  %10 = icmp eq i64 %"iv'ac.0.i", 0
  br i1 %10, label %invertfor.cond5.preheader.lr.ph.preheader.i, label %invertfor.cond.cleanup3.i

invertfor.body8.lr.ph.i:                          ; preds = %invertfor.body8.i, %middle.block, %invertfor.cond.cleanup7.i
  store double %"arrayidx15'ipg_unwrap9.i.promoted", double* %"arrayidx15'ipg_unwrap9.i", align 8
  %11 = icmp eq i64 %"iv1'ac.0.i", 0
  br i1 %11, label %invertfor.cond5.preheader.lr.ph.i, label %invertfor.cond.cleanup7.i

invertfor.cond.cleanup3.i:                        ; preds = %invertfor.cond.cleanup3.i.preheader, %invertfor.cond5.preheader.lr.ph.i
  %"iv'ac.0.in.i" = phi i64 [ %"iv'ac.0.i", %invertfor.cond5.preheader.lr.ph.i ], [ %N, %invertfor.cond.cleanup3.i.preheader ]
  %"iv'ac.0.i" = add i64 %"iv'ac.0.in.i", -1
  %mul_unwrap7.i = mul i64 %"iv'ac.0.i", %N
  %add_unwrap.i.peel = add nsw i64 %"iv3'ac.0.i.peel", %mul_unwrap7.i
  %"arrayidx'ipg_unwrap.i.peel" = getelementptr inbounds double, double* %0, i64 %add_unwrap.i.peel
  br label %invertfor.cond.cleanup7.i

invertfor.cond.cleanup7.i:                        ; preds = %invertfor.cond.cleanup3.i, %invertfor.body8.lr.ph.i
  %"iv1'ac.0.in.i" = phi i64 [ %N, %invertfor.cond.cleanup3.i ], [ %"iv1'ac.0.i", %invertfor.body8.lr.ph.i ]
  %"iv1'ac.0.i" = add i64 %"iv1'ac.0.in.i", -1
  %add14_unwrap8.i = add i64 %"iv1'ac.0.i", %mul_unwrap7.i
  %"arrayidx15'ipg_unwrap9.i" = getelementptr inbounds double, double* %1, i64 %add14_unwrap8.i
  %reass.mul.i = mul i64 %add14_unwrap8.i, %N
  %"arrayidx15'ipg_unwrap9.i.promoted" = load double, double* %"arrayidx15'ipg_unwrap9.i", align 8
  %12 = add i64 %"iv3'ac.0.i.peel", %reass.mul.i
  %13 = getelementptr inbounds double, double* %_malloccache.i, i64 %12
  %14 = load double, double* %13, align 8, !invariant.group !12
  %m0diffe.i.peel = fmul fast double %14, %"arrayidx15'ipg_unwrap9.i.promoted"
  %15 = load double, double* %"arrayidx'ipg_unwrap.i.peel", align 8
  %16 = fadd fast double %15, %m0diffe.i.peel
  store double %16, double* %"arrayidx'ipg_unwrap.i.peel", align 8
  br i1 %6, label %invertfor.body8.lr.ph.i, label %invertfor.body8.i.preheader

invertfor.body8.i.preheader:                      ; preds = %invertfor.cond.cleanup7.i
  br i1 %min.iters.check, label %invertfor.body8.i.preheader12, label %vector.ph

vector.ph:                                        ; preds = %invertfor.body8.i.preheader
  %broadcast.splatinsert = insertelement <2 x double> poison, double %"arrayidx15'ipg_unwrap9.i.promoted", i32 0
  %broadcast.splat = shufflevector <2 x double> %broadcast.splatinsert, <2 x double> poison, <2 x i32> zeroinitializer
  %broadcast.splatinsert4 = insertelement <2 x double> poison, double %"arrayidx15'ipg_unwrap9.i.promoted", i32 0
  %broadcast.splat5 = shufflevector <2 x double> %broadcast.splatinsert4, <2 x double> poison, <2 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %17 = xor i64 %index, -1
  %18 = add i64 %"iv3'ac.0.i.peel", %17
  %19 = add i64 %18, %reass.mul.i
  %20 = getelementptr inbounds double, double* %_malloccache.i, i64 %19
  %21 = getelementptr inbounds double, double* %20, i64 -1
  %22 = bitcast double* %21 to <2 x double>*
  %wide.load = load <2 x double>, <2 x double>* %22, align 8
  %reverse = shufflevector <2 x double> %wide.load, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  %23 = getelementptr inbounds double, double* %20, i64 -2
  %24 = getelementptr inbounds double, double* %23, i64 -1
  %25 = bitcast double* %24 to <2 x double>*
  %wide.load2 = load <2 x double>, <2 x double>* %25, align 8
  %reverse3 = shufflevector <2 x double> %wide.load2, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  %26 = fmul fast <2 x double> %reverse, %broadcast.splat
  %27 = fmul fast <2 x double> %reverse3, %broadcast.splat5
  %28 = add nsw i64 %18, %mul_unwrap7.i
  %29 = getelementptr inbounds double, double* %0, i64 %28
  %30 = getelementptr inbounds double, double* %29, i64 -1
  %31 = bitcast double* %30 to <2 x double>*
  %wide.load6 = load <2 x double>, <2 x double>* %31, align 8
  %reverse7 = shufflevector <2 x double> %wide.load6, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  %32 = getelementptr inbounds double, double* %29, i64 -2
  %33 = getelementptr inbounds double, double* %32, i64 -1
  %34 = bitcast double* %33 to <2 x double>*
  %wide.load8 = load <2 x double>, <2 x double>* %34, align 8
  %reverse9 = shufflevector <2 x double> %wide.load8, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  %35 = fadd fast <2 x double> %reverse7, %26
  %36 = fadd fast <2 x double> %reverse9, %27
  %reverse10 = shufflevector <2 x double> %35, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  %37 = bitcast double* %30 to <2 x double>*
  store <2 x double> %reverse10, <2 x double>* %37, align 8
  %reverse11 = shufflevector <2 x double> %36, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  %38 = bitcast double* %33 to <2 x double>*
  store <2 x double> %reverse11, <2 x double>* %38, align 8
  %index.next = add i64 %index, 4
  %39 = icmp eq i64 %index.next, %n.vec
  br i1 %39, label %middle.block, label %vector.body, !llvm.loop !13

middle.block:                                     ; preds = %vector.body
  br i1 %cmp.n, label %invertfor.body8.lr.ph.i, label %invertfor.body8.i.preheader12

invertfor.body8.i.preheader12:                    ; preds = %invertfor.body8.i.preheader, %middle.block
  %"iv3'ac.0.in.i.ph" = phi i64 [ %"iv3'ac.0.i.peel", %invertfor.body8.i.preheader ], [ %ind.end, %middle.block ]
  br label %invertfor.body8.i

invertfor.body8.i:                                ; preds = %invertfor.body8.i.preheader12, %invertfor.body8.i
  %"iv3'ac.0.in.i" = phi i64 [ %"iv3'ac.0.i", %invertfor.body8.i ], [ %"iv3'ac.0.in.i.ph", %invertfor.body8.i.preheader12 ]
  %"iv3'ac.0.i" = add i64 %"iv3'ac.0.in.i", -1
  %40 = add i64 %"iv3'ac.0.i", %reass.mul.i
  %41 = getelementptr inbounds double, double* %_malloccache.i, i64 %40
  %42 = load double, double* %41, align 8, !invariant.group !12
  %m0diffe.i = fmul fast double %42, %"arrayidx15'ipg_unwrap9.i.promoted"
  %add_unwrap.i = add nsw i64 %"iv3'ac.0.i", %mul_unwrap7.i
  %"arrayidx'ipg_unwrap.i" = getelementptr inbounds double, double* %0, i64 %add_unwrap.i
  %43 = load double, double* %"arrayidx'ipg_unwrap.i", align 8
  %44 = fadd fast double %43, %m0diffe.i
  store double %44, double* %"arrayidx'ipg_unwrap.i", align 8
  %45 = icmp eq i64 %"iv3'ac.0.i", 0
  br i1 %45, label %invertfor.body8.lr.ph.i, label %invertfor.body8.i, !llvm.loop !16

diffeematmul.exit:                                ; preds = %for.cond.cleanup, %invertfor.cond5.preheader.lr.ph.preheader.i
  tail call void @free(i8* %call7)
  ret double* %0
}

; Function Attrs: inaccessiblememonly nofree nounwind willreturn allocsize(0)
declare noalias noundef i8* @malloc(i64) local_unnamed_addr #2

; Function Attrs: inaccessiblemem_or_argmemonly nounwind willreturn
declare void @free(i8* nocapture noundef) local_unnamed_addr #3

; Function Attrs: argmemonly nofree nosync nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #4

; Function Attrs: argmemonly nofree
declare void @memset_pattern16(i8* nocapture writeonly, i8* nocapture readonly, i64) local_unnamed_addr #5

; Function Attrs: nofree norecurse nounwind ssp uwtable
define dso_local void @preprocess_ematmul(i64 %N, double* nocapture readonly %A, double* nocapture readonly %B, double* nocapture %out) local_unnamed_addr #0 {
entry:
  %cmp45 = icmp sgt i64 %N, 0
  br i1 %cmp45, label %for.cond5.preheader.lr.ph, label %for.cond.cleanup

for.cond5.preheader.lr.ph:                        ; preds = %entry, %for.cond.cleanup3
  %tiv = phi i64 [ %tiv.next, %for.cond.cleanup3 ], [ 0, %entry ]
  %mul = mul nsw i64 %tiv, %N
  br label %for.body8.lr.ph

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3, %entry
  ret void

for.body8.lr.ph:                                  ; preds = %for.cond.cleanup7, %for.cond5.preheader.lr.ph
  %j.043 = phi i64 [ 0, %for.cond5.preheader.lr.ph ], [ %inc18, %for.cond.cleanup7 ]
  %add14 = add nsw i64 %j.043, %mul
  %arrayidx15 = getelementptr inbounds double, double* %out, i64 %add14
  %.pre = load double, double* %arrayidx15, align 8, !tbaa !3
  br label %for.body8

for.cond.cleanup3:                                ; preds = %for.cond.cleanup7
  %tiv.next = add nuw nsw i64 %tiv, 1
  %exitcond49.not = icmp eq i64 %tiv.next, %N
  br i1 %exitcond49.not, label %for.cond.cleanup, label %for.cond5.preheader.lr.ph, !llvm.loop !7

for.cond.cleanup7:                                ; preds = %for.body8
  %inc18 = add nuw nsw i64 %j.043, 1
  %exitcond48.not = icmp eq i64 %inc18, %N
  br i1 %exitcond48.not, label %for.cond.cleanup3, label %for.body8.lr.ph, !llvm.loop !10

for.body8:                                        ; preds = %for.body8, %for.body8.lr.ph
  %0 = phi double [ %.pre, %for.body8.lr.ph ], [ %add16, %for.body8 ]
  %k.041 = phi i64 [ 0, %for.body8.lr.ph ], [ %inc, %for.body8 ]
  %add = add nsw i64 %k.041, %mul
  %arrayidx = getelementptr inbounds double, double* %A, i64 %add
  %1 = load double, double* %arrayidx, align 8, !tbaa !3
  %mul9 = mul nsw i64 %k.041, %N
  %add10 = add nsw i64 %mul9, %j.043
  %arrayidx11 = getelementptr inbounds double, double* %B, i64 %add10
  %2 = load double, double* %arrayidx11, align 8, !tbaa !3
  %mul12 = fmul double %1, %2
  %add16 = fadd double %0, %mul12
  store double %add16, double* %arrayidx15, align 8, !tbaa !3
  %inc = add nuw nsw i64 %k.041, 1
  %exitcond.not = icmp eq i64 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup7, label %for.body8, !llvm.loop !11
}

attributes #0 = { nofree norecurse nounwind ssp uwtable "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind ssp uwtable "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { inaccessiblememonly nofree nounwind willreturn allocsize(0) "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { inaccessiblemem_or_argmemonly nounwind willreturn "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { argmemonly nofree nosync nounwind willreturn writeonly }
attributes #5 = { argmemonly nofree }
attributes #6 = { allocsize(0) }
attributes #7 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{!"clang version 12.0.1 (https://github.com/llvm/llvm-project.git fed41342a82f5a3a9201819a82bf7a48313e296b)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"double", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = distinct !{!7, !8, !9}
!8 = !{!"llvm.loop.mustprogress"}
!9 = !{!"llvm.loop.unroll.disable"}
!10 = distinct !{!10, !8, !9}
!11 = distinct !{!11, !8, !9}
!12 = distinct !{}
!13 = distinct !{!13, !14, !15}
!14 = !{!"llvm.loop.peeled.count", i32 1}
!15 = !{!"llvm.loop.isvectorized", i32 1}
!16 = distinct !{!16, !14, !17, !15}
!17 = !{!"llvm.loop.unroll.runtime.disable"}
