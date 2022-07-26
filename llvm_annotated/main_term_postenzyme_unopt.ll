; ModuleID = '<stdin>'
source_filename = "-"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

@enzyme_const = external local_unnamed_addr global i32, align 4
@.mystr = private unnamed_addr constant [13 x i8] c"index: %lld\0A\00", align 1
@.flstr = private unnamed_addr constant [13 x i8] c"debug: %.4e\0A\00", align 1
@.354str = private unnamed_addr constant [13 x i8] c"ll354: %.4e\0A\00", align 1

; Function Attrs: nofree nounwind
declare noundef i32 @printf(i8* nocapture noundef readonly, ...) local_unnamed_addr #1

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare double @llvm.exp.f64(double) #4

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare double @llvm.log.f64(double) #4

; Function Attrs: inaccessiblememonly nofree nounwind willreturn allocsize(0)
declare noalias noundef i8* @malloc(i64) local_unnamed_addr #7

; Function Attrs: inaccessiblemem_or_argmemonly nounwind willreturn
declare void @free(i8* nocapture noundef) local_unnamed_addr #8

; Function Attrs: nounwind ssp uwtable
define dso_local void @enzyme_c_main_term(i32 %d, i32 %k, i32 %n, double* %alphas, double* %alphasb, double* %means, double* %meansb, double* %Qs, double* %Qsb, double* %Ls, double* %Lsb, double* %x) local_unnamed_addr #6 {
entry:
  %out = alloca double, align 8
  %dout = alloca double, align 8
  %0 = bitcast double* %out to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #14
  store double 0.000000e+00, double* %out, align 8, !tbaa !3
  %1 = bitcast double* %dout to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1) #14
  store double 1.000000e+00, double* %dout, align 8, !tbaa !3
  %2 = load i32, i32* @enzyme_const, align 4, !tbaa !20
  call void @diffeec_main_term(i32 %d, i32 %k, i32 %n, double* %alphas, double* %alphasb, double* %means, double* %meansb, double* %Qs, double* %Qsb, double* %Ls, double* %Lsb, double* %x, double* %out, double* %dout)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1) #14
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #14
  ret void
}

declare void @__enzyme_autodiff(i8*, ...) local_unnamed_addr #9

; Function Attrs: inaccessiblememonly nofree nounwind willreturn allocsize(0,1)
declare noalias noundef i8* @calloc(i64, i64) local_unnamed_addr #10

; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn
declare void @llvm.experimental.noalias.scope.decl(metadata) #11

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #1

; Function Attrs: nounwind ssp uwtable
define internal void @diffeec_main_term(i32 %d, i32 %k, i32 %n, double* noalias nocapture readonly %alphas, double* nocapture %"alphas'", double* noalias nocapture readonly %means, double* nocapture %"means'", double* noalias nocapture readonly %Qs, double* nocapture %"Qs'", double* noalias nocapture readonly %Ls, double* nocapture %"Ls'", double* noalias nocapture readonly %x, double* nocapture %err, double* nocapture %"err'") #6 {
entry:
  %"iv'ac" = alloca i64, align 8
  %"iv1'ac" = alloca i64, align 8
  %"iv3'ac" = alloca i64, align 8
  %"iv5'ac" = alloca i64, align 8
  %"iv7'ac" = alloca i64, align 8
  %"iv9'ac" = alloca i64, align 8
  %"iv11'ac" = alloca i64, align 8
  %"iv13'ac" = alloca i64, align 8
  %"iv15'ac" = alloca i64, align 8
  %"iv17'ac" = alloca i64, align 8
  %"iv19'ac" = alloca i64, align 8
  %"'de" = alloca double, align 8
  store double 0.000000e+00, double* %"'de", align 8
  %"'de25" = alloca double, align 8
  store double 0.000000e+00, double* %"'de25", align 8
  %"add8.i'de" = alloca double, align 8
  store double 0.000000e+00, double* %"add8.i'de", align 8
  %"'de26" = alloca double, align 8
  store double 0.000000e+00, double* %"'de26", align 8
  %"'de30" = alloca double, align 8
  store double 0.000000e+00, double* %"'de30", align 8
  %"'de31" = alloca double, align 8
  store double 0.000000e+00, double* %"'de31", align 8
  %"'de32" = alloca double, align 8
  store double 0.000000e+00, double* %"'de32", align 8
  %"'de33" = alloca double, align 8
  store double 0.000000e+00, double* %"'de33", align 8
  %"slse.0143'de" = alloca double, align 8
  store double 0.000000e+00, double* %"slse.0143'de", align 8
  %"add47'de" = alloca double, align 8
  store double 0.000000e+00, double* %"add47'de", align 8
  %"'de34" = alloca double, align 8
  store double 0.000000e+00, double* %"'de34", align 8
  %"'de35" = alloca double, align 8
  store double 0.000000e+00, double* %"'de35", align 8
  %"sub.i124'de" = alloca double, align 8
  store double 0.000000e+00, double* %"sub.i124'de", align 8
  %"'de38" = alloca double, align 8
  store double 0.000000e+00, double* %"'de38", align 8
  %"mul.i113'de" = alloca double, align 8
  store double 0.000000e+00, double* %"mul.i113'de", align 8
  %_cache = alloca double*, align 8
  %"'de50" = alloca double, align 8
  store double 0.000000e+00, double* %"'de50", align 8
  %"'de51" = alloca double, align 8
  store double 0.000000e+00, double* %"'de51", align 8
  %"'de59" = alloca double, align 8
  store double 0.000000e+00, double* %"'de59", align 8
  %"add21.i'de" = alloca double, align 8
  store double 0.000000e+00, double* %"add21.i'de", align 8
  %"'de62" = alloca double, align 8
  store double 0.000000e+00, double* %"'de62", align 8
  %"mul20.i'de" = alloca double, align 8
  store double 0.000000e+00, double* %"mul20.i'de", align 8
  %"'de79" = alloca double, align 8
  store double 0.000000e+00, double* %"'de79", align 8
  %".pre'de" = alloca double, align 8
  store double 0.000000e+00, double* %".pre'de", align 8
  %"mul.i102'de" = alloca double, align 8
  store double 0.000000e+00, double* %"mul.i102'de", align 8
  %_cache82 = alloca double*, align 8
  %"add'de" = alloca double, align 8
  store double 0.000000e+00, double* %"add'de", align 8
  %"'de93" = alloca double, align 8
  store double 0.000000e+00, double* %"'de93", align 8
  %"'de94" = alloca double, align 8
  store double 0.000000e+00, double* %"'de94", align 8
  %"add.i106'de" = alloca double, align 8
  store double 0.000000e+00, double* %"add.i106'de", align 8
  %"res.017.i'de" = alloca double, align 8
  store double 0.000000e+00, double* %"res.017.i'de", align 8
  %"mul5.i'de" = alloca double, align 8
  store double 0.000000e+00, double* %"mul5.i'de", align 8
  %_cache98 = alloca double*, align 8
  %"'de112" = alloca double, align 8
  store double 0.000000e+00, double* %"'de112", align 8
  %"sub43'de" = alloca double, align 8
  store double 0.000000e+00, double* %"sub43'de", align 8
  %"mul42'de" = alloca double, align 8
  store double 0.000000e+00, double* %"mul42'de", align 8
  %"res.0.lcssa.i'de" = alloca double, align 8
  store double 0.000000e+00, double* %"res.0.lcssa.i'de", align 8
  %".pre146'de" = alloca double, align 8
  store double 0.000000e+00, double* %".pre146'de", align 8
  %"cmp2.i.i!manual_lcssa_cache" = alloca i1*, align 8
  %"!manual_lcssa126_cache" = alloca i64*, align 8
  %"m.1.i.i'de" = alloca double, align 8
  store double 0.000000e+00, double* %"m.1.i.i'de", align 8
  %"'de131" = alloca double, align 8
  store double 0.000000e+00, double* %"'de131", align 8
  %"m.015.i.i'de" = alloca double, align 8
  store double 0.000000e+00, double* %"m.015.i.i'de", align 8
  %"m.0.lcssa.i.i'de" = alloca double, align 8
  store double 0.000000e+00, double* %"m.0.lcssa.i.i'de", align 8
  %"add.i136'de" = alloca double, align 8
  store double 0.000000e+00, double* %"add.i136'de", align 8
  %"'de133" = alloca double, align 8
  store double 0.000000e+00, double* %"'de133", align 8
  %sub.i135_cache = alloca double*, align 8
  %"sub.i135'de" = alloca double, align 8
  store double 0.000000e+00, double* %"sub.i135'de", align 8
  %"add.i'de" = alloca double, align 8
  store double 0.000000e+00, double* %"add.i'de", align 8
  %"add.i138'de" = alloca double, align 8
  store double 0.000000e+00, double* %"add.i138'de", align 8
  %"'de139" = alloca double, align 8
  store double 0.000000e+00, double* %"'de139", align 8
  %sub.i_cache = alloca double*, align 8
  %"sub.i'de" = alloca double, align 8
  store double 0.000000e+00, double* %"sub.i'de", align 8
  %".pre.i101'de" = alloca double, align 8
  store double 0.000000e+00, double* %".pre.i101'de", align 8
  %"add1.i'de" = alloca double, align 8
  store double 0.000000e+00, double* %"add1.i'de", align 8
  %"'de150" = alloca double, align 8
  store double 0.000000e+00, double* %"'de150", align 8
  %"add.i!manual_lcssa154_cache" = alloca double*, align 8
  %"semx.0.lcssa.i'de" = alloca double, align 8
  store double 0.000000e+00, double* %"semx.0.lcssa.i'de", align 8
  %"slse.0.lcssa'de" = alloca double, align 8
  store double 0.000000e+00, double* %"slse.0.lcssa'de", align 8
  %sub = add nsw i32 %d, -1
  %mul = mul nsw i32 %sub, %d
  %div = sdiv i32 %mul, 2
  %conv = sext i32 %div to i64
  %mul1 = mul nsw i32 %k, %d
  %conv2 = sext i32 %mul1 to i64
  %mul3 = shl nsw i64 %conv2, 3
  %call = tail call i8* @malloc(i64 %mul3) #13
  %"call'mi" = tail call noalias nonnull i8* @malloc(i64 %mul3) #13
  call void @llvm.memset.p0i8.i64(i8* nonnull %"call'mi", i8 0, i64 %mul3, i1 false)
  %"'ipc21" = bitcast i8* %"call'mi" to double*
  %0 = bitcast i8* %call to double*
  %conv4 = sext i32 %k to i64
  %mul5 = shl nsw i64 %conv4, 3
  %call6 = tail call i8* @malloc(i64 %mul5) #13
  %"call6'mi" = tail call noalias nonnull i8* @malloc(i64 %mul5) #13
  call void @llvm.memset.p0i8.i64(i8* nonnull %"call6'mi", i8 0, i64 %mul5, i1 false)
  %"'ipc" = bitcast i8* %"call6'mi" to double*
  %1 = bitcast i8* %call6 to double*
  %conv7 = sext i32 %d to i64
  %mul8 = shl nsw i64 %conv7, 3
  %call9 = tail call i8* @malloc(i64 %mul8) #13
  %call941 = bitcast i8* %call9 to double*
  %"call9'mi" = tail call noalias nonnull i8* @malloc(i64 %mul8) #13
  call void @llvm.memset.p0i8.i64(i8* nonnull %"call9'mi", i8 0, i64 %mul8, i1 false)
  %"'ipc37" = bitcast i8* %"call9'mi" to double*
  %2 = bitcast i8* %call9 to double*
  %call12 = tail call i8* @malloc(i64 %mul8) #13
  %"call12'mi" = tail call noalias nonnull i8* @malloc(i64 %mul8) #13
  call void @llvm.memset.p0i8.i64(i8* nonnull %"call12'mi", i8 0, i64 %mul8, i1 false)
  %"'ipc40" = bitcast i8* %"call12'mi" to double*
  %3 = bitcast i8* %call12 to double*
  %call15 = tail call i8* @malloc(i64 %mul5) #13
  %"call15'mi" = tail call noalias nonnull i8* @malloc(i64 %mul5) #13
  call void @llvm.memset.p0i8.i64(i8* nonnull %"call15'mi", i8 0, i64 %mul5, i1 false)
  %"'ipc116" = bitcast i8* %"call15'mi" to double*
  %4 = bitcast i8* %call15 to double*
  ;; always true (k > 0)
  %cmp37.i = icmp sgt i32 %k, 0
  br i1 %cmp37.i, label %for.body.lr.ph.i, label %preprocess_qs.exit

for.body.lr.ph.i:                                 ; preds = %entry
  %cmp235.i = icmp sgt i32 %d, 0
  %wide.trip.count44.i = zext i32 %k to i64
  %wide.trip.count.i = zext i32 %d to i64
  %5 = add nsw i64 %wide.trip.count44.i, -1
  %6 = add nsw i64 %wide.trip.count.i, -1
  br label %for.body.i

for.body.i:                                       ; preds = %for.inc15.i, %for.body.lr.ph.i
  %iv = phi i64 [ %iv.next, %for.inc15.i ], [ 0, %for.body.lr.ph.i ]
  %iv.next = add nuw nsw i64 %iv, 1
  %"arrayidx.i'ipg" = getelementptr inbounds double, double* %"'ipc", i64 %iv
  %arrayidx.i = getelementptr inbounds double, double* %1, i64 %iv
  store double 0.000000e+00, double* %arrayidx.i, align 8, !tbaa !3
  br i1 %cmp235.i, label %for.body3.lr.ph.i, label %for.inc15.i

for.body3.lr.ph.i:                                ; preds = %for.body.i
  %7 = trunc i64 %iv to i32
  %mul.i = mul nsw i32 %7, %d
  %8 = sext i32 %mul.i to i64
  br label %for.body3.i

for.body3.i:                                      ; preds = %for.body3.i, %for.body3.lr.ph.i
  %iv1 = phi i64 [ %iv.next2, %for.body3.i ], [ 0, %for.body3.lr.ph.i ]
  %9 = phi double [ 0.000000e+00, %for.body3.lr.ph.i ], [ %add8.i, %for.body3.i ]
  %iv.next2 = add nuw nsw i64 %iv1, 1
  %10 = add nsw i64 %iv1, %8
  %"arrayidx5.i'ipg" = getelementptr inbounds double, double* %"Qs'", i64 %10
  %arrayidx5.i = getelementptr inbounds double, double* %Qs, i64 %10
  %11 = load double, double* %arrayidx5.i, align 8, !tbaa !3, !invariant.group !42
  %add8.i = fadd double %9, %11
  %12 = tail call double @llvm.exp.f64(double %11) #14
  %"arrayidx14.i'ipg" = getelementptr inbounds double, double* %"'ipc21", i64 %10
  %arrayidx14.i = getelementptr inbounds double, double* %0, i64 %10
  store double %12, double* %arrayidx14.i, align 8, !tbaa !3
  %exitcond.not.i = icmp eq i64 %iv.next2, %wide.trip.count.i
  br i1 %exitcond.not.i, label %for.inc15.i.loopexit, label %for.body3.i, !llvm.loop !13

for.inc15.i.loopexit:                             ; preds = %for.body3.i
  store double %add8.i, double* %arrayidx.i, align 8, !tbaa !3
  br label %for.inc15.i

for.inc15.i:                                      ; preds = %for.inc15.i.loopexit, %for.body.i
  %exitcond45.not.i = icmp eq i64 %iv.next, %wide.trip.count44.i
  br i1 %exitcond45.not.i, label %preprocess_qs.exit.loopexit, label %for.body.i, !llvm.loop !14

preprocess_qs.exit.loopexit:                      ; preds = %for.inc15.i
  br label %preprocess_qs.exit

preprocess_qs.exit:                               ; preds = %preprocess_qs.exit.loopexit, %entry
  %conv17 = sext i32 %n to i64
  %cmp140 = icmp sgt i32 %n, 0
  br i1 %cmp140, label %for.cond19.preheader.lr.ph, label %for.end50

for.cond19.preheader.lr.ph:                       ; preds = %preprocess_qs.exit
  %cmp10.i = icmp sgt i32 %d, 0
  %wide.trip.count.i119 = zext i32 %d to i64
  %mul8.i = shl nuw nsw i32 %d, 1
  %cmp15.i = icmp sgt i32 %d, 1
  %cmp13.i.i = icmp sgt i32 %k, 1
  %wide.trip.count.i.i = zext i32 %k to i64
  %exitcond.not.i99137 = icmp eq i32 %k, 1
  ; n - 1
  %13 = add nsw i64 %conv17, -1
  ; k - 1
  %14 = add nsw i64 %conv4, -1
  ; d - 1, wide.trip.count.i119 is d
  %15 = add nsw i64 %wide.trip.count.i119, -1
  %16 = add nsw i64 %wide.trip.count.i119, -1
  %17 = add nsw i64 %wide.trip.count.i119, -1
  %18 = add nsw i64 %wide.trip.count.i119, -2
  %19 = add nsw i64 %wide.trip.count.i119, -2
  ;; k - 2
  %20 = add nsw i64 %wide.trip.count.i.i, -2
  %21 = add nsw i64 %wide.trip.count.i.i, -2
  ; n
  %22 = add nuw i64 %13, 1
  ; k
  %23 = add nuw i64 %14, 1
  ; d
  %_unwrap42 = add nuw nsw i64 %16, 1
  ; k
  %24 = mul nuw nsw i64 1, %23
  ; n * k
  %25 = mul nuw nsw i64 %24, %22
  %26 = mul nuw nsw i64 %25, %_unwrap42
  %mallocsize = mul nuw nsw i64 %26, 8
  %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
  %_malloccache = bitcast i8* %malloccall to double*
  store double* %_malloccache, double** %_cache, align 8, !invariant.group !43
  ; d
  ; %_unwrap67 = add nuw nsw i64 %17, 1
  %27 = mul nuw nsw i64 %23, %22
  %mallocsize83 = mul nuw nsw i64 %27, 8
  %malloccall84 = tail call noalias nonnull i8* @malloc(i64 %mallocsize83)
  %_malloccache85 = bitcast i8* %malloccall84 to double*
  store double* %_malloccache85, double** %_cache82, align 8, !invariant.group !44
  %scevgep = getelementptr i8, i8* %call12, i64 8
  %scevgep96 = bitcast i8* %scevgep to double*
  ; d - 1
  %_unwrap97 = add nuw nsw i64 %19, 1
  ; n * k * (d - 1)
  %28 = mul nuw nsw i64 %25, %_unwrap97
  %mallocsize99 = mul nuw nsw i64 %28, 8
  %malloccall100 = tail call noalias nonnull i8* @malloc(i64 %mallocsize99)
  %_malloccache101 = bitcast i8* %malloccall100 to double*
  store double* %_malloccache101, double** %_cache98, align 8, !invariant.group !45
  %malloccall120 = tail call noalias nonnull i8* @malloc(i64 %22)
  ;; I should check out what these caches do
  %"cmp2.i.i!manual_lcssa_malloccache" = bitcast i8* %malloccall120 to i1*
  store i1* %"cmp2.i.i!manual_lcssa_malloccache", i1** %"cmp2.i.i!manual_lcssa_cache", align 1, !invariant.group !46
  %mallocsize127 = mul nuw nsw i64 %22, 8
  %malloccall128 = tail call noalias nonnull i8* @malloc(i64 %mallocsize127)
  %"!manual_lcssa126_malloccache" = bitcast i8* %malloccall128 to i64*
  store i64* %"!manual_lcssa126_malloccache", i64** %"!manual_lcssa126_cache", align 8, !invariant.group !47
  %mallocsize134 = mul nuw nsw i64 %22, 8
  %malloccall135 = tail call noalias nonnull i8* @malloc(i64 %mallocsize134)
  %sub.i135_malloccache = bitcast i8* %malloccall135 to double*
  store double* %sub.i135_malloccache, double** %sub.i135_cache, align 8, !invariant.group !48
  ;; k - 1 + 2
  %29 = add nuw i64 %21, 1
  ;; (k - 1) * n
  %30 = mul nuw nsw i64 %29, %22
  %mallocsize140 = mul nuw nsw i64 %30, 8
  %malloccall141 = tail call noalias nonnull i8* @malloc(i64 %mallocsize140)
  %sub.i_malloccache = bitcast i8* %malloccall141 to double*
  store double* %sub.i_malloccache, double** %sub.i_cache, align 8, !invariant.group !49
  %mallocsize155 = mul nuw nsw i64 %22, 8
  %malloccall156 = tail call noalias nonnull i8* @malloc(i64 %mallocsize155)
  %"add.i!manual_lcssa154_malloccache" = bitcast i8* %malloccall156 to double*
  store double* %"add.i!manual_lcssa154_malloccache", double** %"add.i!manual_lcssa154_cache", align 8, !invariant.group !50
  br label %for.cond19.preheader

for.cond19.preheader:                             ; preds = %log_sum_exp.exit, %for.cond19.preheader.lr.ph
  ;; %iv3 is ix (to n);
  %iv3 = phi i64 [ %iv.next4, %log_sum_exp.exit ], [ 0, %for.cond19.preheader.lr.ph ]
  %31 = phi double [ undef, %for.cond19.preheader.lr.ph ], [ %86, %log_sum_exp.exit ]
  %32 = phi double [ undef, %for.cond19.preheader.lr.ph ], [ %87, %log_sum_exp.exit ]
  %slse.0143 = phi double [ 0.000000e+00, %for.cond19.preheader.lr.ph ], [ %add47, %log_sum_exp.exit ]
  %iv.next4 = add nuw nsw i64 %iv3, 1
  br i1 %cmp37.i, label %for.body23.lr.ph, label %for.end

for.body23.lr.ph:                                 ; preds = %for.cond19.preheader
  %mul25 = mul nsw i64 %iv3, %conv7
  %arrayidx26 = getelementptr inbounds double, double* %x, i64 %mul25
  br label %for.body23

for.body23:                                       ; preds = %sqnorm.exit, %for.body23.lr.ph
  ;; %iv5 is ik
  %iv5 = phi i64 [ %iv.next6, %sqnorm.exit ], [ 0, %for.body23.lr.ph ]
  %33 = phi double [ %32, %for.body23.lr.ph ], [ %64, %sqnorm.exit ]
  %iv.next6 = add nuw nsw i64 %iv5, 1
  %mul28 = mul nsw i64 %iv5, %conv7
  ; %"arrayidx29'ipg" = getelementptr inbounds double, double* %"means'", i64 %mul28
  %arrayidx29 = getelementptr inbounds double, double* %means, i64 %mul28
  br i1 %cmp10.i, label %for.body.i128.preheader, label %cQtimesx.exit

for.body.i128.preheader:                          ; preds = %for.body23
  br label %for.body.i128

for.body.i128:                                    ; preds = %for.body.i128, %for.body.i128.preheader
  %iv7 = phi i64 [ %iv.next8, %for.body.i128 ], [ 0, %for.body.i128.preheader ]
  %iv.next8 = add nuw nsw i64 %iv7, 1
  %arrayidx.i122 = getelementptr inbounds double, double* %arrayidx26, i64 %iv7
  %34 = load double, double* %arrayidx.i122, align 8, !tbaa !3
  ; %"arrayidx2.i123'ipg" = getelementptr inbounds double, double* %"arrayidx29'ipg", i64 %iv7
  %arrayidx2.i123 = getelementptr inbounds double, double* %arrayidx29, i64 %iv7
  %35 = load double, double* %arrayidx2.i123, align 8, !tbaa !3
  %sub.i124 = fsub double %34, %35
  ; %"arrayidx4.i125'ipg" = getelementptr inbounds double, double* %"'ipc37", i64 %iv7
  %arrayidx4.i125 = getelementptr inbounds double, double* %2, i64 %iv7
  store double %sub.i124, double* %arrayidx4.i125, align 8, !tbaa !3
  %exitcond.not.i127 = icmp eq i64 %iv.next8, %wide.trip.count.i119
  br i1 %exitcond.not.i127, label %for.body.i114.preheader, label %for.body.i128, !llvm.loop !11

for.body.i114.preheader:                          ; preds = %for.body.i128
  %"arrayidx33'ipg" = getelementptr inbounds double, double* %"'ipc21", i64 %mul28
  %arrayidx33 = getelementptr inbounds double, double* %0, i64 %mul28
  ; %conv is tri_size
  %mul34 = mul nsw i64 %iv5, %conv
  %"arrayidx35'ipg" = getelementptr inbounds double, double* %"Ls'", i64 %mul34
  %arrayidx35 = getelementptr inbounds double, double* %Ls, i64 %mul34
  ; %36 is d
  %36 = add nuw nsw i64 %16, 1
  %37 = load double*, double** %_cache, align 8, !dereferenceable !51, !invariant.group !43
  ;; k
  %38 = mul nuw nsw i64 1, %23
  ;; unused
  %39 = mul nuw nsw i64 %38, %22
  ;; ik
  %40 = mul nuw nsw i64 %iv5, 1
  %41 = add nuw nsw i64 0, %40
  ;; ix * k
  %42 = mul nuw nsw i64 %iv3, %38
  %43 = add nuw nsw i64 %41, %42
  %44 = mul nuw nsw i64 %43, %36
  %45 = getelementptr inbounds double, double* %37, i64 %44
  %46 = bitcast double* %45 to i8*
  %47 = bitcast double* %call941 to i8*
  ; %48 is d * sizeof(double)
  %48 = mul nuw nsw i64 8, %36
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %46, i8* nonnull align 8 %47, i64 %48, i1 false)
  br label %for.body.i114

for.body.i114:                                    ; preds = %for.body.i114, %for.body.i114.preheader
  %iv9 = phi i64 [ %iv.next10, %for.body.i114 ], [ 0, %for.body.i114.preheader ]
  %iv.next10 = add nuw nsw i64 %iv9, 1
  %"arrayidx.i111'ipg" = getelementptr inbounds double, double* %"arrayidx33'ipg", i64 %iv9
  %arrayidx.i111 = getelementptr inbounds double, double* %arrayidx33, i64 %iv9
  %49 = load double, double* %arrayidx.i111, align 8, !tbaa !3, !invariant.group !52
  %"arrayidx2.i112'ipg" = getelementptr inbounds double, double* %"'ipc37", i64 %iv9
  %arrayidx2.i112 = getelementptr inbounds double, double* %2, i64 %iv9
  %50 = load double, double* %arrayidx2.i112, align 8, !tbaa !3
  %mul.i113 = fmul double %49, %50
  %"arrayidx4.i'ipg" = getelementptr inbounds double, double* %"'ipc40", i64 %iv9
  %arrayidx4.i = getelementptr inbounds double, double* %3, i64 %iv9
  store double %mul.i113, double* %arrayidx4.i, align 8, !tbaa !3
  %exitcond72.not.i = icmp eq i64 %iv.next10, %wide.trip.count.i119
  br i1 %exitcond72.not.i, label %for.body7.i.preheader, label %for.body.i114, !llvm.loop !15

for.body7.i.preheader:                            ; preds = %for.body.i114
  %51 = add nuw nsw i64 %17, 1
  br label %for.body7.i

for.cond5.loopexit.i.loopexit:                    ; preds = %for.body13.i
  br label %for.cond5.loopexit.i

for.cond5.loopexit.i:                             ; preds = %for.body7.i, %for.cond5.loopexit.i.loopexit
  %indvars.iv.next62.i = add nuw nsw i64 %iv.next12, 1
  %exitcond68.not.i = icmp eq i64 %iv.next12, %wide.trip.count.i119
  br i1 %exitcond68.not.i, label %cQtimesx.exit.loopexit, label %for.body7.i, !llvm.loop !16

for.body7.i:                                      ; preds = %for.cond5.loopexit.i, %for.body7.i.preheader
  %iv11 = phi i64 [ %iv.next12, %for.cond5.loopexit.i ], [ 0, %for.body7.i.preheader ]
  %52 = mul nsw i64 %iv11, -1
  ; (d - 2) - i
  %53 = add i64 %18, %52
  ; i + 1
  %54 = add i64 %iv11, 1
  %55 = trunc i64 %iv11 to i32
  %iv.next12 = add nuw nsw i64 %iv11, 1
  ; i + 1
  ; %add.i115 = add nuw nsw i32 %55, 1
  %cmp1254.i = icmp ult i64 %iv.next12, %wide.trip.count.i119
  br i1 %cmp1254.i, label %for.body13.lr.ph.i, label %for.cond5.loopexit.i

for.body13.lr.ph.i:                               ; preds = %for.body7.i
  %56 = xor i32 %55, -1
  %sub9.i = add i32 %mul8.i, %56
  %57 = trunc i64 %iv11 to i32
  %mul10.i = mul nsw i32 %sub9.i, %57
  %div.i = sdiv i32 %mul10.i, 2
  %"arrayidx19.i'ipg" = getelementptr inbounds double, double* %"'ipc37", i64 %iv11
  %arrayidx19.i = getelementptr inbounds double, double* %2, i64 %iv11
  %58 = sext i32 %div.i to i64
  %59 = load double, double* %arrayidx19.i, align 8, !tbaa !3
  br label %for.body13.i

for.body13.i:                                     ; preds = %for.body13.i, %for.body13.lr.ph.i
  %iv13 = phi i64 [ %iv.next14, %for.body13.i ], [ 0, %for.body13.lr.ph.i ]
  %60 = add i64 %58, %iv13
  %iv.next14 = add nuw nsw i64 %iv13, 1
  %61 = add i64 %54, %iv13
  %"arrayidx15.i'ipg" = getelementptr inbounds double, double* %"'ipc40", i64 %61
  %arrayidx15.i = getelementptr inbounds double, double* %3, i64 %61
  %62 = load double, double* %arrayidx15.i, align 8, !tbaa !3
  %"arrayidx17.i'ipg" = getelementptr inbounds double, double* %"arrayidx35'ipg", i64 %60
  %arrayidx17.i = getelementptr inbounds double, double* %arrayidx35, i64 %60
  %63 = load double, double* %arrayidx17.i, align 8, !tbaa !3, !invariant.group !53
  %mul20.i = fmul double %63, %59
  %add21.i = fadd double %62, %mul20.i
  store double %add21.i, double* %arrayidx15.i, align 8, !tbaa !3
  %indvars.iv.next.i117 = add nsw i64 %60, 1
  %indvars.iv.next64.i = add nuw nsw i64 %61, 1
  %exitcond.not.i118 = icmp eq i64 %indvars.iv.next64.i, %wide.trip.count.i119
  br i1 %exitcond.not.i118, label %for.cond5.loopexit.i.loopexit, label %for.body13.i, !llvm.loop !17

cQtimesx.exit.loopexit:                           ; preds = %for.cond5.loopexit.i
  %.pre = load double, double* %3, align 8, !tbaa !3
  br label %cQtimesx.exit

cQtimesx.exit:                                    ; preds = %cQtimesx.exit.loopexit, %for.body23
  %64 = phi double [ %.pre, %cQtimesx.exit.loopexit ], [ %33, %for.body23 ]
  %65 = load double*, double** %_cache82, align 8, !dereferenceable !51, !invariant.group !44
  ; unused
  %66 = mul nuw nsw i64 %23, %22
  ; ix * k
  %67 = mul nuw nsw i64 %iv3, %23
  ; ix * k + ik
  %68 = add nuw nsw i64 %iv5, %67
  %69 = getelementptr inbounds double, double* %65, i64 %68

  store double %64, double* %69, align 8, !invariant.group !54
  %"arrayidx38'ipg" = getelementptr inbounds double, double* %"alphas'", i64 %iv5
  %arrayidx38 = getelementptr inbounds double, double* %alphas, i64 %iv5
  %70 = load double, double* %arrayidx38, align 8, !tbaa !3
  %"arrayidx39'ipg" = getelementptr inbounds double, double* %"'ipc", i64 %iv5
  %arrayidx39 = getelementptr inbounds double, double* %1, i64 %iv5
  %71 = load double, double* %arrayidx39, align 8, !tbaa !3
  %add = fadd double %70, %71
  %mul.i102 = fmul double %64, %64
  ;; always true, d > 1
  br i1 %cmp15.i, label %for.body.i109.preheader, label %sqnorm.exit

for.body.i109.preheader:                          ; preds = %cQtimesx.exit
  ; d - 1
  %72 = add nuw nsw i64 %19, 1
  %73 = load double*, double** %_cache98, align 8, !dereferenceable !51, !invariant.group !45
  ; k
  %74 = mul nuw nsw i64 1, %23
  ; ; k * n
  %75 = mul nuw nsw i64 %74, %22
  ; ik
  %76 = mul nuw nsw i64 %iv5, 1
  ; ik
  %77 = add nuw nsw i64 0, %76
  ; ix * k
  %78 = mul nuw nsw i64 %iv3, %74
  ; ik + ix * k
  %79 = add nuw nsw i64 %77, %78
  ; (ik + ix * k) * (d - 1)
  %80 = mul nuw nsw i64 %79, %72
  %81 = getelementptr inbounds double, double* %73, i64 %80
  %82 = bitcast double* %81 to i8*
  %83 = bitcast double* %scevgep96 to i8*
  ; (d - 1) * sizeof(double)
  %84 = mul nuw nsw i64 8, %72
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %82, i8* nonnull align 8 %83, i64 %84, i1 false)
  br label %for.body.i109

for.body.i109:                                    ; preds = %for.body.i109, %for.body.i109.preheader
  %iv15 = phi i64 [ %iv.next16, %for.body.i109 ], [ 0, %for.body.i109.preheader ]
  %res.017.i = phi double [ %add.i106, %for.body.i109 ], [ %mul.i102, %for.body.i109.preheader ]
  %iv.next16 = add nuw nsw i64 %iv15, 1
  ; %"arrayidx2.i'ipg" = getelementptr inbounds double, double* %"'ipc40", i64 %iv.next16
  %arrayidx2.i = getelementptr inbounds double, double* %3, i64 %iv.next16
  %85 = load double, double* %arrayidx2.i, align 8, !tbaa !3
  %mul5.i = fmul double %85, %85
  %add.i106 = fadd double %res.017.i, %mul5.i
  %indvars.iv.next.i107 = add nuw nsw i64 %iv.next16, 1
  %exitcond.not.i108 = icmp eq i64 %indvars.iv.next.i107, %wide.trip.count.i119
  br i1 %exitcond.not.i108, label %sqnorm.exit.loopexit, label %for.body.i109, !llvm.loop !10

sqnorm.exit.loopexit:                             ; preds = %for.body.i109
  br label %sqnorm.exit

sqnorm.exit:                                      ; preds = %sqnorm.exit.loopexit, %cQtimesx.exit
  %res.0.lcssa.i = phi double [ %mul.i102, %cQtimesx.exit ], [ %add.i106, %sqnorm.exit.loopexit ]
  %mul42 = fmul double %res.0.lcssa.i, 5.000000e-01
  %sub43 = fsub double %add, %mul42
  ; %"arrayidx44'ipg" = getelementptr inbounds double, double* %"'ipc116", i64 %iv5
  %arrayidx44 = getelementptr inbounds double, double* %4, i64 %iv5
  store double %sub43, double* %arrayidx44, align 8, !tbaa !3
  %exitcond.not = icmp eq i64 %iv.next6, %conv4
  br i1 %exitcond.not, label %for.end.loopexit, label %for.body23, !llvm.loop !18

for.end.loopexit:                                 ; preds = %sqnorm.exit
  %.pre146 = load double, double* %4, align 8, !tbaa !3
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %for.cond19.preheader
  %86 = phi double [ %.pre146, %for.end.loopexit ], [ %31, %for.cond19.preheader ]
  %87 = phi double [ %64, %for.end.loopexit ], [ %32, %for.cond19.preheader ]
  br i1 %cmp13.i.i, label %for.body.i.i.preheader, label %arr_max.exit.i

for.body.i.i.preheader:                           ; preds = %for.end
  br label %for.body.i.i

for.body.i.i:                                     ; preds = %for.body.i.i, %for.body.i.i.preheader
  %88 = phi i64 [ 0, %for.body.i.i.preheader ], [ %90, %for.body.i.i ]
  %iv17 = phi i64 [ %iv.next18, %for.body.i.i ], [ 0, %for.body.i.i.preheader ]
  %m.015.i.i = phi double [ %m.1.i.i, %for.body.i.i ], [ %86, %for.body.i.i.preheader ]
  %iv.next18 = add nuw nsw i64 %iv17, 1
  %"arrayidx1.i.i'ipg" = getelementptr inbounds double, double* %"'ipc116", i64 %iv.next18
  %arrayidx1.i.i = getelementptr inbounds double, double* %4, i64 %iv.next18
  %89 = load double, double* %arrayidx1.i.i, align 8, !tbaa !3
  %cmp2.i.i = fcmp olt double %m.015.i.i, %89
  %90 = select i1 %cmp2.i.i, i64 %iv.next18, i64 %88
  %m.1.i.i = select i1 %cmp2.i.i, double %89, double %m.015.i.i
  %indvars.iv.next.i.i = add nuw nsw i64 %iv.next18, 1
  %exitcond.not.i.i = icmp eq i64 %indvars.iv.next.i.i, %wide.trip.count.i.i
  br i1 %exitcond.not.i.i, label %arr_max.exit.i.loopexit, label %for.body.i.i, !llvm.loop !7

arr_max.exit.i.loopexit:                          ; preds = %for.body.i.i
  %"!manual_lcssa126" = phi i64 [ %88, %for.body.i.i ]
  %"iv.next18!manual_lcssa" = phi i64 [ %iv.next18, %for.body.i.i ]
  %"cmp2.i.i!manual_lcssa" = phi i1 [ %cmp2.i.i, %for.body.i.i ]
  %"!manual_lcssa" = phi i64 [ %90, %for.body.i.i ]
  %91 = load i1*, i1** %"cmp2.i.i!manual_lcssa_cache", align 8, !dereferenceable !51, !invariant.group !46
  %92 = getelementptr inbounds i1, i1* %91, i64 %iv3
  store i1 %"cmp2.i.i!manual_lcssa", i1* %92, align 1, !invariant.group !55
  %93 = load i64*, i64** %"!manual_lcssa126_cache", align 8, !dereferenceable !51, !invariant.group !47
  %94 = getelementptr inbounds i64, i64* %93, i64 %iv3
  store i64 %"!manual_lcssa126", i64* %94, align 8, !invariant.group !56

  br label %arr_max.exit.i

arr_max.exit.i:                                   ; preds = %arr_max.exit.i.loopexit, %for.end
  %m.0.lcssa.i.i = phi double [ %86, %for.end ], [ %m.1.i.i, %arr_max.exit.i.loopexit ]
  br i1 %cmp37.i, label %for.body.preheader.i, label %log_sum_exp.exit

for.body.preheader.i:                             ; preds = %arr_max.exit.i
  ; main_term[0] - max
  %sub.i135 = fsub double %86, %m.0.lcssa.i.i
  %95 = load double*, double** %sub.i135_cache, align 8, !dereferenceable !51, !invariant.group !48
  %96 = getelementptr inbounds double, double* %95, i64 %iv3
  store double %sub.i135, double* %96, align 8, !invariant.group !57
  %97 = tail call double @llvm.exp.f64(double %sub.i135) #14
  %add.i136 = fadd double %97, 0.000000e+00

  br i1 %exitcond.not.i99137, label %log_sum_exp.exit, label %for.body.for.body_crit_edge.i.preheader, !llvm.loop !12

for.body.for.body_crit_edge.i.preheader:          ; preds = %for.body.preheader.i
  br label %for.body.for.body_crit_edge.i

for.body.for.body_crit_edge.i:                    ; preds = %for.body.for.body_crit_edge.i, %for.body.for.body_crit_edge.i.preheader
  %iv19 = phi i64 [ %iv.next20, %for.body.for.body_crit_edge.i ], [ 0, %for.body.for.body_crit_edge.i.preheader ]
  %add.i138 = phi double [ %add.i, %for.body.for.body_crit_edge.i ], [ %add.i136, %for.body.for.body_crit_edge.i.preheader ]
  %iv.next20 = add nuw nsw i64 %iv19, 1
  ; %"arrayidx.phi.trans.insert.i'ipg" = getelementptr inbounds double, double* %"'ipc116", i64 %iv.next20
  %arrayidx.phi.trans.insert.i = getelementptr inbounds double, double* %4, i64 %iv.next20
  %.pre.i101 = load double, double* %arrayidx.phi.trans.insert.i, align 8, !tbaa !3
  %sub.i = fsub double %.pre.i101, %m.0.lcssa.i.i
  %98 = load double*, double** %sub.i_cache, align 8, !dereferenceable !51, !invariant.group !49
  ;; unused, (k - 1) * n
  %99 = mul nuw nsw i64 %29, %22
  ;; ix * (k - 1)
  %100 = mul nuw nsw i64 %iv3, %29
  %101 = add nuw nsw i64 %iv19, %100
  ; %printfcall = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([13 x i8], [13 x i8]* @.flstr, i64 0, i64 0), double %sub.i)
  ; %printfc4ll = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([13 x i8], [13 x i8]* @.mystr, i64 0, i64 0), i64 %iv19)
  %102 = getelementptr inbounds double, double* %98, i64 %101
  store double %sub.i, double* %102, align 8, !invariant.group !58
  %103 = tail call double @llvm.exp.f64(double %sub.i) #14
  %add.i = fadd double %add.i138, %103
  %indvars.iv.next.i98 = add nuw nsw i64 %iv.next20, 1
  %exitcond.not.i99 = icmp eq i64 %indvars.iv.next.i98, %wide.trip.count.i.i
  br i1 %exitcond.not.i99, label %log_sum_exp.exit.loopexit, label %for.body.for.body_crit_edge.i, !llvm.loop !12

log_sum_exp.exit.loopexit:                        ; preds = %for.body.for.body_crit_edge.i
  %"add.i!manual_lcssa" = phi double [ %add.i, %for.body.for.body_crit_edge.i ]
  br label %log_sum_exp.exit

log_sum_exp.exit:                                 ; preds = %log_sum_exp.exit.loopexit, %for.body.preheader.i, %arr_max.exit.i
  %"add.i!manual_lcssa154" = phi double [ %add.i, %log_sum_exp.exit.loopexit ], [ undef, %for.body.preheader.i ], [ undef, %arr_max.exit.i ]
  %semx.0.lcssa.i = phi double [ 0.000000e+00, %arr_max.exit.i ], [ %add.i136, %for.body.preheader.i ], [ %add.i, %log_sum_exp.exit.loopexit ]
  %104 = load double*, double** %"add.i!manual_lcssa154_cache", align 8, !dereferenceable !51, !invariant.group !50
  %105 = getelementptr inbounds double, double* %104, i64 %iv3
  store double %"add.i!manual_lcssa154", double* %105, align 8, !invariant.group !59
  %106 = tail call double @llvm.log.f64(double %semx.0.lcssa.i) #14
  %add1.i = fadd double %m.0.lcssa.i.i, %106
  %add47 = fadd double %slse.0143, %add1.i
  %exitcond145.not = icmp eq i64 %iv.next4, %conv17
  br i1 %exitcond145.not, label %for.end50.loopexit, label %for.cond19.preheader, !llvm.loop !19

for.end50.loopexit:                               ; preds = %log_sum_exp.exit
  br label %for.end50

for.end50:                                        ; preds = %for.end50.loopexit, %preprocess_qs.exit
  %slse.0.lcssa = phi double [ 0.000000e+00, %preprocess_qs.exit ], [ %add47, %for.end50.loopexit ]
  store double %slse.0.lcssa, double* %err, align 8, !tbaa !3
  tail call void @free(i8* %call6)
  br label %invertfor.end50

invertentry:                                      ; preds = %invertpreprocess_qs.exit, %invertfor.body.lr.ph.i
  tail call void @free(i8* nonnull %"call15'mi")
  tail call void @free(i8* nonnull %"call12'mi")
  tail call void @free(i8* nonnull %"call9'mi")
  tail call void @free(i8* nonnull %"call6'mi")
  tail call void @free(i8* nonnull %"call'mi")
  tail call void @free(i8* %call)
  ret void

invertfor.body.lr.ph.i:                           ; preds = %invertfor.body.i
  br label %invertentry

invertfor.body.i:                                 ; preds = %invertfor.inc15.i, %invertfor.body3.lr.ph.i
  %107 = load i64, i64* %"iv'ac", align 8
  %"arrayidx.i'ipg_unwrap" = getelementptr inbounds double, double* %"'ipc", i64 %107
  store double 0.000000e+00, double* %"arrayidx.i'ipg_unwrap", align 8
  %108 = load i64, i64* %"iv'ac", align 8
  %109 = icmp eq i64 %108, 0
  %110 = xor i1 %109, true
  br i1 %109, label %invertfor.body.lr.ph.i, label %incinvertfor.body.i

incinvertfor.body.i:                              ; preds = %invertfor.body.i
  %111 = load i64, i64* %"iv'ac", align 8
  %112 = add nsw i64 %111, -1
  store i64 %112, i64* %"iv'ac", align 8
  br label %invertfor.inc15.i

invertfor.body3.lr.ph.i:                          ; preds = %invertfor.body3.i
  br label %invertfor.body.i

invertfor.body3.i:                                ; preds = %mergeinvertfor.body3.i_for.inc15.i.loopexit, %incinvertfor.body3.i
  ;; id
  %113 = load i64, i64* %"iv1'ac", align 8
  ;; ik
  %114 = load i64, i64* %"iv'ac", align 8
  %_unwrap = trunc i64 %114 to i32
  ;; ik * d
  %mul.i_unwrap = mul nsw i32 %_unwrap, %d
  %_unwrap22 = sext i32 %mul.i_unwrap to i64
  ;; ik * d + id
  %_unwrap23 = add nsw i64 %113, %_unwrap22
  %"arrayidx14.i'ipg_unwrap" = getelementptr inbounds double, double* %"'ipc21", i64 %_unwrap23
  %115 = load double, double* %"arrayidx14.i'ipg_unwrap", align 8
  store double 0.000000e+00, double* %"arrayidx14.i'ipg_unwrap", align 8

  %116 = load double, double* %"'de", align 8
  %117 = fadd fast double %116, %115
  store double %117, double* %"'de", align 8
  %118 = load double, double* %"'de", align 8
  store double 0.000000e+00, double* %"'de", align 8
  ;; id
  %119 = load i64, i64* %"iv1'ac", align 8
  ;; ik
  %120 = load i64, i64* %"iv'ac", align 8
  %arrayidx5.i_unwrap = getelementptr inbounds double, double* %Qs, i64 %_unwrap23
  %_unwrap24 = load double, double* %arrayidx5.i_unwrap, align 8, !tbaa !3, !invariant.group !42
  %121 = call fast double @llvm.exp.f64(double %_unwrap24)
  %122 = fmul fast double %118, %121
  %123 = load double, double* %"'de25", align 8
  %124 = fadd fast double %123, %122
  store double %124, double* %"'de25", align 8
  %125 = load double, double* %"add8.i'de", align 8
  store double 0.000000e+00, double* %"add8.i'de", align 8

  %126 = load double, double* %"'de26", align 8
  %127 = fadd fast double %126, %125
  store double %127, double* %"'de26", align 8

  %128 = load double, double* %"'de25", align 8
  %129 = fadd fast double %128, %125
  store double %129, double* %"'de25", align 8
  %130 = load double, double* %"'de25", align 8
  store double 0.000000e+00, double* %"'de25", align 8
  %131 = load i64, i64* %"iv1'ac", align 8
  %132 = load i64, i64* %"iv'ac", align 8

  %"arrayidx5.i'ipg_unwrap" = getelementptr inbounds double, double* %"Qs'", i64 %_unwrap23
  %133 = load double, double* %"arrayidx5.i'ipg_unwrap", align 8
  %134 = fadd fast double %133, %130
  store double %134, double* %"arrayidx5.i'ipg_unwrap", align 8

  %135 = load double, double* %"'de26", align 8
  store double 0.000000e+00, double* %"'de26", align 8
  %136 = load i64, i64* %"iv1'ac", align 8
  %137 = icmp eq i64 %136, 0
  %138 = xor i1 %137, true
  ; unused
  %139 = select fast i1 %138, double %135, double 0.000000e+00
  %140 = load double, double* %"add8.i'de", align 8
  %141 = fadd fast double %140, %135
  %142 = select fast i1 %137, double %140, double %141
  store double %142, double* %"add8.i'de", align 8
  br i1 %137, label %invertfor.body3.lr.ph.i, label %incinvertfor.body3.i

incinvertfor.body3.i:                             ; preds = %invertfor.body3.i
  %143 = load i64, i64* %"iv1'ac", align 8
  %144 = add nsw i64 %143, -1
  store i64 %144, i64* %"iv1'ac", align 8
  br label %invertfor.body3.i

invertfor.inc15.i.loopexit:                       ; preds = %invertfor.inc15.i
  ;; DEBUG_POINT_12
  ;; ik
  %145 = load i64, i64* %"iv'ac", align 8
  %"arrayidx.i'ipg_unwrap27" = getelementptr inbounds double, double* %"'ipc", i64 %145
  %146 = load double, double* %"arrayidx.i'ipg_unwrap27", align 8
  store double 0.000000e+00, double* %"arrayidx.i'ipg_unwrap27", align 8

  %147 = load double, double* %"add8.i'de", align 8
  %148 = fadd fast double %147, %146
  store double %148, double* %"add8.i'de", align 8
  %149 = load i64, i64* %"iv'ac", align 8
  %wide.trip.count.i_unwrap = zext i32 %d to i64
  %_unwrap28 = add nsw i64 %wide.trip.count.i_unwrap, -1
  br label %mergeinvertfor.body3.i_for.inc15.i.loopexit

mergeinvertfor.body3.i_for.inc15.i.loopexit:      ; preds = %invertfor.inc15.i.loopexit
  ;; id
  store i64 %_unwrap28, i64* %"iv1'ac", align 8
  br label %invertfor.body3.i

invertfor.inc15.i:                                ; preds = %mergeinvertfor.body.i_preprocess_qs.exit.loopexit, %incinvertfor.body.i
  %150 = load i64, i64* %"iv'ac", align 8
  ;; always true, d > 0
  %cmp235.i_unwrap = icmp sgt i32 %d, 0
  br i1 %cmp235.i_unwrap, label %invertfor.inc15.i.loopexit, label %invertfor.body.i

invertpreprocess_qs.exit.loopexit:                ; preds = %invertpreprocess_qs.exit
  %wide.trip.count44.i_unwrap = zext i32 %k to i64
  ;; k - 1
  %_unwrap29 = add nsw i64 %wide.trip.count44.i_unwrap, -1
  br label %mergeinvertfor.body.i_preprocess_qs.exit.loopexit

mergeinvertfor.body.i_preprocess_qs.exit.loopexit: ; preds = %invertpreprocess_qs.exit.loopexit
  ;; iv'ac is ik
  store i64 %_unwrap29, i64* %"iv'ac", align 8
  br label %invertfor.inc15.i

invertpreprocess_qs.exit:                         ; preds = %invertfor.end50, %invertfor.cond19.preheader.lr.ph
  ;; always true
  br i1 %cmp37.i, label %invertpreprocess_qs.exit.loopexit, label %invertentry

invertfor.cond19.preheader.lr.ph:                 ; preds = %invertfor.cond19.preheader
  %151 = load i64, i64* %"iv3'ac", align 8
  %152 = load i64, i64* %"iv5'ac", align 8
  %forfree = load double*, double** %_cache, align 8, !dereferenceable !51, !invariant.group !43
  %153 = bitcast double* %forfree to i8*
  tail call void @free(i8* nonnull %153)
  %154 = load i64, i64* %"iv3'ac", align 8
  %155 = load i64, i64* %"iv5'ac", align 8
  %forfree86 = load double*, double** %_cache82, align 8, !dereferenceable !51, !invariant.group !44
  %156 = bitcast double* %forfree86 to i8*
  tail call void @free(i8* nonnull %156)
  %157 = load i64, i64* %"iv3'ac", align 8
  %158 = load i64, i64* %"iv5'ac", align 8
  %forfree102 = load double*, double** %_cache98, align 8, !dereferenceable !51, !invariant.group !45
  %159 = bitcast double* %forfree102 to i8*
  tail call void @free(i8* nonnull %159)
  %160 = load i64, i64* %"iv3'ac", align 8
  %forfree121 = load i1*, i1** %"cmp2.i.i!manual_lcssa_cache", align 1, !dereferenceable !60, !invariant.group !46
  %161 = bitcast i1* %forfree121 to i8*
  tail call void @free(i8* nonnull %161)
  %162 = load i64, i64* %"iv3'ac", align 8
  %forfree129 = load i64*, i64** %"!manual_lcssa126_cache", align 8, !dereferenceable !51, !invariant.group !47
  %163 = bitcast i64* %forfree129 to i8*
  tail call void @free(i8* nonnull %163)
  %164 = load i64, i64* %"iv3'ac", align 8
  %forfree136 = load double*, double** %sub.i135_cache, align 8, !dereferenceable !51, !invariant.group !48
  %165 = bitcast double* %forfree136 to i8*
  tail call void @free(i8* nonnull %165)
  %166 = load i64, i64* %"iv3'ac", align 8
  %167 = load i64, i64* %"iv19'ac", align 8
  %forfree142 = load double*, double** %sub.i_cache, align 8, !dereferenceable !51, !invariant.group !49
  %168 = bitcast double* %forfree142 to i8*
  tail call void @free(i8* nonnull %168)
  %169 = load i64, i64* %"iv3'ac", align 8
  %forfree157 = load double*, double** %"add.i!manual_lcssa154_cache", align 8, !dereferenceable !51, !invariant.group !50
  %170 = bitcast double* %forfree157 to i8*
  tail call void @free(i8* nonnull %170)
  br label %invertpreprocess_qs.exit

invertfor.cond19.preheader:                       ; preds = %invertfor.end, %invertfor.body23.lr.ph
  ;; DEBUG_POINT_11
  %171 = load double, double* %"'de30", align 8
  store double 0.000000e+00, double* %"'de30", align 8
  %172 = load i64, i64* %"iv3'ac", align 8
  ;; ix == 0
  %173 = icmp eq i64 %172, 0
  %174 = xor i1 %173, true
  ; unused
  %175 = select fast i1 %174, double %171, double 0.000000e+00
  %176 = load double, double* %"'de31", align 8
  %177 = fadd fast double %176, %171
  %178 = select fast i1 %173, double %176, double %177
  store double %178, double* %"'de31", align 8

  %179 = load double, double* %"'de32", align 8
  store double 0.000000e+00, double* %"'de32", align 8
  ; unused
  %180 = select fast i1 %174, double %179, double 0.000000e+00
  %181 = load double, double* %"'de33", align 8
  %182 = fadd fast double %181, %179
  %183 = select fast i1 %173, double %181, double %182
  store double %183, double* %"'de33", align 8
  %184 = load double, double* %"slse.0143'de", align 8
  store double 0.000000e+00, double* %"slse.0143'de", align 8
  %185 = select fast i1 %174, double %184, double 0.000000e+00
  %186 = load double, double* %"add47'de", align 8
  %187 = fadd fast double %186, %184
  %188 = select fast i1 %173, double %186, double %187
  store double %188, double* %"add47'de", align 8
  ;;bookmark3
  br i1 %173, label %invertfor.cond19.preheader.lr.ph, label %incinvertfor.cond19.preheader

incinvertfor.cond19.preheader:                    ; preds = %invertfor.cond19.preheader
  %189 = load i64, i64* %"iv3'ac", align 8
  %190 = add nsw i64 %189, -1
  store i64 %190, i64* %"iv3'ac", align 8
  br label %invertlog_sum_exp.exit

invertfor.body23.lr.ph:                           ; preds = %invertfor.body23
  br label %invertfor.cond19.preheader

invertfor.body23:                                 ; preds = %invertcQtimesx.exit, %invertfor.body.i128.preheader
  %191 = load double, double* %"'de34", align 8
  store double 0.000000e+00, double* %"'de34", align 8
  ; ik
  %192 = load i64, i64* %"iv5'ac", align 8
  ; ik == 0
  %193 = icmp eq i64 %192, 0
  %194 = xor i1 %193, true
  ; unused
  %195 = select fast i1 %194, double %191, double 0.000000e+00
  %196 = load double, double* %"'de35", align 8
  %197 = fadd fast double %196, %191
  %198 = select fast i1 %193, double %196, double %197
  store double %198, double* %"'de35", align 8

  %199 = select fast i1 %193, double %191, double 0.000000e+00
  %200 = load double, double* %"'de32", align 8
  %201 = fadd fast double %200, %191
  %202 = select fast i1 %193, double %201, double %200
  store double %202, double* %"'de32", align 8
  br i1 %193, label %invertfor.body23.lr.ph, label %incinvertfor.body23

incinvertfor.body23:                              ; preds = %invertfor.body23
  %203 = load i64, i64* %"iv5'ac", align 8
  %204 = add nsw i64 %203, -1
  store i64 %204, i64* %"iv5'ac", align 8
  br label %invertsqnorm.exit

invertfor.body.i128.preheader:                    ; preds = %invertfor.body.i128
  br label %invertfor.body23

invertfor.body.i128:                              ; preds = %mergeinvertfor.body.i128_for.body.i114.preheader, %incinvertfor.body.i128
  %205 = load i64, i64* %"iv7'ac", align 8
  %206 = load i64, i64* %"iv5'ac", align 8
  %207 = load i64, i64* %"iv3'ac", align 8
  %"arrayidx4.i125'ipg_unwrap" = getelementptr inbounds double, double* %"'ipc37", i64 %205
  %208 = load double, double* %"arrayidx4.i125'ipg_unwrap", align 8
  store double 0.000000e+00, double* %"arrayidx4.i125'ipg_unwrap", align 8
  %209 = load double, double* %"sub.i124'de", align 8
  %210 = fadd fast double %209, %208
  store double %210, double* %"sub.i124'de", align 8
  %211 = load double, double* %"sub.i124'de", align 8
  %212 = fneg fast double %211
  store double 0.000000e+00, double* %"sub.i124'de", align 8

  %213 = load double, double* %"'de38", align 8
  %214 = fadd fast double %213, %212
  store double %214, double* %"'de38", align 8
  %215 = load double, double* %"'de38", align 8
  store double 0.000000e+00, double* %"'de38", align 8

  %216 = load i64, i64* %"iv7'ac", align 8
  %217 = load i64, i64* %"iv5'ac", align 8
  %218 = load i64, i64* %"iv3'ac", align 8
  ; ik * d
  %mul28_unwrap = mul nsw i64 %217, %conv7
  %"arrayidx29'ipg_unwrap" = getelementptr inbounds double, double* %"means'", i64 %mul28_unwrap
  %"arrayidx2.i123'ipg_unwrap" = getelementptr inbounds double, double* %"arrayidx29'ipg_unwrap", i64 %216
  %219 = load double, double* %"arrayidx2.i123'ipg_unwrap", align 8
  %220 = fadd fast double %219, %215
  store double %220, double* %"arrayidx2.i123'ipg_unwrap", align 8
  %221 = load i64, i64* %"iv7'ac", align 8
  %222 = icmp eq i64 %221, 0
  %223 = xor i1 %222, true
  br i1 %222, label %invertfor.body.i128.preheader, label %incinvertfor.body.i128

incinvertfor.body.i128:                           ; preds = %invertfor.body.i128
  %224 = load i64, i64* %"iv7'ac", align 8
  %225 = add nsw i64 %224, -1
  store i64 %225, i64* %"iv7'ac", align 8
  br label %invertfor.body.i128

invertfor.body.i114.preheader:                    ; preds = %invertfor.body.i114
  ;; DEBUG_POINT_10
  %226 = load i64, i64* %"iv5'ac", align 8
  %227 = load i64, i64* %"iv3'ac", align 8
  %wide.trip.count.i119_unwrap = zext i32 %d to i64
  %_unwrap39 = add nsw i64 %wide.trip.count.i119_unwrap, -1
  br label %mergeinvertfor.body.i128_for.body.i114.preheader

mergeinvertfor.body.i128_for.body.i114.preheader: ; preds = %invertfor.body.i114.preheader
  store i64 %_unwrap39, i64* %"iv7'ac", align 8
  br label %invertfor.body.i128

invertfor.body.i114:                              ; preds = %mergeinvertfor.body.i114_for.body7.i.preheader, %incinvertfor.body.i114
  ;; DEBUG_POINT_9
  %228 = load i64, i64* %"iv9'ac", align 8
  %229 = load i64, i64* %"iv5'ac", align 8
  %230 = load i64, i64* %"iv3'ac", align 8
  %"arrayidx4.i'ipg_unwrap" = getelementptr inbounds double, double* %"'ipc40", i64 %228
  %231 = load double, double* %"arrayidx4.i'ipg_unwrap", align 8
  store double 0.000000e+00, double* %"arrayidx4.i'ipg_unwrap", align 8
  %232 = load double, double* %"mul.i113'de", align 8
  %233 = fadd fast double %232, %231
  store double %233, double* %"mul.i113'de", align 8
  %234 = load double, double* %"mul.i113'de", align 8
  %235 = load i64, i64* %"iv9'ac", align 8
  %236 = load i64, i64* %"iv5'ac", align 8
  %237 = load i64, i64* %"iv3'ac", align 8
  %conv17_unwrap = sext i32 %n to i64
  %_unwrap43 = add nsw i64 %conv17_unwrap, -1
  ;; n
  %238 = add nuw i64 %_unwrap43, 1
  %conv4_unwrap = sext i32 %k to i64
  %_unwrap44 = add nsw i64 %conv4_unwrap, -1
  ;; k
  %239 = add nuw i64 %_unwrap44, 1
  ;; k
  %240 = mul nuw nsw i64 1, %239
  ;; unused; k * n
  %241 = mul nuw nsw i64 %240, %238
  %242 = load double*, double** %_cache, align 8, !dereferenceable !51, !invariant.group !43
  ; ik
  %243 = load i64, i64* %"iv5'ac", align 8
  ; k
  %244 = mul nuw nsw i64 1, %239
  %245 = load i64, i64* %"iv3'ac", align 8
  ; unused
  %246 = mul nuw nsw i64 %244, %238
  ; ik
  %247 = mul nuw nsw i64 %243, 1
  %248 = add nuw nsw i64 0, %247
  ; ix * k
  %249 = mul nuw nsw i64 %245, %244
  ; ik + ix * k
  %250 = add nuw nsw i64 %248, %249
  %251 = load i64, i64* %"iv9'ac", align 8
  %252 = load i64, i64* %"iv5'ac", align 8
  %253 = load i64, i64* %"iv3'ac", align 8
  %wide.trip.count.i119_unwrap45 = zext i32 %d to i64
  %_unwrap46 = add nsw i64 %wide.trip.count.i119_unwrap45, -1
  ; d
  %_unwrap47 = add nuw nsw i64 %_unwrap46, 1
  ; (ik + ix * k) * d
  %254 = mul nuw nsw i64 %250, %_unwrap47
  %255 = getelementptr inbounds double, double* %242, i64 %254
  %256 = getelementptr inbounds double, double* %255, i64 %235
  %257 = load double, double* %256, align 8, !invariant.group !61
  %m0diffe = fmul fast double %234, %257

  %258 = load i64, i64* %"iv9'ac", align 8
  %259 = load i64, i64* %"iv5'ac", align 8
  %260 = load i64, i64* %"iv3'ac", align 8
  ; ik * d
  %mul28_unwrap48 = mul nsw i64 %259, %conv7
  %arrayidx33_unwrap = getelementptr inbounds double, double* %0, i64 %mul28_unwrap48
  %arrayidx.i111_unwrap = getelementptr inbounds double, double* %arrayidx33_unwrap, i64 %258
  %_unwrap49 = load double, double* %arrayidx.i111_unwrap, align 8, !tbaa !3, !invariant.group !52
  %m1diffe = fmul fast double %234, %_unwrap49
  store double 0.000000e+00, double* %"mul.i113'de", align 8
  %261 = load double, double* %"'de50", align 8
  %262 = fadd fast double %261, %m0diffe
  store double %262, double* %"'de50", align 8
  %263 = load double, double* %"'de51", align 8
  %264 = fadd fast double %263, %m1diffe
  store double %264, double* %"'de51", align 8
  %265 = load double, double* %"'de51", align 8
  store double 0.000000e+00, double* %"'de51", align 8

  %266 = load i64, i64* %"iv9'ac", align 8
  %267 = load i64, i64* %"iv5'ac", align 8
  %268 = load i64, i64* %"iv3'ac", align 8
  %"arrayidx2.i112'ipg_unwrap" = getelementptr inbounds double, double* %"'ipc37", i64 %266
  %269 = load double, double* %"arrayidx2.i112'ipg_unwrap", align 8
  %270 = fadd fast double %269, %265
  store double %270, double* %"arrayidx2.i112'ipg_unwrap", align 8

  %271 = load double, double* %"'de50", align 8
  store double 0.000000e+00, double* %"'de50", align 8
  %272 = load i64, i64* %"iv9'ac", align 8
  %273 = load i64, i64* %"iv5'ac", align 8
  %274 = load i64, i64* %"iv3'ac", align 8
  %"arrayidx33'ipg_unwrap" = getelementptr inbounds double, double* %"'ipc21", i64 %mul28_unwrap48
  %"arrayidx.i111'ipg_unwrap" = getelementptr inbounds double, double* %"arrayidx33'ipg_unwrap", i64 %272
  %275 = load double, double* %"arrayidx.i111'ipg_unwrap", align 8
  %276 = fadd fast double %275, %271
  store double %276, double* %"arrayidx.i111'ipg_unwrap", align 8

  %277 = load i64, i64* %"iv9'ac", align 8
  %278 = icmp eq i64 %277, 0
  %279 = xor i1 %278, true
  br i1 %278, label %invertfor.body.i114.preheader, label %incinvertfor.body.i114

incinvertfor.body.i114:                           ; preds = %invertfor.body.i114
  %280 = load i64, i64* %"iv9'ac", align 8
  %281 = add nsw i64 %280, -1
  store i64 %281, i64* %"iv9'ac", align 8
  br label %invertfor.body.i114

invertfor.body7.i.preheader:                      ; preds = %invertfor.body7.i
  %282 = load i64, i64* %"iv5'ac", align 8
  %283 = load i64, i64* %"iv3'ac", align 8
  %wide.trip.count.i119_unwrap52 = zext i32 %d to i64
  ;; d - 1
  %_unwrap53 = add nsw i64 %wide.trip.count.i119_unwrap52, -1
  br label %mergeinvertfor.body.i114_for.body7.i.preheader

mergeinvertfor.body.i114_for.body7.i.preheader:   ; preds = %invertfor.body7.i.preheader
  ;; iv9'ac is id
  store i64 %_unwrap53, i64* %"iv9'ac", align 8
  br label %invertfor.body.i114

invertfor.cond5.loopexit.i.loopexit:              ; preds = %invertfor.cond5.loopexit.i
  %284 = load i64, i64* %"iv11'ac", align 8
  %285 = load i64, i64* %"iv5'ac", align 8
  %286 = load i64, i64* %"iv3'ac", align 8
  %wide.trip.count.i119_unwrap54 = zext i32 %d to i64
  ; d - 2
  %_unwrap55 = add nsw i64 %wide.trip.count.i119_unwrap54, -2
  ; -i
  %_unwrap56 = mul nsw i64 %284, -1
  ; d - 2 - i
  %_unwrap57 = add i64 %_unwrap55, %_unwrap56
  br label %mergeinvertfor.body13.i_for.cond5.loopexit.i.loopexit

mergeinvertfor.body13.i_for.cond5.loopexit.i.loopexit: ; preds = %invertfor.cond5.loopexit.i.loopexit
  store i64 %_unwrap57, i64* %"iv13'ac", align 8
  br label %invertfor.body13.i

invertfor.cond5.loopexit.i:                       ; preds = %mergeinvertfor.body7.i_cQtimesx.exit.loopexit, %incinvertfor.body7.i
  ; id
  %287 = load i64, i64* %"iv11'ac", align 8
  ; ik
  %288 = load i64, i64* %"iv5'ac", align 8
  ; ix
  %289 = load i64, i64* %"iv3'ac", align 8
  ; id + 1
  %iv.next12_unwrap = add nuw nsw i64 %287, 1
  %wide.trip.count.i119_unwrap58 = zext i32 %d to i64
  ; id + 1 < d, it does something different the first iteration
  %cmp1254.i_unwrap = icmp ult i64 %iv.next12_unwrap, %wide.trip.count.i119_unwrap58
  ; %printfc4ll = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([13 x i8], [13 x i8]* @.mystr, i64 0, i64 0), i64 %287)
  br i1 %cmp1254.i_unwrap, label %invertfor.cond5.loopexit.i.loopexit, label %invertfor.body7.i

invertfor.body7.i:                                ; preds = %invertfor.body13.lr.ph.i, %invertfor.cond5.loopexit.i
  ;; It does this the first iteration
  %290 = load i64, i64* %"iv11'ac", align 8
  ;; id == 0
  %291 = icmp eq i64 %290, 0
  %292 = xor i1 %291, true
  br i1 %291, label %invertfor.body7.i.preheader, label %incinvertfor.body7.i

incinvertfor.body7.i:                             ; preds = %invertfor.body7.i
  %293 = load i64, i64* %"iv11'ac", align 8
  %294 = add nsw i64 %293, -1
  store i64 %294, i64* %"iv11'ac", align 8
  br label %invertfor.cond5.loopexit.i

invertfor.body13.lr.ph.i:                         ; preds = %invertfor.body13.i
  %295 = load double, double* %"'de59", align 8
  ; bookmark1
  ; %printfc4ll = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([13 x i8], [13 x i8]* @.flstr, i64 0, i64 0), double %295)
  store double 0.000000e+00, double* %"'de59", align 8
  %296 = load i64, i64* %"iv11'ac", align 8
  %297 = load i64, i64* %"iv5'ac", align 8
  %298 = load i64, i64* %"iv3'ac", align 8
  %"arrayidx19.i'ipg_unwrap" = getelementptr inbounds double, double* %"'ipc37", i64 %296
  %299 = load double, double* %"arrayidx19.i'ipg_unwrap", align 8
  %300 = fadd fast double %299, %295
  store double %300, double* %"arrayidx19.i'ipg_unwrap", align 8
  br label %invertfor.body7.i

invertfor.body13.i:                               ; preds = %incinvertfor.body13.i, %mergeinvertfor.body13.i_for.cond5.loopexit.i.loopexit
  ;; DEBUG_POINT_8
  %301 = load i64, i64* %"iv13'ac", align 8
  %302 = load i64, i64* %"iv11'ac", align 8
  %303 = load i64, i64* %"iv5'ac", align 8
  %304 = load i64, i64* %"iv3'ac", align 8
  ; i + 1
  %_unwrap60 = add i64 %302, 1
  ; i + 1 + j
  %_unwrap61 = add i64 %_unwrap60, %301
  %"arrayidx15.i'ipg_unwrap" = getelementptr inbounds double, double* %"'ipc40", i64 %_unwrap61
  %305 = load double, double* %"arrayidx15.i'ipg_unwrap", align 8
  store double 0.000000e+00, double* %"arrayidx15.i'ipg_unwrap", align 8

  %306 = load double, double* %"add21.i'de", align 8
  %307 = fadd fast double %306, %305
  store double %307, double* %"add21.i'de", align 8

  %308 = load double, double* %"add21.i'de", align 8
  ; %printfc4ll = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([13 x i8], [13 x i8]* @.flstr, i64 0, i64 0), double %308)
  store double 0.000000e+00, double* %"add21.i'de", align 8

  %309 = load double, double* %"'de62", align 8
  %310 = fadd fast double %309, %308
  store double %310, double* %"'de62", align 8

  %311 = load double, double* %"mul20.i'de", align 8
  %312 = fadd fast double %311, %308
  store double %312, double* %"mul20.i'de", align 8
  %313 = load double, double* %"mul20.i'de", align 8

  %314 = load i64, i64* %"iv13'ac", align 8
  %315 = load i64, i64* %"iv11'ac", align 8
  %316 = load i64, i64* %"iv5'ac", align 8
  %317 = load i64, i64* %"iv3'ac", align 8
  %conv17_unwrap63 = sext i32 %n to i64
  ; n - 1
  %_unwrap64 = add nsw i64 %conv17_unwrap63, -1
  ; n
  %318 = add nuw i64 %_unwrap64, 1
  %conv4_unwrap65 = sext i32 %k to i64
  %_unwrap66 = add nsw i64 %conv4_unwrap65, -1
  ; k
  %319 = add nuw i64 %_unwrap66, 1
  %320 = mul nuw nsw i64 1, %319
  ; k * n
  %321 = mul nuw nsw i64 %320, %318
  %322 = load double*, double** %_cache, align 8, !dereferenceable !51, !invariant.group !43
  %323 = load i64, i64* %"iv5'ac", align 8
  ; k
  %324 = mul nuw nsw i64 1, %319
  %325 = load i64, i64* %"iv3'ac", align 8
  %326 = mul nuw nsw i64 %324, %318
  ; ik
  %327 = mul nuw nsw i64 %323, 1
  %328 = add nuw nsw i64 0, %327
  ; ix * k
  %329 = mul nuw nsw i64 %325, %324
  ; ik + ix * k
  %330 = add nuw nsw i64 %328, %329
  %331 = load i64, i64* %"iv13'ac", align 8
  %332 = load i64, i64* %"iv11'ac", align 8
  %333 = load i64, i64* %"iv5'ac", align 8
  %334 = load i64, i64* %"iv3'ac", align 8
  %wide.trip.count.i119_unwrap68 = zext i32 %d to i64
  %_unwrap69 = add nsw i64 %wide.trip.count.i119_unwrap68, -1
  ; d
  %_unwrap70 = add nuw nsw i64 %_unwrap69, 1
  ; (ik + ix * k) * d
  %335 = mul nuw nsw i64 %330, %_unwrap70
  %336 = getelementptr inbounds double, double* %322, i64 %335
  %337 = getelementptr inbounds double, double* %336, i64 %315
  %338 = load double, double* %337, align 8, !invariant.group !61
  ;; bookmark4
  %m0diffe71 = fmul fast double %313, %338
  %339 = load i64, i64* %"iv13'ac", align 8
  %340 = load i64, i64* %"iv11'ac", align 8
  %341 = load i64, i64* %"iv5'ac", align 8
  %342 = load i64, i64* %"iv3'ac", align 8
  ;; ik * tri_size
  %mul34_unwrap = mul nsw i64 %341, %conv
  %arrayidx35_unwrap = getelementptr inbounds double, double* %Ls, i64 %mul34_unwrap
  ;; 2 * d
  %mul8.i_unwrap = shl nuw nsw i32 %d, 1
  %_unwrap72 = trunc i64 %340 to i32
  %_unwrap73 = xor i32 %_unwrap72, -1
  ;; 2 * d + (i ^ -1)
  %sub9.i_unwrap = add i32 %mul8.i_unwrap, %_unwrap73
  %_unwrap74 = trunc i64 %340 to i32
  ;; (2 * d + (i ^ -1)) * i
  %mul10.i_unwrap = mul nsw i32 %sub9.i_unwrap, %_unwrap74
  %div.i_unwrap = sdiv i32 %mul10.i_unwrap, 2
  %_unwrap75 = sext i32 %div.i_unwrap to i64
  %_unwrap76 = add i64 %_unwrap75, %339
  %arrayidx17.i_unwrap = getelementptr inbounds double, double* %arrayidx35_unwrap, i64 %_unwrap76
  %_unwrap77 = load double, double* %arrayidx17.i_unwrap, align 8, !tbaa !3, !invariant.group !53
  %m1diffe78 = fmul fast double %313, %_unwrap77
  ; %printfc4ll = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([13 x i8], [13 x i8]* @.flstr, i64 0, i64 0), double %m1diffe78)
  store double 0.000000e+00, double* %"mul20.i'de", align 8

  %343 = load double, double* %"'de79", align 8
  %344 = fadd fast double %343, %m0diffe71
  store double %344, double* %"'de79", align 8

  %345 = load double, double* %"'de59", align 8
  %346 = fadd fast double %345, %m1diffe78
  store double %346, double* %"'de59", align 8
  %347 = load double, double* %"'de79", align 8
  store double 0.000000e+00, double* %"'de79", align 8

  %348 = load i64, i64* %"iv13'ac", align 8
  %349 = load i64, i64* %"iv11'ac", align 8
  %350 = load i64, i64* %"iv5'ac", align 8
  %351 = load i64, i64* %"iv3'ac", align 8
  %"arrayidx35'ipg_unwrap" = getelementptr inbounds double, double* %"Ls'", i64 %mul34_unwrap
  %"arrayidx17.i'ipg_unwrap" = getelementptr inbounds double, double* %"arrayidx35'ipg_unwrap", i64 %_unwrap76
  %352 = load double, double* %"arrayidx17.i'ipg_unwrap", align 8
  %353 = fadd fast double %352, %347
  store double %353, double* %"arrayidx17.i'ipg_unwrap", align 8

  %354 = load double, double* %"'de62", align 8
  store double 0.000000e+00, double* %"'de62", align 8
  %355 = load double, double* %"arrayidx15.i'ipg_unwrap", align 8
  %356 = fadd fast double %355, %354
  store double %356, double* %"arrayidx15.i'ipg_unwrap", align 8
  %357 = load i64, i64* %"iv13'ac", align 8
  %358 = icmp eq i64 %357, 0
  %359 = xor i1 %358, true
  br i1 %358, label %invertfor.body13.lr.ph.i, label %incinvertfor.body13.i

incinvertfor.body13.i:                            ; preds = %invertfor.body13.i
  %360 = load i64, i64* %"iv13'ac", align 8
  %361 = add nsw i64 %360, -1
  store i64 %361, i64* %"iv13'ac", align 8
  br label %invertfor.body13.i

invertcQtimesx.exit.loopexit:                     ; preds = %invertcQtimesx.exit
  %362 = load double, double* %".pre'de", align 8

  store double 0.000000e+00, double* %".pre'de", align 8

  %363 = load double, double* %"'ipc40", align 8
  %364 = fadd fast double %363, %362
  store double %364, double* %"'ipc40", align 8
  ; %printfc4ll = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([13 x i8], [13 x i8]* @.flstr, i64 0, i64 0), double %364)

  %365 = load i64, i64* %"iv5'ac", align 8
  %366 = load i64, i64* %"iv3'ac", align 8
  %wide.trip.count.i119_unwrap80 = zext i32 %d to i64
  ; d - 1
  %_unwrap81 = add nsw i64 %wide.trip.count.i119_unwrap80, -1
  br label %mergeinvertfor.body7.i_cQtimesx.exit.loopexit

mergeinvertfor.body7.i_cQtimesx.exit.loopexit:    ; preds = %invertcQtimesx.exit.loopexit
  ;; iv11'ac is id
  store i64 %_unwrap81, i64* %"iv11'ac", align 8
  br label %invertfor.cond5.loopexit.i

invertcQtimesx.exit:                              ; preds = %invertsqnorm.exit, %invertfor.body.i109.preheader
  ;; DEBUG_POINT_6
  %367 = load double, double* %"mul.i102'de", align 8
  %368 = load i64, i64* %"iv5'ac", align 8
  %369 = load i64, i64* %"iv3'ac", align 8
  %conv17_unwrap87 = sext i32 %n to i64
  ; n - 1
  %_unwrap88 = add nsw i64 %conv17_unwrap87, -1
  ; n
  %370 = add nuw i64 %_unwrap88, 1
  %conv4_unwrap89 = sext i32 %k to i64
  ; k - 1
  %_unwrap90 = add nsw i64 %conv4_unwrap89, -1
  %371 = add nuw i64 %_unwrap90, 1
  %372 = mul nuw nsw i64 %371, %370
  %373 = load double*, double** %_cache82, align 8, !dereferenceable !51, !invariant.group !44
  ; ik
  %374 = load i64, i64* %"iv5'ac", align 8
  ; ix
  %375 = load i64, i64* %"iv3'ac", align 8
  ; unused
  %376 = mul nuw nsw i64 %371, %370
  ; ix * (k - 1 + 1) -> ix * k
  %377 = mul nuw nsw i64 %375, %371
  ; ix * k + ik
  %378 = add nuw nsw i64 %374, %377
  %379 = getelementptr inbounds double, double* %373, i64 %378
  %380 = load double, double* %379, align 8, !invariant.group !54

  %m0diffe91 = fmul fast double %367, %380
  %m1diffe92 = fmul fast double %367, %380
  store double 0.000000e+00, double* %"mul.i102'de", align 8
  %381 = load double, double* %"'de35", align 8
  %382 = fadd fast double %381, %m0diffe91
  store double %382, double* %"'de35", align 8
  %383 = load double, double* %"'de35", align 8
  %384 = fadd fast double %383, %m1diffe92
  store double %384, double* %"'de35", align 8
  %385 = load double, double* %"add'de", align 8
  store double 0.000000e+00, double* %"add'de", align 8
  %386 = load double, double* %"'de93", align 8
  %387 = fadd fast double %386, %385
  store double %387, double* %"'de93", align 8
  %388 = load double, double* %"'de94", align 8
  %389 = fadd fast double %388, %385
  store double %389, double* %"'de94", align 8
  %390 = load double, double* %"'de94", align 8
  store double 0.000000e+00, double* %"'de94", align 8
  ;; DEBUG_POINT_7
  %391 = load i64, i64* %"iv5'ac", align 8
  %392 = load i64, i64* %"iv3'ac", align 8
  %"arrayidx39'ipg_unwrap" = getelementptr inbounds double, double* %"'ipc", i64 %391
  %393 = load double, double* %"arrayidx39'ipg_unwrap", align 8
  %394 = fadd fast double %393, %390
  store double %394, double* %"arrayidx39'ipg_unwrap", align 8

  %395 = load double, double* %"'de93", align 8
  store double 0.000000e+00, double* %"'de93", align 8

  %396 = load i64, i64* %"iv5'ac", align 8
  %397 = load i64, i64* %"iv3'ac", align 8
  %"arrayidx38'ipg_unwrap" = getelementptr inbounds double, double* %"alphas'", i64 %396
  %398 = load double, double* %"arrayidx38'ipg_unwrap", align 8
  %399 = fadd fast double %398, %395
  store double %399, double* %"arrayidx38'ipg_unwrap", align 8

  %400 = load double, double* %"'de35", align 8
  store double 0.000000e+00, double* %"'de35", align 8

  %401 = load i64, i64* %"iv5'ac", align 8
  %402 = load i64, i64* %"iv3'ac", align 8
  ;; d > 0, always true
  %cmp10.i_unwrap = icmp sgt i32 %d, 0
  ; always false
  %403 = xor i1 %cmp10.i_unwrap, true
  ; unused
  %404 = select fast i1 %cmp10.i_unwrap, double %400, double 0.000000e+00
  %405 = load double, double* %".pre'de", align 8
  %406 = fadd fast double %405, %400
  %407 = select fast i1 %cmp10.i_unwrap, double %406, double %405
  store double %407, double* %".pre'de", align 8
  ; unused
  %408 = select fast i1 %403, double %400, double 0.000000e+00
  %409 = load double, double* %"'de34", align 8
  %410 = fadd fast double %409, %400
  %411 = select fast i1 %cmp10.i_unwrap, double %409, double %410
  store double %411, double* %"'de34", align 8
  br i1 %cmp10.i_unwrap, label %invertcQtimesx.exit.loopexit, label %invertfor.body23

invertfor.body.i109.preheader:                    ; preds = %invertfor.body.i109
  br label %invertcQtimesx.exit

invertfor.body.i109:                              ; preds = %mergeinvertfor.body.i109_sqnorm.exit.loopexit, %incinvertfor.body.i109
  ;; DEBUG_POINT_5
  %412 = load double, double* %"add.i106'de", align 8
  store double 0.000000e+00, double* %"add.i106'de", align 8
  %413 = load double, double* %"res.017.i'de", align 8
  %414 = fadd fast double %413, %412
  store double %414, double* %"res.017.i'de", align 8

  %415 = load double, double* %"mul5.i'de", align 8
  %416 = fadd fast double %415, %412
  store double %416, double* %"mul5.i'de", align 8
  %417 = load double, double* %"mul5.i'de", align 8
  %418 = load i64, i64* %"iv15'ac", align 8
  %419 = load i64, i64* %"iv5'ac", align 8
  %420 = load i64, i64* %"iv3'ac", align 8
  %conv17_unwrap103 = sext i32 %n to i64
  %_unwrap104 = add nsw i64 %conv17_unwrap103, -1
  ;; n - 1 + 1
  %421 = add nuw i64 %_unwrap104, 1
  %conv4_unwrap105 = sext i32 %k to i64
  ;; k - 1
  %_unwrap106 = add nsw i64 %conv4_unwrap105, -1
  ;; k
  %422 = add nuw i64 %_unwrap106, 1
  %423 = mul nuw nsw i64 1, %422
  %424 = mul nuw nsw i64 %423, %421
  %425 = load double*, double** %_cache98, align 8, !dereferenceable !51, !invariant.group !45
  ;; ik
  %426 = load i64, i64* %"iv5'ac", align 8
  ;; k * 1
  %427 = mul nuw nsw i64 1, %422
  ;; ix
  %428 = load i64, i64* %"iv3'ac", align 8
  ;; unused; k * n
  %429 = mul nuw nsw i64 %427, %421
  %430 = mul nuw nsw i64 %426, 1
  ;; ik * 1 + 0
  %431 = add nuw nsw i64 0, %430
  ;; ix * k
  %432 = mul nuw nsw i64 %428, %427
  ;; ik + ix * k
  %433 = add nuw nsw i64 %431, %432
  %434 = load i64, i64* %"iv15'ac", align 8
  %435 = load i64, i64* %"iv5'ac", align 8
  %436 = load i64, i64* %"iv3'ac", align 8
  %wide.trip.count.i119_unwrap107 = zext i32 %d to i64
  %_unwrap108 = add nsw i64 %wide.trip.count.i119_unwrap107, -2
  ;; d - 1
  %_unwrap109 = add nuw nsw i64 %_unwrap108, 1
  ;; (ik + ix * k) * (d - 1)
  %437 = mul nuw nsw i64 %433, %_unwrap109
  %438 = getelementptr inbounds double, double* %425, i64 %437
  ;; &Qxcentered_cache[id] (cache98)
  %439 = getelementptr inbounds double, double* %438, i64 %418
  %440 = load double, double* %439, align 8, !invariant.group !62
  %m0diffe110 = fmul fast double %417, %440
  %m1diffe111 = fmul fast double %417, %440
  store double 0.000000e+00, double* %"mul5.i'de", align 8

  %441 = load double, double* %"'de112", align 8
  %442 = fadd fast double %441, %m0diffe110
  store double %442, double* %"'de112", align 8
  %443 = load double, double* %"'de112", align 8
  %444 = fadd fast double %443, %m1diffe111
  store double %444, double* %"'de112", align 8
  %445 = load double, double* %"'de112", align 8
  store double 0.000000e+00, double* %"'de112", align 8

  %446 = load i64, i64* %"iv15'ac", align 8
  %447 = load i64, i64* %"iv5'ac", align 8
  %448 = load i64, i64* %"iv3'ac", align 8
  ;; id + 1
  %iv.next16_unwrap = add nuw nsw i64 %446, 1
  %"arrayidx2.i'ipg_unwrap" = getelementptr inbounds double, double* %"'ipc40", i64 %iv.next16_unwrap
  %449 = load double, double* %"arrayidx2.i'ipg_unwrap", align 8
  %450 = fadd fast double %449, %445
  store double %450, double* %"arrayidx2.i'ipg_unwrap", align 8

  %451 = load double, double* %"res.017.i'de", align 8
  store double 0.000000e+00, double* %"res.017.i'de", align 8
  %452 = load i64, i64* %"iv15'ac", align 8
  ;; id == 0
  %453 = icmp eq i64 %452, 0
  ;; unused
  %454 = xor i1 %453, true
  ;; unused
  %455 = select fast i1 %453, double %451, double 0.000000e+00
  %456 = load double, double* %"mul.i102'de", align 8
  %457 = fadd fast double %456, %451
  %458 = select fast i1 %453, double %457, double %456
  store double %458, double* %"mul.i102'de", align 8
  ;; unused
  %459 = select fast i1 %454, double %451, double 0.000000e+00
  %460 = load double, double* %"add.i106'de", align 8
  %461 = fadd fast double %460, %451
  %462 = select fast i1 %453, double %460, double %461
  store double %462, double* %"add.i106'de", align 8
  br i1 %453, label %invertfor.body.i109.preheader, label %incinvertfor.body.i109

incinvertfor.body.i109:                           ; preds = %invertfor.body.i109
  %463 = load i64, i64* %"iv15'ac", align 8
  %464 = add nsw i64 %463, -1
  store i64 %464, i64* %"iv15'ac", align 8
  br label %invertfor.body.i109

invertsqnorm.exit.loopexit:                       ; preds = %invertsqnorm.exit
  %465 = load i64, i64* %"iv5'ac", align 8
  %466 = load i64, i64* %"iv3'ac", align 8
  %wide.trip.count.i119_unwrap114 = zext i32 %d to i64
  ;; d - 2
  %_unwrap115 = add nsw i64 %wide.trip.count.i119_unwrap114, -2
  br label %mergeinvertfor.body.i109_sqnorm.exit.loopexit

mergeinvertfor.body.i109_sqnorm.exit.loopexit:    ; preds = %invertsqnorm.exit.loopexit
  ;; iv15'ac iterates over d - 2 .. 0
  store i64 %_unwrap115, i64* %"iv15'ac", align 8
  br label %invertfor.body.i109

invertsqnorm.exit:                                ; preds = %mergeinvertfor.body23_for.end.loopexit, %incinvertfor.body23
  ;; DEBUG_POINT_4
  %467 = load i64, i64* %"iv5'ac", align 8
  %468 = load i64, i64* %"iv3'ac", align 8
  %"arrayidx44'ipg_unwrap" = getelementptr inbounds double, double* %"'ipc116", i64 %467
  %469 = load double, double* %"arrayidx44'ipg_unwrap", align 8
  store double 0.000000e+00, double* %"arrayidx44'ipg_unwrap", align 8
  %470 = load double, double* %"sub43'de", align 8
  %471 = fadd fast double %470, %469
  store double %471, double* %"sub43'de", align 8
  %472 = load double, double* %"sub43'de", align 8
  %473 = fneg fast double %472
  store double 0.000000e+00, double* %"sub43'de", align 8
  ;; sub43'de not used after this
  %474 = load double, double* %"add'de", align 8
  %475 = fadd fast double %474, %472
  store double %475, double* %"add'de", align 8

  %476 = load double, double* %"mul42'de", align 8
  %477 = fadd fast double %476, %473
  store double %477, double* %"mul42'de", align 8
  %478 = load double, double* %"mul42'de", align 8
  %m0differes.0.lcssa.i = fmul fast double %478, 5.000000e-01
  store double 0.000000e+00, double* %"mul42'de", align 8
  %479 = load double, double* %"res.0.lcssa.i'de", align 8
  %480 = fadd fast double %479, %m0differes.0.lcssa.i
  store double %480, double* %"res.0.lcssa.i'de", align 8
  %481 = load double, double* %"res.0.lcssa.i'de", align 8
  store double 0.000000e+00, double* %"res.0.lcssa.i'de", align 8
  %482 = load i64, i64* %"iv5'ac", align 8
  %483 = load i64, i64* %"iv3'ac", align 8
  ;; always true
  %cmp15.i_unwrap = icmp sgt i32 %d, 1
  %484 = xor i1 %cmp15.i_unwrap, true
  %485 = select fast i1 %cmp15.i_unwrap, double %481, double 0.000000e+00
  %486 = load double, double* %"add.i106'de", align 8
  %487 = fadd fast double %486, %481
  %488 = select fast i1 %cmp15.i_unwrap, double %487, double %486
  store double %488, double* %"add.i106'de", align 8
  ;; always false, will be 0.0
  %489 = select fast i1 %484, double %481, double 0.000000e+00
  %490 = load double, double* %"mul.i102'de", align 8
  %491 = fadd fast double %490, %481
  %492 = select fast i1 %cmp15.i_unwrap, double %490, double %491
  store double %492, double* %"mul.i102'de", align 8
  ;; always true
  br i1 %cmp15.i_unwrap, label %invertsqnorm.exit.loopexit, label %invertcQtimesx.exit

invertfor.end.loopexit:                           ; preds = %invertfor.end
  %493 = load double, double* %".pre146'de", align 8
  store double 0.000000e+00, double* %".pre146'de", align 8
  %494 = load double, double* %"'ipc116", align 8
  %495 = fadd fast double %494, %493
  store double %495, double* %"'ipc116", align 8
  %496 = load i64, i64* %"iv3'ac", align 8
  ;; k - 1
  %_unwrap118 = add nsw i64 %conv4, -1
  br label %mergeinvertfor.body23_for.end.loopexit

mergeinvertfor.body23_for.end.loopexit:           ; preds = %invertfor.end.loopexit
  ;; iv5'ac is ik
  store i64 %_unwrap118, i64* %"iv5'ac", align 8
  br label %invertsqnorm.exit

invertfor.end:                                    ; preds = %invertarr_max.exit.i, %invertfor.body.i.i.preheader
  ;; DEBUG_POINT_3
  %497 = load double, double* %"'de31", align 8
  store double 0.000000e+00, double* %"'de31", align 8
  ;; always false, true xor true
  %498 = xor i1 %cmp37.i, true
  ;; unused
  %499 = select fast i1 %cmp37.i, double %497, double 0.000000e+00
  %500 = load double, double* %".pre146'de", align 8
  %501 = fadd fast double %500, %497
  ;; always true, will add %497
  %502 = select fast i1 %cmp37.i, double %501, double %500
  store double %502, double* %".pre146'de", align 8

  %503 = select fast i1 %498, double %497, double 0.000000e+00
  %504 = load double, double* %"'de30", align 8
  %505 = fadd fast double %504, %497
  ;; always true, won't add anything
  %506 = select fast i1 %cmp37.i, double %504, double %505
  store double %506, double* %"'de30", align 8
  %507 = load double, double* %"'de33", align 8
  store double 0.000000e+00, double* %"'de33", align 8
  ;; unused, always %507
  %508 = select fast i1 %cmp37.i, double %507, double 0.000000e+00
  %509 = load double, double* %"'de35", align 8
  %510 = fadd fast double %509, %507
  %511 = select fast i1 %cmp37.i, double %510, double %509
  store double %511, double* %"'de35", align 8
  ; unused
  %512 = select fast i1 %498, double %507, double 0.000000e+00
  %513 = load double, double* %"'de32", align 8
  %514 = fadd fast double %513, %507
  ;; always true, won't add anything
  %515 = select fast i1 %cmp37.i, double %513, double %514
  store double %515, double* %"'de32", align 8
  ;; always true
  br i1 %cmp37.i, label %invertfor.end.loopexit, label %invertfor.cond19.preheader

invertfor.body.i.i.preheader:                     ; preds = %invertfor.body.i.i
  br label %invertfor.end

invertfor.body.i.i:                               ; preds = %mergeinvertfor.body.i.i_arr_max.exit.i.loopexit, %incinvertfor.body.i.i
  ;; adjoint loop for arr_max
  %516 = load i64, i64* %"iv17'ac", align 8
  %517 = load i64, i64* %"iv3'ac", align 8
  ;; ik + 1
  %iv.next18_unwrap = add nuw nsw i64 %516, 1
  %518 = icmp eq i64 %_unwrap130, %iv.next18_unwrap
  %519 = load double, double* %"m.1.i.i'de", align 8
  ;; unused
  %520 = select fast i1 %518, double %519, double 0.000000e+00
  %521 = load double, double* %"'de131", align 8
  %522 = fadd fast double %521, %519
  %523 = select fast i1 %518, double %522, double %521
  store double %523, double* %"'de131", align 8
  %524 = load double, double* %"'de131", align 8
  store double 0.000000e+00, double* %"'de131", align 8
  ;; 'de131 is unused after this point
  %525 = load i64, i64* %"iv17'ac", align 8
  %526 = load i64, i64* %"iv3'ac", align 8
  %"arrayidx1.i.i'ipg_unwrap" = getelementptr inbounds double, double* %"'ipc116", i64 %iv.next18_unwrap
  %527 = load double, double* %"arrayidx1.i.i'ipg_unwrap", align 8
  %528 = fadd fast double %527, %524
  store double %528, double* %"arrayidx1.i.i'ipg_unwrap", align 8
  ;; unused
  %529 = load double, double* %"m.015.i.i'de", align 8
  %530 = load i64, i64* %"iv17'ac", align 8
  ;; ik == 0
  %531 = icmp eq i64 %530, 0
  ;; unused`
  %532 = xor i1 %531, true
  %533 = load double, double* %"m.1.i.i'de", align 8
  %534 = select fast i1 %531, double 0.000000e+00, double %533
  ;; m1iide = (ik == 0) ? 0 : m1iide
  store double %534, double* %"m.1.i.i'de", align 8
  %535 = icmp eq i64 %_unwrap130, 0
  %536 = select fast i1 %535, double %533, double 0.000000e+00
  %537 = select fast i1 %531, double %536, double 0.000000e+00
  %538 = load double, double* %"'de31", align 8
  %539 = fadd fast double %538, %536
  %540 = select fast i1 %531, double %539, double %538
  store double %540, double* %"'de31", align 8

  br i1 %531, label %invertfor.body.i.i.preheader, label %incinvertfor.body.i.i

incinvertfor.body.i.i:                            ; preds = %invertfor.body.i.i
  %541 = load i64, i64* %"iv17'ac", align 8
  %542 = add nsw i64 %541, -1
  store i64 %542, i64* %"iv17'ac", align 8
  br label %invertfor.body.i.i

invertarr_max.exit.i.loopexit:                    ; preds = %invertarr_max.exit.i
  %543 = load i64, i64* %"iv3'ac", align 8
  %conv17_unwrap122 = sext i32 %n to i64
  %_unwrap123 = add nsw i64 %conv17_unwrap122, -1
  ;; unused
  %544 = add nuw i64 %_unwrap123, 1
  %545 = load i1*, i1** %"cmp2.i.i!manual_lcssa_cache", align 8, !dereferenceable !51, !invariant.group !46
  %546 = load i64, i64* %"iv3'ac", align 8
  %547 = getelementptr inbounds i1, i1* %545, i64 %546
  %548 = load i1, i1* %547, align 1, !invariant.group !55
  %549 = load i64, i64* %"iv3'ac", align 8
  %wide.trip.count.i.i_unwrap = zext i32 %k to i64
  ;; k - 2
  %_unwrap124 = add nsw i64 %wide.trip.count.i.i_unwrap, -2
  ;; k - 1
  %iv.next18_unwrap125 = add nuw nsw i64 %_unwrap124, 1
  ;; unused
  %550 = add nuw i64 %_unwrap123, 1
  %551 = load i64*, i64** %"!manual_lcssa126_cache", align 8, !dereferenceable !51, !invariant.group !47
  %552 = load i64, i64* %"iv3'ac", align 8
  %553 = getelementptr inbounds i64, i64* %551, i64 %552
  %554 = load i64, i64* %553, align 8, !invariant.group !56
  %_unwrap130 = select i1 %548, i64 %iv.next18_unwrap125, i64 %554
  %555 = load i64, i64* %"iv3'ac", align 8
  br label %mergeinvertfor.body.i.i_arr_max.exit.i.loopexit

mergeinvertfor.body.i.i_arr_max.exit.i.loopexit:  ; preds = %invertarr_max.exit.i.loopexit
  ;; iv17'ac is ik
  store i64 %_unwrap124, i64* %"iv17'ac", align 8
  br label %invertfor.body.i.i

invertarr_max.exit.i:                             ; preds = %invertlog_sum_exp.exit_phimerge, %invertfor.body.preheader.i
  %556 = load double, double* %"m.0.lcssa.i.i'de", align 8
  store double 0.000000e+00, double* %"m.0.lcssa.i.i'de", align 8
  %557 = load i64, i64* %"iv3'ac", align 8
  ;; always true (k > 1)
  %cmp13.i.i_unwrap = icmp sgt i32 %k, 1
  ;; always false
  %558 = xor i1 %cmp13.i.i_unwrap, true
  ; unused
  %559 = select fast i1 %cmp13.i.i_unwrap, double %556, double 0.000000e+00
  %560 = load double, double* %"m.1.i.i'de", align 8
  %561 = fadd fast double %560, %556
  %562 = select fast i1 %cmp13.i.i_unwrap, double %561, double %560
  store double %562, double* %"m.1.i.i'de", align 8
  %563 = select fast i1 %558, double %556, double 0.000000e+00

  %564 = load double, double* %"'de31", align 8
  %565 = fadd fast double %564, %556
  %566 = select fast i1 %cmp13.i.i_unwrap, double %564, double %565
  store double %566, double* %"'de31", align 8
  br i1 %cmp13.i.i_unwrap, label %invertarr_max.exit.i.loopexit, label %invertfor.end

invertfor.body.preheader.i:                       ; preds = %staging, %invertfor.body.for.body_crit_edge.i.preheader
  ;; DEBUG_POINT_2
  %567 = load double, double* %"add.i136'de", align 8
  store double 0.000000e+00, double* %"add.i136'de", align 8
  %568 = load double, double* %"'de133", align 8
  %569 = fadd fast double %568, %567
  store double %569, double* %"'de133", align 8
  %570 = load double, double* %"'de133", align 8
  store double 0.000000e+00, double* %"'de133", align 8
  ;; %"'de133" unused after this point
  %571 = load i64, i64* %"iv3'ac", align 8
  %conv17_unwrap137 = sext i32 %n to i64
  ;; n - 1
  %_unwrap138 = add nsw i64 %conv17_unwrap137, -1
  ;; unused
  %572 = add nuw i64 %_unwrap138, 1
  %573 = load double*, double** %sub.i135_cache, align 8, !dereferenceable !51, !invariant.group !48
  %574 = load i64, i64* %"iv3'ac", align 8
  %575 = getelementptr inbounds double, double* %573, i64 %574
  %576 = load double, double* %575, align 8, !invariant.group !57
  %577 = call fast double @llvm.exp.f64(double %576)
  %578 = fmul fast double %570, %577
  %579 = load double, double* %"sub.i135'de", align 8
  %580 = fadd fast double %579, %578
  store double %580, double* %"sub.i135'de", align 8
  %581 = load double, double* %"sub.i135'de", align 8
  %582 = fneg fast double %581
  store double 0.000000e+00, double* %"sub.i135'de", align 8
  %583 = load double, double* %"'de31", align 8
  %584 = fadd fast double %583, %581
  store double %584, double* %"'de31", align 8
  %585 = load double, double* %"m.0.lcssa.i.i'de", align 8
  %586 = fadd fast double %585, %582
  store double %586, double* %"m.0.lcssa.i.i'de", align 8
  br label %invertarr_max.exit.i

invertfor.body.for.body_crit_edge.i.preheader:    ; preds = %invertfor.body.for.body_crit_edge.i
  br label %invertfor.body.preheader.i

invertfor.body.for.body_crit_edge.i:              ; preds = %mergeinvertfor.body.for.body_crit_edge.i_log_sum_exp.exit.loopexit, %incinvertfor.body.for.body_crit_edge.i
  ;; This should be just inside the adjoint k loop for grad_logsumexp
  %587 = load double, double* %"add.i'de", align 8
  store double 0.000000e+00, double* %"add.i'de", align 8
  %588 = load double, double* %"add.i138'de", align 8
  %589 = fadd fast double %588, %587
  store double %589, double* %"add.i138'de", align 8
  %590 = load double, double* %"'de139", align 8
  %591 = fadd fast double %590, %587
  store double %591, double* %"'de139", align 8
  %592 = load double, double* %"'de139", align 8
  store double 0.000000e+00, double* %"'de139", align 8
  ;; DEBUG_POINT_0
  ;; %"'de139" is unused after this
  ; %printfc4ll = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([13 x i8], [13 x i8]* @.flstr, i64 0, i64 0), double %587)
  ;; iv19'ac is ik
  %593 = load i64, i64* %"iv19'ac", align 8
  %594 = load i64, i64* %"iv3'ac", align 8
  %conv17_unwrap143 = sext i32 %n to i64
  %_unwrap144 = add nsw i64 %conv17_unwrap143, -1
  ;; n - 1 + 1
  %595 = add nuw i64 %_unwrap144, 1
  %wide.trip.count.i.i_unwrap145 = zext i32 %k to i64
  %_unwrap146 = add nsw i64 %wide.trip.count.i.i_unwrap145, -2
  ;; k - 2 + 1
  %596 = add nuw i64 %_unwrap146, 1
  ;; (k - 1) * n, unused
  %597 = mul nuw nsw i64 %596, %595
  %598 = load double*, double** %sub.i_cache, align 8, !dereferenceable !51, !invariant.group !49
  ;; iv19 is ik, iv3 is ix
  %599 = load i64, i64* %"iv19'ac", align 8
  %600 = load i64, i64* %"iv3'ac", align 8
  ;; %601 is unused
  %601 = mul nuw nsw i64 %596, %595
  %602 = mul nuw nsw i64 %600, %596
  ;; ix * (k - 1) + ik
  %603 = add nuw nsw i64 %599, %602

  %604 = getelementptr inbounds double, double* %598, i64 %603
  %605 = load double, double* %604, align 8, !invariant.group !58
  %606 = call fast double @llvm.exp.f64(double %605)
  %607 = fmul fast double %592, %606
  %608 = load double, double* %"sub.i'de", align 8
  %609 = fadd fast double %608, %607
  store double %609, double* %"sub.i'de", align 8
  %610 = load double, double* %"sub.i'de", align 8
  %611 = fneg fast double %610
  store double 0.000000e+00, double* %"sub.i'de", align 8
  %612 = load double, double* %".pre.i101'de", align 8
  %613 = fadd fast double %612, %610
  store double %613, double* %".pre.i101'de", align 8
  %614 = load double, double* %"m.0.lcssa.i.i'de", align 8
  %615 = fadd fast double %614, %611
  store double %615, double* %"m.0.lcssa.i.i'de", align 8
  %616 = load double, double* %".pre.i101'de", align 8
  store double 0.000000e+00, double* %".pre.i101'de", align 8
  %617 = load i64, i64* %"iv19'ac", align 8
  %618 = load i64, i64* %"iv3'ac", align 8
  ;; main_termb[ik + 1] += prei101'de
  %iv.next20_unwrap = add nuw nsw i64 %617, 1
  %"arrayidx.phi.trans.insert.i'ipg_unwrap" = getelementptr inbounds double, double* %"'ipc116", i64 %iv.next20_unwrap
  %619 = load double, double* %"arrayidx.phi.trans.insert.i'ipg_unwrap", align 8
  %620 = fadd fast double %619, %616
  store double %620, double* %"arrayidx.phi.trans.insert.i'ipg_unwrap", align 8

  ; %printfcall = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([13 x i8], [13 x i8]* @.flstr, i64 0, i64 0), double %620)
  ;; end of main_termb[ik + 1] += prei101'de

  %621 = load double, double* %"add.i138'de", align 8
  store double 0.000000e+00, double* %"add.i138'de", align 8
  %622 = load i64, i64* %"iv19'ac", align 8
  ;; true iff at the end of the k loop
  %623 = icmp eq i64 %622, 0
  %624 = xor i1 %623, true
  ;; unused
  %625 = select fast i1 %623, double %621, double 0.000000e+00
  %626 = load double, double* %"add.i136'de", align 8
  %627 = fadd fast double %626, %621
  %628 = select fast i1 %623, double %627, double %626
  store double %628, double* %"add.i136'de", align 8
  ; unused
  %629 = select fast i1 %624, double %621, double 0.000000e+00
  %630 = load double, double* %"add.i'de", align 8
  %631 = fadd fast double %630, %621
  %632 = select fast i1 %623, double %630, double %631
  store double %632, double* %"add.i'de", align 8
  ;; DEBUG_POINT_1
  br i1 %623, label %invertfor.body.for.body_crit_edge.i.preheader, label %incinvertfor.body.for.body_crit_edge.i

incinvertfor.body.for.body_crit_edge.i:           ; preds = %invertfor.body.for.body_crit_edge.i
  %633 = load i64, i64* %"iv19'ac", align 8
  %634 = add nsw i64 %633, -1
  store i64 %634, i64* %"iv19'ac", align 8
  ;; goes back into the k loop for main_termb (grad_logsumexp)
  br label %invertfor.body.for.body_crit_edge.i

invertlog_sum_exp.exit.loopexit:                  ; preds = %staging
  %635 = load i64, i64* %"iv3'ac", align 8
  %wide.trip.count.i.i_unwrap148 = zext i32 %k to i64
  %_unwrap149 = add nsw i64 %wide.trip.count.i.i_unwrap148, -2
  br label %mergeinvertfor.body.for.body_crit_edge.i_log_sum_exp.exit.loopexit

mergeinvertfor.body.for.body_crit_edge.i_log_sum_exp.exit.loopexit: ; preds = %invertlog_sum_exp.exit.loopexit
  store i64 %_unwrap149, i64* %"iv19'ac", align 8
  br label %invertfor.body.for.body_crit_edge.i

invertlog_sum_exp.exit:                           ; preds = %mergeinvertfor.cond19.preheader_for.end50.loopexit, %incinvertfor.cond19.preheader
  ;; %636 starts at 1.0
  %636 = load double, double* %"add47'de", align 8
  store double 0.000000e+00, double* %"add47'de", align 8
  %637 = load double, double* %"slse.0143'de", align 8
  %638 = fadd fast double %637, %636
  store double %638, double* %"slse.0143'de", align 8
  %639 = load double, double* %"add1.i'de", align 8
  %640 = fadd fast double %639, %636
  store double %640, double* %"add1.i'de", align 8
  %641 = load double, double* %"add1.i'de", align 8
  store double 0.000000e+00, double* %"add1.i'de", align 8
  ; %"add1.i'de" is unused after this

  %642 = load double, double* %"m.0.lcssa.i.i'de", align 8
  %643 = fadd fast double %642, %641
  store double %643, double* %"m.0.lcssa.i.i'de", align 8
  %644 = load double, double* %"'de150", align 8
  %645 = fadd fast double %644, %641
  store double %645, double* %"'de150", align 8
  ;; I think %646 is always 1
  %646 = load double, double* %"'de150", align 8
  store double 0.000000e+00, double* %"'de150", align 8
  ;; %"'de150" unusd after this
  %647 = load i64, i64* %"iv3'ac", align 8
  ; k == 1 (usually false)
  %exitcond.not.i99137_unwrap = icmp eq i32 %k, 1
  %conv17_unwrap151 = sext i32 %n to i64
  %_unwrap152 = add nsw i64 %conv17_unwrap151, -1
  %648 = add nuw i64 %_unwrap152, 1
  %649 = load double*, double** %sub.i135_cache, align 8, !dereferenceable !51, !invariant.group !48
  %650 = load i64, i64* %"iv3'ac", align 8
  %651 = getelementptr inbounds double, double* %649, i64 %650
  %652 = load double, double* %651, align 8, !invariant.group !57
  %653 = tail call double @llvm.exp.f64(double %652) #14
  ; exp(subicache[ix]) + 0.0
  %add.i136_unwrap = fadd double %653, 0.000000e+00
  ; n - 1 + 1, looks unused
  %654 = add nuw i64 %_unwrap152, 1
  %655 = load double*, double** %"add.i!manual_lcssa154_cache", align 8, !dereferenceable !51, !invariant.group !50
  %656 = load i64, i64* %"iv3'ac", align 8
  %657 = getelementptr inbounds double, double* %655, i64 %656
  ; %658 is semx (154cache[ix])
  %658 = load double, double* %657, align 8, !invariant.group !59
  ; %cmp37.i is always true (k > 0)
  br i1 %cmp37.i, label %invertlog_sum_exp.exit_phisplt, label %invertlog_sum_exp.exit_phirc158

;; This next group of BBs doesn't appear to do anything meaningful
invertlog_sum_exp.exit_phisplt:                   ; preds = %invertlog_sum_exp.exit
  ;; Pretty much always false (k == 1)
  br i1 %exitcond.not.i99137_unwrap, label %invertlog_sum_exp.exit_phirc, label %invertlog_sum_exp.exit_phirc153

invertlog_sum_exp.exit_phirc:                     ; preds = %invertlog_sum_exp.exit_phisplt
  br label %invertlog_sum_exp.exit_phimerge

invertlog_sum_exp.exit_phirc153:                  ; preds = %invertlog_sum_exp.exit_phisplt
  br label %invertlog_sum_exp.exit_phimerge

invertlog_sum_exp.exit_phirc158:                  ; preds = %invertlog_sum_exp.exit
  br label %invertlog_sum_exp.exit_phimerge

invertlog_sum_exp.exit_phimerge:                  ; preds = %invertlog_sum_exp.exit_phirc158, %invertlog_sum_exp.exit_phirc153, %invertlog_sum_exp.exit_phirc
  ;; This will take the value of %658, which starts as semx[ix]
  %659 = phi fast double [ %add.i136_unwrap, %invertlog_sum_exp.exit_phirc ], [ %658, %invertlog_sum_exp.exit_phirc153 ], [ 0.000000e+00, %invertlog_sum_exp.exit_phirc158 ]
  ;; Looks like log VJP, 1.0 / semx
  %660 = fdiv fast double %646, %659
  %661 = load double, double* %"semx.0.lcssa.i'de", align 8
  %662 = fadd fast double %661, %660
  ;; 0.0 + 1.0 / semx
  store double %662, double* %"semx.0.lcssa.i'de", align 8
  %663 = load double, double* %"semx.0.lcssa.i'de", align 8

  ; %printfcall = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([13 x i8], [13 x i8]* @.flstr, i64 0, i64 0), double %646)
  store double 0.000000e+00, double* %"semx.0.lcssa.i'de", align 8
  ;; %"semx.0.lcssa.i'de" unused after this.

  ;; always false (xor true, true)
  %anot1_ = xor i1 %cmp37.i, true
  ;; always false (false && true)
  %andVal0 = and i1 %exitcond.not.i99137_unwrap, %cmp37.i
  ;; always true (false xor true)
  %bnot1_ = xor i1 %exitcond.not.i99137_unwrap, true
  ;; always true
  %andVal1 = and i1 %bnot1_, %cmp37.i
  ;; %664 is unused
  %664 = select fast i1 %andVal1, double %663, double 0.000000e+00
  %665 = load double, double* %"add.i'de", align 8
  %666 = fadd fast double %665, %663
  ;; storing 1 / semx (154cache[ix])
  %667 = select fast i1 %andVal1, double %666, double %665
  ;; (starts at 0.0) + 1.0 / 154cache[ix]
  store double %667, double* %"add.i'de", align 8
  ;; unused
  %668 = select fast i1 %andVal0, double %663, double 0.000000e+00
  %669 = load double, double* %"add.i136'de", align 8
  %670 = fadd fast double %669, %663
  %671 = select fast i1 %andVal0, double %670, double %669
  ;; Looks like this is zero the first run
  store double %671, double* %"add.i136'de", align 8
  ;; always true
  br i1 %cmp37.i, label %staging, label %invertarr_max.exit.i

invertfor.end50.loopexit:                         ; preds = %invertfor.end50
  ; unwrap160 is n - 1
  %_unwrap160 = add nsw i64 %conv17, -1
  br label %mergeinvertfor.cond19.preheader_for.end50.loopexit

mergeinvertfor.cond19.preheader_for.end50.loopexit: ; preds = %invertfor.end50.loopexit
  ; iv3'ac is n - 1, I think it's ix in reverse
  store i64 %_unwrap160, i64* %"iv3'ac", align 8
  br label %invertlog_sum_exp.exit

invertfor.end50:                                  ; preds = %for.end50
  ; %672 is g (1.0)
  %672 = load double, double* %"err'", align 8
  store double 0.000000e+00, double* %"err'", align 8
  %673 = load double, double* %"slse.0.lcssa'de", align 8
  ; 0 + 1.0
  %674 = fadd fast double %673, %672
  store double %674, double* %"slse.0.lcssa'de", align 8
  ; 1.0
  %675 = load double, double* %"slse.0.lcssa'de", align 8
  ;; %"slse.0.lcssa'de" no longer used after this
  store double 0.000000e+00, double* %"slse.0.lcssa'de", align 8
  ;; %cmp140 always true (n > 0); 1.0
  %676 = select fast i1 %cmp140, double %675, double 0.000000e+00
  ;; At this point add47'de is 0.0
  %677 = load double, double* %"add47'de", align 8
  %678 = fadd fast double %677, %675
  %679 = select fast i1 %cmp140, double %678, double %677
  ;; Now add47.de is 1.0
  store double %679, double* %"add47'de", align 8
  br i1 %cmp140, label %invertfor.end50.loopexit, label %invertpreprocess_qs.exit

staging:                                          ; preds = %invertlog_sum_exp.exit_phimerge
  ;; false (k == 1)
  br i1 %exitcond.not.i99137_unwrap, label %invertfor.body.preheader.i, label %invertlog_sum_exp.exit.loopexit
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #12

attributes #0 = { norecurse nounwind readonly ssp uwtable "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nofree nosync nounwind willreturn }
attributes #2 = { nofree norecurse nounwind ssp uwtable "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind readonly ssp uwtable "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #5 = { nofree nounwind ssp uwtable "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { nounwind ssp uwtable "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { inaccessiblememonly nofree nounwind willreturn allocsize(0) "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { inaccessiblemem_or_argmemonly nounwind willreturn "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #9 = { "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #10 = { inaccessiblememonly nofree nounwind willreturn allocsize(0,1) "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #11 = { inaccessiblememonly nofree nosync nounwind willreturn }
attributes #12 = { argmemonly nofree nosync nounwind willreturn writeonly }
attributes #13 = { allocsize(0) }
attributes #14 = { nounwind }
attributes #15 = { allocsize(0,1) }

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
!12 = distinct !{!12, !8, !9}
!13 = distinct !{!13, !8, !9}
!14 = distinct !{!14, !8, !9}
!15 = distinct !{!15, !8, !9}
!16 = distinct !{!16, !8, !9}
!17 = distinct !{!17, !8, !9}
!18 = distinct !{!18, !8, !9}
!19 = distinct !{!19, !8, !9}
!20 = !{!21, !21, i64 0}
!21 = !{!"int", !5, i64 0}
!22 = distinct !{!22, !8, !9}
!23 = distinct !{!23, !8, !9}
!24 = distinct !{!24, !8, !9}
!25 = distinct !{!25, !8, !9}
!26 = distinct !{!26, !8, !9}
!27 = distinct !{!27, !8, !9}
!28 = distinct !{!28, !8, !9}
!29 = distinct !{!29, !8, !9}
!30 = !{!31}
!31 = distinct !{!31, !32, !"cgrad_log_sum_exp: %dx"}
!32 = distinct !{!32, !"cgrad_log_sum_exp"}
!33 = distinct !{!33, !8, !9}
!34 = distinct !{!34, !8, !9}
!35 = distinct !{!35, !8, !9}
!36 = distinct !{!36, !8, !9}
!37 = distinct !{!37, !8, !9}
!38 = distinct !{!38, !8, !9}
!39 = distinct !{!39, !8, !9}
!40 = distinct !{!40, !8, !9}
!41 = distinct !{!41, !8, !9}
!42 = distinct !{}
!43 = distinct !{}
!44 = distinct !{}
!45 = distinct !{}
!46 = distinct !{}
!47 = distinct !{}
!48 = distinct !{}
!49 = distinct !{}
!50 = distinct !{}
!51 = !{i64 8}
!52 = distinct !{}
!53 = distinct !{}
!54 = distinct !{}
!55 = distinct !{}
!56 = distinct !{}
!57 = distinct !{}
!58 = distinct !{}
!59 = distinct !{}
!60 = !{i64 1}
!61 = distinct !{}
!62 = distinct !{}
