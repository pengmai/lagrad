; Function Attrs: nounwind ssp uwtable
define dso_local void @enzyme_c_main_term(i32 %d, i32 %k, i32 %n, double* nocapture readonly %alphas, double* nocapture %alphasb, double* nocapture readonly %means, double* nocapture %meansb, double* nocapture readonly %Qs, double* nocapture %Qsb, double* nocapture readonly %Ls, double* nocapture %Lsb, double* nocapture readonly %x) local_unnamed_addr #5 {
entry:
  tail call void @llvm.experimental.noalias.scope.decl(metadata !64)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !67)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !69)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !71)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !73)
  %sub.i = add nsw i32 %d, -1
  %mul.i = mul nsw i32 %sub.i, %d
  %div.i = sdiv i32 %mul.i, 2
  %conv.i = sext i32 %div.i to i64
  %mul1.i = mul nsw i32 %k, %d
  %conv2.i = sext i32 %mul1.i to i64
  %mul3.i = shl nsw i64 %conv2.i, 3
  %call.i = tail call i8* @malloc(i64 %mul3.i) #15, !noalias !75
  %"call'mi.i" = tail call noalias nonnull i8* @malloc(i64 %mul3.i) #15, !noalias !75
  tail call void @llvm.memset.p0i8.i64(i8* nonnull align 1 %"call'mi.i", i8 0, i64 %mul3.i, i1 false) #14, !noalias !75
  %"'ipc21.i" = bitcast i8* %"call'mi.i" to double*
  %0 = bitcast i8* %call.i to double*
  %conv4.i = sext i32 %k to i64
  %mul5.i = shl nsw i64 %conv4.i, 3
  %call6.i = tail call i8* @malloc(i64 %mul5.i) #15, !noalias !75
  %"call6'mi.i" = tail call noalias nonnull i8* @malloc(i64 %mul5.i) #15, !noalias !75
  tail call void @llvm.memset.p0i8.i64(i8* nonnull align 1 %"call6'mi.i", i8 0, i64 %mul5.i, i1 false) #14, !noalias !75
  %"'ipc.i" = bitcast i8* %"call6'mi.i" to double*
  %1 = bitcast i8* %call6.i to double*
  %conv7.i = sext i32 %d to i64
  %mul8.i = shl nsw i64 %conv7.i, 3
  %call9.i = tail call i8* @malloc(i64 %mul8.i) #15, !noalias !75
  %"call9'mi.i" = tail call noalias nonnull i8* @malloc(i64 %mul8.i) #15, !noalias !75
  tail call void @llvm.memset.p0i8.i64(i8* nonnull align 1 %"call9'mi.i", i8 0, i64 %mul8.i, i1 false) #14, !noalias !75
  %"'ipc37.i" = bitcast i8* %"call9'mi.i" to double*
  %2 = bitcast i8* %call9.i to double*
  %call12.i = tail call i8* @malloc(i64 %mul8.i) #15, !noalias !75
  %"call12'mi.i" = tail call noalias nonnull i8* @malloc(i64 %mul8.i) #15, !noalias !75
  tail call void @llvm.memset.p0i8.i64(i8* nonnull align 1 %"call12'mi.i", i8 0, i64 %mul8.i, i1 false) #14, !noalias !75
  %"'ipc40.i" = bitcast i8* %"call12'mi.i" to double*
  %3 = bitcast i8* %call12.i to double*
  %call15.i = tail call i8* @malloc(i64 %mul5.i) #15, !noalias !75
  %"call15'mi.i" = tail call noalias nonnull i8* @malloc(i64 %mul5.i) #15, !noalias !75
  tail call void @llvm.memset.p0i8.i64(i8* nonnull align 1 %"call15'mi.i", i8 0, i64 %mul5.i, i1 false) #14, !noalias !75
  %"'ipc116.i" = bitcast i8* %"call15'mi.i" to double*
  %4 = bitcast i8* %call15.i to double*
  %cmp37.i.i = icmp sgt i32 %k, 0
  br i1 %cmp37.i.i, label %for.body.lr.ph.i.i, label %preprocess_qs.exit.i

for.body.lr.ph.i.i:                               ; preds = %entry
  %cmp235.i.i = icmp sgt i32 %d, 0
  %wide.trip.count44.i.i = zext i32 %k to i64
  %wide.trip.count.i.i = zext i32 %d to i64
  br i1 %cmp235.i.i, label %for.body.i.us.i, label %for.body.i.preheader.i

for.body.i.preheader.i:                           ; preds = %for.body.lr.ph.i.i
  %5 = shl nuw nsw i64 %wide.trip.count44.i.i, 3
  tail call void @llvm.memset.p0i8.i64(i8* align 8 %call6.i, i8 0, i64 %5, i1 false) #14, !noalias !75
  br label %preprocess_qs.exit.i

for.body.i.us.i:                                  ; preds = %for.body.lr.ph.i.i, %for.inc15.i.loopexit.us.i
  %iv.us.i = phi i64 [ %iv.next.us.i, %for.inc15.i.loopexit.us.i ], [ 0, %for.body.lr.ph.i.i ]
  %arrayidx.i.us.i = getelementptr inbounds double, double* %1, i64 %iv.us.i
  store double 0.000000e+00, double* %arrayidx.i.us.i, align 8, !tbaa !3, !noalias !75
  %6 = trunc i64 %iv.us.i to i32
  %mul.i.us.i = mul nsw i32 %6, %d
  %7 = sext i32 %mul.i.us.i to i64
  br label %for.body3.i.us.i

for.body3.i.us.i:                                 ; preds = %for.body3.i.us.i, %for.body.i.us.i
  %iv1.us.i = phi i64 [ %iv.next2.us.i, %for.body3.i.us.i ], [ 0, %for.body.i.us.i ]
  %8 = phi double [ %add8.i.us.i, %for.body3.i.us.i ], [ 0.000000e+00, %for.body.i.us.i ]
  %iv.next2.us.i = add nuw nsw i64 %iv1.us.i, 1
  %9 = add nsw i64 %iv1.us.i, %7
  %arrayidx5.i.us.i = getelementptr inbounds double, double* %Qs, i64 %9
  %10 = load double, double* %arrayidx5.i.us.i, align 8, !tbaa !3, !alias.scope !69, !noalias !76, !invariant.group !77
  %add8.i.us.i = fadd double %8, %10
  %11 = tail call double @llvm.exp.f64(double %10) #14
  %arrayidx14.i.us.i = getelementptr inbounds double, double* %0, i64 %9
  store double %11, double* %arrayidx14.i.us.i, align 8, !tbaa !3, !noalias !75
  %exitcond.not.i.us.i = icmp eq i64 %iv.next2.us.i, %wide.trip.count.i.i
  br i1 %exitcond.not.i.us.i, label %for.inc15.i.loopexit.us.i, label %for.body3.i.us.i, !llvm.loop !33

for.inc15.i.loopexit.us.i:                        ; preds = %for.body3.i.us.i
  %iv.next.us.i = add nuw nsw i64 %iv.us.i, 1
  store double %add8.i.us.i, double* %arrayidx.i.us.i, align 8, !tbaa !3, !noalias !75
  %exitcond45.not.i.us.i = icmp eq i64 %iv.next.us.i, %wide.trip.count44.i.i
  br i1 %exitcond45.not.i.us.i, label %preprocess_qs.exit.i, label %for.body.i.us.i, !llvm.loop !34

preprocess_qs.exit.i:                             ; preds = %for.inc15.i.loopexit.us.i, %for.body.i.preheader.i, %entry
  %conv17.i = sext i32 %n to i64
  %cmp140.i = icmp sgt i32 %n, 0
  br i1 %cmp140.i, label %for.cond19.preheader.lr.ph.i, label %for.end50.i.thread

for.end50.i.thread:                               ; preds = %preprocess_qs.exit.i
  tail call void @free(i8* %call6.i) #14, !noalias !75
  br label %invertpreprocess_qs.exit.i

for.cond19.preheader.lr.ph.i:                     ; preds = %preprocess_qs.exit.i
  %cmp10.i.i = icmp sgt i32 %d, 0
  %wide.trip.count.i119.i = zext i32 %d to i64
  %mul8.i.i = shl nuw nsw i32 %d, 1
  %cmp15.i.i = icmp sgt i32 %d, 1
  %cmp13.i.i.i = icmp sgt i32 %k, 1
  %wide.trip.count.i.i.i = zext i32 %k to i64
  %exitcond.not.i99137.i = icmp eq i32 %k, 1
  %12 = mul nuw nsw i64 %conv17.i, %conv4.i
  %13 = shl nuw nsw i64 %wide.trip.count.i119.i, 3
  %mallocsize.i = mul i64 %12, %13
  %malloccall.i = tail call noalias nonnull i8* @malloc(i64 %mallocsize.i) #14, !noalias !75
  %_malloccache.i = bitcast i8* %malloccall.i to double*
  %mallocsize83.i = shl nuw nsw i64 %12, 3
  %malloccall84.i = tail call noalias nonnull i8* @malloc(i64 %mallocsize83.i) #14, !noalias !75
  %_malloccache85.i = bitcast i8* %malloccall84.i to double*
  %scevgep.i = getelementptr i8, i8* %call12.i, i64 8
  %_unwrap97.i = add nsw i64 %wide.trip.count.i119.i, -1
  %14 = shl nsw i64 %_unwrap97.i, 3
  %mallocsize99.i = mul i64 %12, %14
  %malloccall100.i = tail call noalias nonnull i8* @malloc(i64 %mallocsize99.i) #14, !noalias !75
  %_malloccache101.i = bitcast i8* %malloccall100.i to double*
  %malloccall120.i = tail call noalias nonnull i8* @malloc(i64 %conv17.i) #14, !noalias !75
  %mallocsize127.i = shl nsw i64 %conv17.i, 3
  %malloccall128.i = tail call noalias nonnull i8* @malloc(i64 %mallocsize127.i) #14, !noalias !75
  %"!manual_lcssa126_malloccache.i" = bitcast i8* %malloccall128.i to i64*
  %malloccall135.i = tail call noalias nonnull i8* @malloc(i64 %mallocsize127.i) #14, !noalias !75
  %sub.i135_malloccache.i = bitcast i8* %malloccall135.i to double*
  %15 = add nsw i64 %wide.trip.count.i.i.i, -1
  %mallocsize140.i = mul i64 %mallocsize127.i, %15
  %malloccall141.i = tail call noalias nonnull i8* @malloc(i64 %mallocsize140.i) #14, !noalias !75
  %sub.i_malloccache.i = bitcast i8* %malloccall141.i to double*
  %malloccall156.i = tail call noalias nonnull i8* @malloc(i64 %mallocsize127.i) #14, !noalias !75
  %"add.i!manual_lcssa154_malloccache.i" = bitcast i8* %malloccall156.i to double*
  %min.iters.check50 = icmp ult i32 %d, 4
  %n.vec53 = and i64 %wide.trip.count.i119.i, 4294967292
  %cmp.n57 = icmp eq i64 %n.vec53, %wide.trip.count.i119.i
  %min.iters.check35 = icmp ult i32 %d, 4
  %n.vec38 = and i64 %wide.trip.count.i119.i, 4294967292
  %cmp.n42 = icmp eq i64 %n.vec38, %wide.trip.count.i119.i
  br label %for.cond19.preheader.i

for.cond19.preheader.i:                           ; preds = %log_sum_exp.exit.i, %for.cond19.preheader.lr.ph.i
  %iv3.i = phi i64 [ %iv.next4.i, %log_sum_exp.exit.i ], [ 0, %for.cond19.preheader.lr.ph.i ]
  %16 = phi double [ %98, %log_sum_exp.exit.i ], [ undef, %for.cond19.preheader.lr.ph.i ]
  %17 = phi double [ %99, %log_sum_exp.exit.i ], [ undef, %for.cond19.preheader.lr.ph.i ]
  %iv.next4.i = add nuw nsw i64 %iv3.i, 1
  br i1 %cmp37.i.i, label %for.body23.lr.ph.i, label %for.end.i

for.body23.lr.ph.i:                               ; preds = %for.cond19.preheader.i
  %mul25.i = mul nsw i64 %iv3.i, %conv7.i
  %arrayidx26.i = getelementptr inbounds double, double* %x, i64 %mul25.i
  %18 = mul nuw nsw i64 %iv3.i, %conv4.i
  br i1 %cmp10.i.i, label %for.body23.i.us, label %for.body23.i.preheader

for.body23.i.preheader:                           ; preds = %for.body23.lr.ph.i
  %mul.i102.i = fmul double %17, %17
  br label %for.body23.i

for.body23.i.us:                                  ; preds = %for.body23.lr.ph.i, %sqnorm.exit.i.us
  %iv5.i.us = phi i64 [ %iv.next6.i.us, %sqnorm.exit.i.us ], [ 0, %for.body23.lr.ph.i ]
  %mul28.i.us = mul nsw i64 %iv5.i.us, %conv7.i
  %arrayidx29.i.us = getelementptr inbounds double, double* %means, i64 %mul28.i.us
  br i1 %min.iters.check50, label %for.body.i128.i.us.preheader, label %vector.body49

vector.body49:                                    ; preds = %for.body23.i.us, %vector.body49
  %index54 = phi i64 [ %index.next55, %vector.body49 ], [ 0, %for.body23.i.us ]
  %19 = getelementptr inbounds double, double* %arrayidx26.i, i64 %index54
  %20 = bitcast double* %19 to <2 x double>*
  %wide.load58 = load <2 x double>, <2 x double>* %20, align 8, !tbaa !3, !alias.scope !73, !noalias !78
  %21 = getelementptr inbounds double, double* %19, i64 2
  %22 = bitcast double* %21 to <2 x double>*
  %wide.load59 = load <2 x double>, <2 x double>* %22, align 8, !tbaa !3, !alias.scope !73, !noalias !78
  %23 = getelementptr inbounds double, double* %arrayidx29.i.us, i64 %index54
  %24 = bitcast double* %23 to <2 x double>*
  %wide.load60 = load <2 x double>, <2 x double>* %24, align 8, !tbaa !3, !alias.scope !67, !noalias !79
  %25 = getelementptr inbounds double, double* %23, i64 2
  %26 = bitcast double* %25 to <2 x double>*
  %wide.load61 = load <2 x double>, <2 x double>* %26, align 8, !tbaa !3, !alias.scope !67, !noalias !79
  %27 = fsub <2 x double> %wide.load58, %wide.load60
  %28 = fsub <2 x double> %wide.load59, %wide.load61
  %29 = getelementptr inbounds double, double* %2, i64 %index54
  %30 = bitcast double* %29 to <2 x double>*
  store <2 x double> %27, <2 x double>* %30, align 8, !tbaa !3, !noalias !75
  %31 = getelementptr inbounds double, double* %29, i64 2
  %32 = bitcast double* %31 to <2 x double>*
  store <2 x double> %28, <2 x double>* %32, align 8, !tbaa !3, !noalias !75
  %index.next55 = add i64 %index54, 4
  %33 = icmp eq i64 %index.next55, %n.vec53
  br i1 %33, label %middle.block47, label %vector.body49, !llvm.loop !80

middle.block47:                                   ; preds = %vector.body49
  br i1 %cmp.n57, label %for.body.i114.preheader.i.us, label %for.body.i128.i.us.preheader

for.body.i128.i.us.preheader:                     ; preds = %for.body23.i.us, %middle.block47
  %iv7.i.us.ph = phi i64 [ 0, %for.body23.i.us ], [ %n.vec53, %middle.block47 ]
  br label %for.body.i128.i.us

for.body.i128.i.us:                               ; preds = %for.body.i128.i.us.preheader, %for.body.i128.i.us
  %iv7.i.us = phi i64 [ %iv.next8.i.us, %for.body.i128.i.us ], [ %iv7.i.us.ph, %for.body.i128.i.us.preheader ]
  %iv.next8.i.us = add nuw nsw i64 %iv7.i.us, 1
  %arrayidx.i122.i.us = getelementptr inbounds double, double* %arrayidx26.i, i64 %iv7.i.us
  %34 = load double, double* %arrayidx.i122.i.us, align 8, !tbaa !3, !alias.scope !73, !noalias !78
  %arrayidx2.i123.i.us = getelementptr inbounds double, double* %arrayidx29.i.us, i64 %iv7.i.us
  %35 = load double, double* %arrayidx2.i123.i.us, align 8, !tbaa !3, !alias.scope !67, !noalias !79
  %sub.i124.i.us = fsub double %34, %35
  %arrayidx4.i125.i.us = getelementptr inbounds double, double* %2, i64 %iv7.i.us
  store double %sub.i124.i.us, double* %arrayidx4.i125.i.us, align 8, !tbaa !3, !noalias !75
  %exitcond.not.i127.i.us = icmp eq i64 %iv.next8.i.us, %wide.trip.count.i119.i
  br i1 %exitcond.not.i127.i.us, label %for.body.i114.preheader.i.us, label %for.body.i128.i.us, !llvm.loop !81

for.body.i114.preheader.i.us:                     ; preds = %for.body.i128.i.us, %middle.block47
  %arrayidx33.i.us = getelementptr inbounds double, double* %0, i64 %mul28.i.us
  %36 = add nuw nsw i64 %iv5.i.us, %18
  %37 = mul nuw nsw i64 %36, %wide.trip.count.i119.i
  %38 = getelementptr inbounds double, double* %_malloccache.i, i64 %37
  %39 = bitcast double* %38 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %39, i8* nonnull align 8 %call9.i, i64 %13, i1 false) #14, !noalias !75
  br i1 %min.iters.check35, label %for.body.i114.i.us.preheader, label %vector.body34

vector.body34:                                    ; preds = %for.body.i114.preheader.i.us, %vector.body34
  %index39 = phi i64 [ %index.next40, %vector.body34 ], [ 0, %for.body.i114.preheader.i.us ]
  %40 = getelementptr inbounds double, double* %arrayidx33.i.us, i64 %index39
  %41 = bitcast double* %40 to <2 x double>*
  %wide.load43 = load <2 x double>, <2 x double>* %41, align 8, !tbaa !3, !noalias !75
  %42 = getelementptr inbounds double, double* %40, i64 2
  %43 = bitcast double* %42 to <2 x double>*
  %wide.load44 = load <2 x double>, <2 x double>* %43, align 8, !tbaa !3, !noalias !75
  %44 = getelementptr inbounds double, double* %2, i64 %index39
  %45 = bitcast double* %44 to <2 x double>*
  %wide.load45 = load <2 x double>, <2 x double>* %45, align 8, !tbaa !3, !noalias !75
  %46 = getelementptr inbounds double, double* %44, i64 2
  %47 = bitcast double* %46 to <2 x double>*
  %wide.load46 = load <2 x double>, <2 x double>* %47, align 8, !tbaa !3, !noalias !75
  %48 = fmul <2 x double> %wide.load43, %wide.load45
  %49 = fmul <2 x double> %wide.load44, %wide.load46
  %50 = getelementptr inbounds double, double* %3, i64 %index39
  %51 = bitcast double* %50 to <2 x double>*
  store <2 x double> %48, <2 x double>* %51, align 8, !tbaa !3, !noalias !75
  %52 = getelementptr inbounds double, double* %50, i64 2
  %53 = bitcast double* %52 to <2 x double>*
  store <2 x double> %49, <2 x double>* %53, align 8, !tbaa !3, !noalias !75
  %index.next40 = add i64 %index39, 4
  %54 = icmp eq i64 %index.next40, %n.vec38
  br i1 %54, label %middle.block32, label %vector.body34, !llvm.loop !82

middle.block32:                                   ; preds = %vector.body34
  br i1 %cmp.n42, label %for.body7.i.preheader.i.us, label %for.body.i114.i.us.preheader

for.body.i114.i.us.preheader:                     ; preds = %for.body.i114.preheader.i.us, %middle.block32
  %iv9.i.us.ph = phi i64 [ 0, %for.body.i114.preheader.i.us ], [ %n.vec38, %middle.block32 ]
  br label %for.body.i114.i.us

for.body.i114.i.us:                               ; preds = %for.body.i114.i.us.preheader, %for.body.i114.i.us
  %iv9.i.us = phi i64 [ %iv.next10.i.us, %for.body.i114.i.us ], [ %iv9.i.us.ph, %for.body.i114.i.us.preheader ]
  %iv.next10.i.us = add nuw nsw i64 %iv9.i.us, 1
  %arrayidx.i111.i.us = getelementptr inbounds double, double* %arrayidx33.i.us, i64 %iv9.i.us
  %55 = load double, double* %arrayidx.i111.i.us, align 8, !tbaa !3, !noalias !75, !invariant.group !83
  %arrayidx2.i112.i.us = getelementptr inbounds double, double* %2, i64 %iv9.i.us
  %56 = load double, double* %arrayidx2.i112.i.us, align 8, !tbaa !3, !noalias !75
  %mul.i113.i.us = fmul double %55, %56
  %arrayidx4.i.i.us = getelementptr inbounds double, double* %3, i64 %iv9.i.us
  store double %mul.i113.i.us, double* %arrayidx4.i.i.us, align 8, !tbaa !3, !noalias !75
  %exitcond72.not.i.i.us = icmp eq i64 %iv.next10.i.us, %wide.trip.count.i119.i
  br i1 %exitcond72.not.i.i.us, label %for.body7.i.preheader.i.us, label %for.body.i114.i.us, !llvm.loop !84

for.body7.i.preheader.i.us:                       ; preds = %for.body.i114.i.us, %middle.block32
  %mul34.i.us = mul nsw i64 %iv5.i.us, %conv.i
  %arrayidx35.i.us = getelementptr inbounds double, double* %Ls, i64 %mul34.i.us
  br label %for.body7.i.i.us

for.body7.i.i.us:                                 ; preds = %for.cond5.loopexit.i.i.us, %for.body7.i.preheader.i.us
  %indvars.iv.i.us = phi i64 [ %_unwrap97.i, %for.body7.i.preheader.i.us ], [ %indvars.iv.next.i.us, %for.cond5.loopexit.i.i.us ]
  %iv11.i.us = phi i64 [ 0, %for.body7.i.preheader.i.us ], [ %58, %for.cond5.loopexit.i.i.us ]
  %57 = sub i64 %_unwrap97.i, %iv11.i.us
  %58 = add nuw nsw i64 %iv11.i.us, 1
  %cmp1254.i.i.us = icmp ult i64 %58, %wide.trip.count.i119.i
  br i1 %cmp1254.i.i.us, label %for.body13.lr.ph.i.i.us, label %for.cond5.loopexit.i.i.us

for.body13.lr.ph.i.i.us:                          ; preds = %for.body7.i.i.us
  %59 = trunc i64 %iv11.i.us to i32
  %60 = xor i32 %59, -1
  %sub9.i.i.us = add i32 %mul8.i.i, %60
  %mul10.i.i.us = mul nsw i32 %sub9.i.i.us, %59
  %div.i.i.us = sdiv i32 %mul10.i.i.us, 2
  %arrayidx19.i.i.us = getelementptr inbounds double, double* %2, i64 %iv11.i.us
  %61 = sext i32 %div.i.i.us to i64
  %62 = load double, double* %arrayidx19.i.i.us, align 8, !tbaa !3, !noalias !75
  %min.iters.check = icmp ult i64 %57, 4
  br i1 %min.iters.check, label %for.body13.i.i.us.preheader, label %vector.ph

vector.ph:                                        ; preds = %for.body13.lr.ph.i.i.us
  %n.vec = and i64 %57, -4
  %broadcast.splatinsert = insertelement <2 x double> poison, double %62, i32 0
  %broadcast.splat = shufflevector <2 x double> %broadcast.splatinsert, <2 x double> poison, <2 x i32> zeroinitializer
  %broadcast.splatinsert30 = insertelement <2 x double> poison, double %62, i32 0
  %broadcast.splat31 = shufflevector <2 x double> %broadcast.splatinsert30, <2 x double> poison, <2 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %63 = add i64 %index, %61
  %64 = add i64 %index, %58
  %65 = getelementptr inbounds double, double* %3, i64 %64
  %66 = bitcast double* %65 to <2 x double>*
  %wide.load = load <2 x double>, <2 x double>* %66, align 8, !tbaa !3, !noalias !75
  %67 = getelementptr inbounds double, double* %65, i64 2
  %68 = bitcast double* %67 to <2 x double>*
  %wide.load27 = load <2 x double>, <2 x double>* %68, align 8, !tbaa !3, !noalias !75
  %69 = getelementptr inbounds double, double* %arrayidx35.i.us, i64 %63
  %70 = bitcast double* %69 to <2 x double>*
  %wide.load28 = load <2 x double>, <2 x double>* %70, align 8, !tbaa !3, !alias.scope !71, !noalias !85
  %71 = getelementptr inbounds double, double* %69, i64 2
  %72 = bitcast double* %71 to <2 x double>*
  %wide.load29 = load <2 x double>, <2 x double>* %72, align 8, !tbaa !3, !alias.scope !71, !noalias !85
  %73 = fmul <2 x double> %broadcast.splat, %wide.load28
  %74 = fmul <2 x double> %broadcast.splat31, %wide.load29
  %75 = fadd <2 x double> %wide.load, %73
  %76 = fadd <2 x double> %wide.load27, %74
  %77 = bitcast double* %65 to <2 x double>*
  store <2 x double> %75, <2 x double>* %77, align 8, !tbaa !3, !noalias !75
  %78 = bitcast double* %67 to <2 x double>*
  store <2 x double> %76, <2 x double>* %78, align 8, !tbaa !3, !noalias !75
  %index.next = add i64 %index, 4
  %79 = icmp eq i64 %index.next, %n.vec
  br i1 %79, label %middle.block, label %vector.body, !llvm.loop !86

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %57, %n.vec
  br i1 %cmp.n, label %for.cond5.loopexit.i.i.us, label %for.body13.i.i.us.preheader

for.body13.i.i.us.preheader:                      ; preds = %for.body13.lr.ph.i.i.us, %middle.block
  %iv13.i.us.ph = phi i64 [ 0, %for.body13.lr.ph.i.i.us ], [ %n.vec, %middle.block ]
  br label %for.body13.i.i.us

for.body13.i.i.us:                                ; preds = %for.body13.i.i.us.preheader, %for.body13.i.i.us
  %iv13.i.us = phi i64 [ %iv.next14.i.us, %for.body13.i.i.us ], [ %iv13.i.us.ph, %for.body13.i.i.us.preheader ]
  %80 = add i64 %iv13.i.us, %61
  %iv.next14.i.us = add nuw i64 %iv13.i.us, 1
  %81 = add i64 %iv13.i.us, %58
  %arrayidx15.i.i.us = getelementptr inbounds double, double* %3, i64 %81
  %82 = load double, double* %arrayidx15.i.i.us, align 8, !tbaa !3, !noalias !75
  %arrayidx17.i.i.us = getelementptr inbounds double, double* %arrayidx35.i.us, i64 %80
  %83 = load double, double* %arrayidx17.i.i.us, align 8, !tbaa !3, !alias.scope !71, !noalias !85, !invariant.group !87
  %mul20.i.i.us = fmul double %62, %83
  %add21.i.i.us = fadd double %82, %mul20.i.i.us
  store double %add21.i.i.us, double* %arrayidx15.i.i.us, align 8, !tbaa !3, !noalias !75
  %exitcond.i.us = icmp eq i64 %iv.next14.i.us, %indvars.iv.i.us
  br i1 %exitcond.i.us, label %for.cond5.loopexit.i.i.us, label %for.body13.i.i.us, !llvm.loop !88

for.cond5.loopexit.i.i.us:                        ; preds = %for.body13.i.i.us, %middle.block, %for.body7.i.i.us
  %exitcond68.not.i.i.us = icmp eq i64 %58, %wide.trip.count.i119.i
  %indvars.iv.next.i.us = add nsw i64 %indvars.iv.i.us, -1
  br i1 %exitcond68.not.i.i.us, label %cQtimesx.exit.loopexit.i.us, label %for.body7.i.i.us, !llvm.loop !45

cQtimesx.exit.loopexit.i.us:                      ; preds = %for.cond5.loopexit.i.i.us
  %iv.next6.i.us = add nuw nsw i64 %iv5.i.us, 1
  %.pre.i.us = load double, double* %3, align 8, !tbaa !3, !noalias !75
  %84 = getelementptr inbounds double, double* %_malloccache85.i, i64 %36
  store double %.pre.i.us, double* %84, align 8, !noalias !75, !invariant.group !89
  %arrayidx38.i.us = getelementptr inbounds double, double* %alphas, i64 %iv5.i.us
  %85 = load double, double* %arrayidx38.i.us, align 8, !tbaa !3, !alias.scope !64, !noalias !90
  %arrayidx39.i.us = getelementptr inbounds double, double* %1, i64 %iv5.i.us
  %86 = load double, double* %arrayidx39.i.us, align 8, !tbaa !3, !noalias !75
  %add.i.us = fadd double %85, %86
  %mul.i102.i.us = fmul double %.pre.i.us, %.pre.i.us
  br i1 %cmp15.i.i, label %for.body.i109.preheader.i.us, label %sqnorm.exit.i.us

for.body.i109.preheader.i.us:                     ; preds = %cQtimesx.exit.loopexit.i.us
  %87 = mul nuw nsw i64 %36, %_unwrap97.i
  %88 = getelementptr inbounds double, double* %_malloccache101.i, i64 %87
  %89 = bitcast double* %88 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %89, i8* nonnull align 8 %scevgep.i, i64 %14, i1 false) #14, !noalias !75
  br label %for.body.i109.i.us

for.body.i109.i.us:                               ; preds = %for.body.i109.i.us, %for.body.i109.preheader.i.us
  %iv15.i.us = phi i64 [ %iv.next16.i.us, %for.body.i109.i.us ], [ 0, %for.body.i109.preheader.i.us ]
  %res.017.i.i.us = phi double [ %add.i106.i.us, %for.body.i109.i.us ], [ %mul.i102.i.us, %for.body.i109.preheader.i.us ]
  %iv.next16.i.us = add nuw nsw i64 %iv15.i.us, 1
  %arrayidx2.i.i.us = getelementptr inbounds double, double* %3, i64 %iv.next16.i.us
  %90 = load double, double* %arrayidx2.i.i.us, align 8, !tbaa !3, !noalias !75
  %mul5.i.i.us = fmul double %90, %90
  %add.i106.i.us = fadd double %res.017.i.i.us, %mul5.i.i.us
  %indvars.iv.next.i107.i.us = add nuw nsw i64 %iv15.i.us, 2
  %exitcond.not.i108.i.us = icmp eq i64 %indvars.iv.next.i107.i.us, %wide.trip.count.i119.i
  br i1 %exitcond.not.i108.i.us, label %sqnorm.exit.i.us, label %for.body.i109.i.us, !llvm.loop !10

sqnorm.exit.i.us:                                 ; preds = %for.body.i109.i.us, %cQtimesx.exit.loopexit.i.us
  %res.0.lcssa.i.i.us = phi double [ %mul.i102.i.us, %cQtimesx.exit.loopexit.i.us ], [ %add.i106.i.us, %for.body.i109.i.us ]
  %mul42.i.us = fmul double %res.0.lcssa.i.i.us, 5.000000e-01
  %sub43.i.us = fsub double %add.i.us, %mul42.i.us
  %arrayidx44.i.us = getelementptr inbounds double, double* %4, i64 %iv5.i.us
  store double %sub43.i.us, double* %arrayidx44.i.us, align 8, !tbaa !3, !noalias !75
  %exitcond.not.i.us = icmp eq i64 %iv.next6.i.us, %conv4.i
  br i1 %exitcond.not.i.us, label %for.end.loopexit.i, label %for.body23.i.us, !llvm.loop !62

for.body23.i:                                     ; preds = %for.body23.i.preheader, %sqnorm.exit.i
  %iv5.i = phi i64 [ %iv.next6.i, %sqnorm.exit.i ], [ 0, %for.body23.i.preheader ]
  %iv.next6.i = add nuw nsw i64 %iv5.i, 1
  %.pre179.i = add nuw nsw i64 %iv5.i, %18
  %91 = getelementptr inbounds double, double* %_malloccache85.i, i64 %.pre179.i
  store double %17, double* %91, align 8, !noalias !75, !invariant.group !89
  %arrayidx38.i = getelementptr inbounds double, double* %alphas, i64 %iv5.i
  %92 = load double, double* %arrayidx38.i, align 8, !tbaa !3, !alias.scope !64, !noalias !90
  %arrayidx39.i = getelementptr inbounds double, double* %1, i64 %iv5.i
  %93 = load double, double* %arrayidx39.i, align 8, !tbaa !3, !noalias !75
  %add.i = fadd double %92, %93
  br i1 %cmp15.i.i, label %for.body.i109.preheader.i, label %sqnorm.exit.i

for.body.i109.preheader.i:                        ; preds = %for.body23.i
  %94 = mul nuw nsw i64 %.pre179.i, %_unwrap97.i
  %95 = getelementptr inbounds double, double* %_malloccache101.i, i64 %94
  %96 = bitcast double* %95 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %96, i8* nonnull align 8 %scevgep.i, i64 %14, i1 false) #14, !noalias !75
  br label %for.body.i109.i

for.body.i109.i:                                  ; preds = %for.body.i109.i, %for.body.i109.preheader.i
  %iv15.i = phi i64 [ %iv.next16.i, %for.body.i109.i ], [ 0, %for.body.i109.preheader.i ]
  %res.017.i.i = phi double [ %add.i106.i, %for.body.i109.i ], [ %mul.i102.i, %for.body.i109.preheader.i ]
  %iv.next16.i = add nuw nsw i64 %iv15.i, 1
  %arrayidx2.i.i = getelementptr inbounds double, double* %3, i64 %iv.next16.i
  %97 = load double, double* %arrayidx2.i.i, align 8, !tbaa !3, !noalias !75
  %mul5.i.i = fmul double %97, %97
  %add.i106.i = fadd double %res.017.i.i, %mul5.i.i
  %indvars.iv.next.i107.i = add nuw nsw i64 %iv15.i, 2
  %exitcond.not.i108.i = icmp eq i64 %indvars.iv.next.i107.i, %wide.trip.count.i119.i
  br i1 %exitcond.not.i108.i, label %sqnorm.exit.i, label %for.body.i109.i, !llvm.loop !10

sqnorm.exit.i:                                    ; preds = %for.body.i109.i, %for.body23.i
  %res.0.lcssa.i.i = phi double [ %mul.i102.i, %for.body23.i ], [ %add.i106.i, %for.body.i109.i ]
  %mul42.i = fmul double %res.0.lcssa.i.i, 5.000000e-01
  %sub43.i = fsub double %add.i, %mul42.i
  %arrayidx44.i = getelementptr inbounds double, double* %4, i64 %iv5.i
  store double %sub43.i, double* %arrayidx44.i, align 8, !tbaa !3, !noalias !75
  %exitcond.not.i = icmp eq i64 %iv.next6.i, %conv4.i
  br i1 %exitcond.not.i, label %for.end.loopexit.i, label %for.body23.i, !llvm.loop !62

for.end.loopexit.i:                               ; preds = %sqnorm.exit.i, %sqnorm.exit.i.us
  %.lcssa13 = phi double [ %.pre.i.us, %sqnorm.exit.i.us ], [ %17, %sqnorm.exit.i ]
  %.pre146.i = load double, double* %4, align 8, !tbaa !3, !noalias !75
  br label %for.end.i

for.end.i:                                        ; preds = %for.end.loopexit.i, %for.cond19.preheader.i
  %98 = phi double [ %.pre146.i, %for.end.loopexit.i ], [ %16, %for.cond19.preheader.i ]
  %99 = phi double [ %.lcssa13, %for.end.loopexit.i ], [ %17, %for.cond19.preheader.i ]
  br i1 %cmp13.i.i.i, label %for.body.i.i.i, label %arr_max.exit.i.i

for.body.i.i.i:                                   ; preds = %for.end.i, %for.body.i.i.i
  %100 = phi i64 [ %102, %for.body.i.i.i ], [ 0, %for.end.i ]
  %iv17.i = phi i64 [ %iv.next18.i, %for.body.i.i.i ], [ 0, %for.end.i ]
  %m.015.i.i.i = phi double [ %m.1.i.i.i, %for.body.i.i.i ], [ %98, %for.end.i ]
  %iv.next18.i = add nuw nsw i64 %iv17.i, 1
  %arrayidx1.i.i.i = getelementptr inbounds double, double* %4, i64 %iv.next18.i
  %101 = load double, double* %arrayidx1.i.i.i, align 8, !tbaa !3, !noalias !75
  %cmp2.i.i.i = fcmp olt double %m.015.i.i.i, %101
  %102 = select i1 %cmp2.i.i.i, i64 %iv.next18.i, i64 %100
  %m.1.i.i.i = select i1 %cmp2.i.i.i, double %101, double %m.015.i.i.i
  %indvars.iv.next.i.i.i = add nuw nsw i64 %iv17.i, 2
  %exitcond.not.i.i.i = icmp eq i64 %indvars.iv.next.i.i.i, %wide.trip.count.i.i.i
  br i1 %exitcond.not.i.i.i, label %arr_max.exit.i.loopexit.i, label %for.body.i.i.i, !llvm.loop !7

arr_max.exit.i.loopexit.i:                        ; preds = %for.body.i.i.i
  %103 = getelementptr inbounds i8, i8* %malloccall120.i, i64 %iv3.i
  %104 = bitcast i8* %103 to i1*
  store i1 %cmp2.i.i.i, i1* %104, align 1, !noalias !75, !invariant.group !91
  %105 = getelementptr inbounds i64, i64* %"!manual_lcssa126_malloccache.i", i64 %iv3.i
  store i64 %100, i64* %105, align 8, !noalias !75, !invariant.group !92
  br label %arr_max.exit.i.i

arr_max.exit.i.i:                                 ; preds = %arr_max.exit.i.loopexit.i, %for.end.i
  %m.0.lcssa.i.i.i = phi double [ %98, %for.end.i ], [ %m.1.i.i.i, %arr_max.exit.i.loopexit.i ]
  br i1 %cmp37.i.i, label %for.body.preheader.i.i, label %log_sum_exp.exit.i

for.body.preheader.i.i:                           ; preds = %arr_max.exit.i.i
  %sub.i135.i = fsub double %98, %m.0.lcssa.i.i.i
  %106 = getelementptr inbounds double, double* %sub.i135_malloccache.i, i64 %iv3.i
  store double %sub.i135.i, double* %106, align 8, !noalias !75, !invariant.group !93
  br i1 %exitcond.not.i99137.i, label %log_sum_exp.exit.i, label %for.body.for.body_crit_edge.i.preheader.i, !llvm.loop !22

for.body.for.body_crit_edge.i.preheader.i:        ; preds = %for.body.preheader.i.i
  %107 = tail call double @llvm.exp.f64(double %sub.i135.i) #14
  %add.i136.i = fadd double %107, 0.000000e+00
  %108 = mul nuw nsw i64 %iv3.i, %15
  br label %for.body.for.body_crit_edge.i.i

for.body.for.body_crit_edge.i.i:                  ; preds = %for.body.for.body_crit_edge.i.i, %for.body.for.body_crit_edge.i.preheader.i
  %iv19.i = phi i64 [ %iv.next20.i, %for.body.for.body_crit_edge.i.i ], [ 0, %for.body.for.body_crit_edge.i.preheader.i ]
  %add.i138.i = phi double [ %add.i.i, %for.body.for.body_crit_edge.i.i ], [ %add.i136.i, %for.body.for.body_crit_edge.i.preheader.i ]
  %iv.next20.i = add nuw nsw i64 %iv19.i, 1
  %arrayidx.phi.trans.insert.i.i = getelementptr inbounds double, double* %4, i64 %iv.next20.i
  %.pre.i101.i = load double, double* %arrayidx.phi.trans.insert.i.i, align 8, !tbaa !3, !noalias !75
  %sub.i.i = fsub double %.pre.i101.i, %m.0.lcssa.i.i.i
  %109 = add nuw nsw i64 %iv19.i, %108
  %110 = getelementptr inbounds double, double* %sub.i_malloccache.i, i64 %109
  store double %sub.i.i, double* %110, align 8, !noalias !75, !invariant.group !94
  %111 = tail call double @llvm.exp.f64(double %sub.i.i) #14
  %add.i.i = fadd double %add.i138.i, %111
  %indvars.iv.next.i98.i = add nuw nsw i64 %iv19.i, 2
  %exitcond.not.i99.i = icmp eq i64 %indvars.iv.next.i98.i, %wide.trip.count.i.i.i
  br i1 %exitcond.not.i99.i, label %log_sum_exp.exit.i, label %for.body.for.body_crit_edge.i.i, !llvm.loop !22

log_sum_exp.exit.i:                               ; preds = %for.body.for.body_crit_edge.i.i, %for.body.preheader.i.i, %arr_max.exit.i.i
  %"add.i!manual_lcssa154.i" = phi double [ undef, %for.body.preheader.i.i ], [ undef, %arr_max.exit.i.i ], [ %add.i.i, %for.body.for.body_crit_edge.i.i ]
  %112 = getelementptr inbounds double, double* %"add.i!manual_lcssa154_malloccache.i", i64 %iv3.i
  store double %"add.i!manual_lcssa154.i", double* %112, align 8, !noalias !75, !invariant.group !95
  %exitcond145.not.i = icmp eq i64 %iv.next4.i, %conv17.i
  br i1 %exitcond145.not.i, label %invertlog_sum_exp.exit.preheader.i, label %for.cond19.preheader.i, !llvm.loop !63

invertlog_sum_exp.exit.preheader.i:               ; preds = %log_sum_exp.exit.i
  tail call void @free(i8* %call6.i) #14, !noalias !75
  %_unwrap149.i = add nsw i64 %wide.trip.count.i.i.i, -2
  %_unwrap115.i = add nsw i64 %wide.trip.count.i119.i, -2
  %113 = and i64 %wide.trip.count.i119.i, 4294967294
  %114 = add nsw i64 %113, -2
  %115 = lshr exact i64 %114, 1
  %116 = add nuw i64 %115, 1
  %min.iters.check164 = icmp ult i64 %15, 4
  %n.vec167 = and i64 %15, -4
  %ind.end171 = sub nsw i64 %_unwrap149.i, %n.vec167
  %117 = getelementptr inbounds i8, i8* %"call15'mi.i", i64 -24
  %118 = bitcast i8* %117 to double*
  %cmp.n172 = icmp eq i64 %15, %n.vec167
  %xtraiter = and i64 %15, 1
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0
  %iv.next18_unwrap.i.prol = add nsw i64 %wide.trip.count.i.i.i, -1
  %"arrayidx1.i.i'ipg_unwrap.i.prol" = getelementptr inbounds double, double* %"'ipc116.i", i64 %iv.next18_unwrap.i.prol
  %119 = add nsw i64 %wide.trip.count.i.i.i, -3
  %120 = icmp eq i64 %_unwrap149.i, 0
  %min.iters.check137 = icmp ult i64 %_unwrap97.i, 4
  %n.vec140 = and i64 %_unwrap97.i, -4
  %ind.end144 = sub nsw i64 %_unwrap115.i, %n.vec140
  %121 = getelementptr inbounds i8, i8* %"call12'mi.i", i64 -24
  %122 = bitcast i8* %121 to double*
  %cmp.n145 = icmp eq i64 %_unwrap97.i, %n.vec140
  %min.iters.check80 = icmp ult i32 %d, 2
  %n.vec83 = and i64 %wide.trip.count.i119.i, 4294967294
  %ind.end87 = and i64 %wide.trip.count.i119.i, 1
  %123 = getelementptr inbounds i8, i8* %"call12'mi.i", i64 -8
  %124 = bitcast i8* %123 to double*
  %125 = getelementptr inbounds i8, i8* %"call9'mi.i", i64 -8
  %126 = bitcast i8* %125 to double*
  %cmp.n88 = icmp eq i64 %n.vec83, %wide.trip.count.i119.i
  %min.iters.check65 = icmp ult i32 %d, 2
  %n.vec68 = and i64 %wide.trip.count.i119.i, 4294967294
  %ind.end = and i64 %wide.trip.count.i119.i, 1
  %xtraiter223 = and i64 %116, 1
  %127 = icmp eq i64 %114, 0
  %unroll_iter = and i64 %116, -2
  %128 = getelementptr inbounds i8, i8* %"call9'mi.i", i64 -8
  %129 = bitcast i8* %128 to double*
  %130 = getelementptr inbounds i8, i8* %"call9'mi.i", i64 -8
  %131 = bitcast i8* %130 to double*
  %lcmp.mod224.not = icmp eq i64 %xtraiter223, 0
  %132 = getelementptr inbounds i8, i8* %"call9'mi.i", i64 -8
  %133 = bitcast i8* %132 to double*
  %cmp.n72 = icmp eq i64 %n.vec68, %wide.trip.count.i119.i
  br label %invertlog_sum_exp.exit.i

invertfor.inc15.i.us.i.preheader:                 ; preds = %invertpreprocess_qs.exit.i
  %wide.trip.count44.i_unwrap.i = zext i32 %k to i64
  %min.iters.check195 = icmp ult i32 %d, 2
  %n.vec198 = and i64 %wide.trip.count.i_unwrap.i, 4294967294
  %ind.end202 = and i64 %wide.trip.count.i_unwrap.i, 1
  %134 = getelementptr inbounds i8, i8* %"call'mi.i", i64 -8
  %135 = bitcast i8* %134 to double*
  %136 = getelementptr inbounds double, double* %Qs, i64 -1
  %137 = getelementptr inbounds double, double* %Qsb, i64 -1
  %cmp.n203 = icmp eq i64 %n.vec198, %wide.trip.count.i_unwrap.i
  br label %invertfor.inc15.i.us.i

invertfor.inc15.i.us.i:                           ; preds = %invertfor.inc15.i.us.i.preheader, %invertfor.body.i.loopexit.us.i
  %"iv'ac.0.in.us.i" = phi i64 [ %"iv'ac.0.us.i", %invertfor.body.i.loopexit.us.i ], [ %wide.trip.count44.i_unwrap.i, %invertfor.inc15.i.us.i.preheader ]
  %"iv'ac.0.us.i" = add nsw i64 %"iv'ac.0.in.us.i", -1
  %"arrayidx.i'ipg_unwrap27.us.i" = getelementptr inbounds double, double* %"'ipc.i", i64 %"iv'ac.0.us.i"
  %138 = load double, double* %"arrayidx.i'ipg_unwrap27.us.i", align 8, !noalias !75
  store double 0.000000e+00, double* %"arrayidx.i'ipg_unwrap27.us.i", align 8, !noalias !75
  %_unwrap.us.i = trunc i64 %"iv'ac.0.us.i" to i32
  %mul.i_unwrap.us.i = mul nsw i32 %_unwrap.us.i, %d
  %_unwrap22.us.i = sext i32 %mul.i_unwrap.us.i to i64
  br i1 %min.iters.check195, label %invertfor.body3.i.us.i.preheader, label %vector.ph196

vector.ph196:                                     ; preds = %invertfor.inc15.i.us.i
  %broadcast.splatinsert211 = insertelement <2 x double> poison, double %138, i32 0
  %broadcast.splat212 = shufflevector <2 x double> %broadcast.splatinsert211, <2 x double> poison, <2 x i32> zeroinitializer
  br label %vector.body194

vector.body194:                                   ; preds = %vector.body194, %vector.ph196
  %index199 = phi i64 [ 0, %vector.ph196 ], [ %index.next200, %vector.body194 ]
  %139 = xor i64 %index199, -1
  %140 = add i64 %139, %wide.trip.count.i_unwrap.i
  %141 = add nsw i64 %140, %_unwrap22.us.i
  %142 = getelementptr inbounds double, double* %135, i64 %141
  %143 = bitcast double* %142 to <2 x double>*
  %wide.load205 = load <2 x double>, <2 x double>* %143, align 8, !noalias !75
  %reverse206 = shufflevector <2 x double> %wide.load205, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  %144 = bitcast double* %142 to <2 x double>*
  store <2 x double> zeroinitializer, <2 x double>* %144, align 8, !noalias !75
  %145 = getelementptr inbounds double, double* %136, i64 %141
  %146 = bitcast double* %145 to <2 x double>*
  %wide.load207 = load <2 x double>, <2 x double>* %146, align 8, !tbaa !3, !alias.scope !69, !noalias !76
  %reverse208 = shufflevector <2 x double> %wide.load207, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  %147 = call fast <2 x double> @llvm.exp.v2f64(<2 x double> %reverse208)
  %148 = fmul fast <2 x double> %147, %reverse206
  %149 = getelementptr inbounds double, double* %137, i64 %141
  %150 = bitcast double* %149 to <2 x double>*
  %wide.load209 = load <2 x double>, <2 x double>* %150, align 8, !noalias !75
  %reverse210 = shufflevector <2 x double> %wide.load209, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  %151 = fadd fast <2 x double> %reverse210, %broadcast.splat212
  %152 = fadd fast <2 x double> %151, %148
  %reverse213 = shufflevector <2 x double> %152, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  %153 = bitcast double* %149 to <2 x double>*
  store <2 x double> %reverse213, <2 x double>* %153, align 8, !noalias !75
  %index.next200 = add i64 %index199, 2
  %154 = icmp eq i64 %index.next200, %n.vec198
  br i1 %154, label %middle.block192, label %vector.body194, !llvm.loop !96

middle.block192:                                  ; preds = %vector.body194
  br i1 %cmp.n203, label %invertfor.body.i.loopexit.us.i, label %invertfor.body3.i.us.i.preheader

invertfor.body3.i.us.i.preheader:                 ; preds = %invertfor.inc15.i.us.i, %middle.block192
  %"iv1'ac.0.in.us.i.ph" = phi i64 [ %wide.trip.count.i_unwrap.i, %invertfor.inc15.i.us.i ], [ %ind.end202, %middle.block192 ]
  br label %invertfor.body3.i.us.i

invertfor.body3.i.us.i:                           ; preds = %invertfor.body3.i.us.i.preheader, %invertfor.body3.i.us.i
  %"iv1'ac.0.in.us.i" = phi i64 [ %"iv1'ac.0.us.i", %invertfor.body3.i.us.i ], [ %"iv1'ac.0.in.us.i.ph", %invertfor.body3.i.us.i.preheader ]
  %"iv1'ac.0.us.i" = add nsw i64 %"iv1'ac.0.in.us.i", -1
  %_unwrap23.us.i = add nsw i64 %"iv1'ac.0.us.i", %_unwrap22.us.i
  %"arrayidx14.i'ipg_unwrap.us.i" = getelementptr inbounds double, double* %"'ipc21.i", i64 %_unwrap23.us.i
  %155 = load double, double* %"arrayidx14.i'ipg_unwrap.us.i", align 8, !noalias !75
  store double 0.000000e+00, double* %"arrayidx14.i'ipg_unwrap.us.i", align 8, !noalias !75
  %arrayidx5.i_unwrap.us.i = getelementptr inbounds double, double* %Qs, i64 %_unwrap23.us.i
  %_unwrap24.us.i = load double, double* %arrayidx5.i_unwrap.us.i, align 8, !tbaa !3, !alias.scope !69, !noalias !76, !invariant.group !77
  %156 = tail call fast double @llvm.exp.f64(double %_unwrap24.us.i) #14
  %157 = fmul fast double %156, %155
  %"arrayidx5.i'ipg_unwrap.us.i" = getelementptr inbounds double, double* %Qsb, i64 %_unwrap23.us.i
  %158 = load double, double* %"arrayidx5.i'ipg_unwrap.us.i", align 8, !noalias !75
  %159 = fadd fast double %158, %138
  %160 = fadd fast double %159, %157
  store double %160, double* %"arrayidx5.i'ipg_unwrap.us.i", align 8, !noalias !75
  %161 = icmp eq i64 %"iv1'ac.0.us.i", 0
  br i1 %161, label %invertfor.body.i.loopexit.us.i, label %invertfor.body3.i.us.i, !llvm.loop !97

invertfor.body.i.loopexit.us.i:                   ; preds = %invertfor.body3.i.us.i, %middle.block192
  store double 0.000000e+00, double* %"arrayidx.i'ipg_unwrap27.us.i", align 8, !noalias !75
  %162 = icmp eq i64 %"iv'ac.0.us.i", 0
  br i1 %162, label %diffeec_main_term.exit, label %invertfor.inc15.i.us.i

invertpreprocess_qs.exit.i:                       ; preds = %for.end50.i.thread, %invertfor.cond19.preheader.lr.ph.i
  %cmp235.i_unwrap.i = icmp sgt i32 %d, 0
  %wide.trip.count.i_unwrap.i = zext i32 %d to i64
  %or.cond = and i1 %cmp37.i.i, %cmp235.i_unwrap.i
  br i1 %or.cond, label %invertfor.inc15.i.us.i.preheader, label %diffeec_main_term.exit

invertfor.cond19.preheader.lr.ph.i:               ; preds = %invertfor.cond19.preheader.i
  tail call void @free(i8* nonnull %malloccall.i) #14, !noalias !75
  tail call void @free(i8* nonnull %malloccall84.i) #14, !noalias !75
  tail call void @free(i8* nonnull %malloccall100.i) #14, !noalias !75
  tail call void @free(i8* nonnull %malloccall120.i) #14, !noalias !75
  tail call void @free(i8* nonnull %malloccall128.i) #14, !noalias !75
  tail call void @free(i8* nonnull %malloccall135.i) #14, !noalias !75
  tail call void @free(i8* nonnull %malloccall141.i) #14, !noalias !75
  tail call void @free(i8* nonnull %malloccall156.i) #14, !noalias !75
  br label %invertpreprocess_qs.exit.i

invertfor.cond19.preheader.i:                     ; preds = %invertfor.body23.i, %invertfor.end.i
  %"'de32.0.i" = phi double [ %"'de33.0.i", %invertfor.end.i ], [ %287, %invertfor.body23.i ]
  %163 = icmp eq i64 %"iv3'ac.0.i", 0
  br i1 %163, label %invertfor.cond19.preheader.lr.ph.i, label %invertlog_sum_exp.exit.i

invertfor.body23.i:                               ; preds = %invertfor.body.i128.i, %middle.block62, %invertcQtimesx.exit.i
  %164 = icmp eq i64 %"iv5'ac.1.i", 0
  br i1 %164, label %invertfor.cond19.preheader.i, label %invertsqnorm.exit.i

invertfor.body.i128.i:                            ; preds = %invertfor.body.i128.i.preheader, %invertfor.body.i128.i
  %"iv7'ac.0.in.i" = phi i64 [ %"iv7'ac.0.i", %invertfor.body.i128.i ], [ %"iv7'ac.0.in.i.ph", %invertfor.body.i128.i.preheader ]
  %"iv7'ac.0.i" = add nsw i64 %"iv7'ac.0.in.i", -1
  %"arrayidx4.i125'ipg_unwrap.i" = getelementptr inbounds double, double* %"'ipc37.i", i64 %"iv7'ac.0.i"
  %165 = load double, double* %"arrayidx4.i125'ipg_unwrap.i", align 8, !noalias !75
  store double 0.000000e+00, double* %"arrayidx4.i125'ipg_unwrap.i", align 8, !noalias !75
  %"arrayidx2.i123'ipg_unwrap.i" = getelementptr inbounds double, double* %"arrayidx29'ipg_unwrap.i", i64 %"iv7'ac.0.i"
  %166 = load double, double* %"arrayidx2.i123'ipg_unwrap.i", align 8, !noalias !75
  %167 = fsub fast double %166, %165
  store double %167, double* %"arrayidx2.i123'ipg_unwrap.i", align 8, !noalias !75
  %168 = icmp eq i64 %"iv7'ac.0.i", 0
  br i1 %168, label %invertfor.body23.i, label %invertfor.body.i128.i, !llvm.loop !99

invertfor.body.i114.i:                            ; preds = %invertfor.body.i114.i.preheader, %invertfor.body.i114.i
  %"iv9'ac.0.in.i" = phi i64 [ %"iv9'ac.0.i", %invertfor.body.i114.i ], [ %"iv9'ac.0.in.i.ph", %invertfor.body.i114.i.preheader ]
  %"iv9'ac.0.i" = add nsw i64 %"iv9'ac.0.in.i", -1
  %"arrayidx4.i'ipg_unwrap.i" = getelementptr inbounds double, double* %"'ipc40.i", i64 %"iv9'ac.0.i"
  %169 = load double, double* %"arrayidx4.i'ipg_unwrap.i", align 8, !noalias !75
  store double 0.000000e+00, double* %"arrayidx4.i'ipg_unwrap.i", align 8, !noalias !75
  %170 = getelementptr inbounds double, double* %279, i64 %"iv9'ac.0.i"
  %171 = load double, double* %170, align 8, !noalias !75, !invariant.group !100
  %m0diffe.i = fmul fast double %171, %169
  %arrayidx.i111_unwrap.i = getelementptr inbounds double, double* %arrayidx33_unwrap.i, i64 %"iv9'ac.0.i"
  %_unwrap49.i = load double, double* %arrayidx.i111_unwrap.i, align 8, !tbaa !3, !noalias !75, !invariant.group !83
  %m1diffe.i = fmul fast double %_unwrap49.i, %169
  %"arrayidx2.i112'ipg_unwrap.i" = getelementptr inbounds double, double* %"'ipc37.i", i64 %"iv9'ac.0.i"
  %172 = load double, double* %"arrayidx2.i112'ipg_unwrap.i", align 8, !noalias !75
  %173 = fadd fast double %172, %m1diffe.i
  store double %173, double* %"arrayidx2.i112'ipg_unwrap.i", align 8, !noalias !75
  %"arrayidx.i111'ipg_unwrap.i" = getelementptr inbounds double, double* %"arrayidx33'ipg_unwrap.i", i64 %"iv9'ac.0.i"
  %174 = load double, double* %"arrayidx.i111'ipg_unwrap.i", align 8, !noalias !75
  %175 = fadd fast double %174, %m0diffe.i
  store double %175, double* %"arrayidx.i111'ipg_unwrap.i", align 8, !noalias !75
  %176 = icmp eq i64 %"iv9'ac.0.i", 0
  br i1 %176, label %invertfor.body.i128.preheader.i, label %invertfor.body.i114.i, !llvm.loop !101

invertfor.body.i128.preheader.i:                  ; preds = %invertfor.body.i114.i, %middle.block77
  %"arrayidx29'ipg_unwrap.i" = getelementptr inbounds double, double* %meansb, i64 %mul28_unwrap48.i
  br i1 %min.iters.check65, label %invertfor.body.i128.i.preheader, label %vector.ph66

vector.ph66:                                      ; preds = %invertfor.body.i128.preheader.i
  br i1 %127, label %middle.block62.unr-lcssa, label %vector.ph66.new

vector.ph66.new:                                  ; preds = %vector.ph66
  %177 = getelementptr inbounds double, double* %"arrayidx29'ipg_unwrap.i", i64 -1
  %178 = getelementptr inbounds double, double* %"arrayidx29'ipg_unwrap.i", i64 -1
  br label %vector.body64

vector.body64:                                    ; preds = %vector.body64, %vector.ph66.new
  %index69 = phi i64 [ 0, %vector.ph66.new ], [ %index.next70.1, %vector.body64 ]
  %niter = phi i64 [ %unroll_iter, %vector.ph66.new ], [ %niter.nsub.1, %vector.body64 ]
  %179 = xor i64 %index69, -1
  %180 = add i64 %179, %wide.trip.count.i119.i
  %181 = getelementptr inbounds double, double* %129, i64 %180
  %182 = bitcast double* %181 to <2 x double>*
  %wide.load73 = load <2 x double>, <2 x double>* %182, align 8, !noalias !75
  %183 = bitcast double* %181 to <2 x double>*
  store <2 x double> zeroinitializer, <2 x double>* %183, align 8, !noalias !75
  %184 = getelementptr inbounds double, double* %177, i64 %180
  %185 = bitcast double* %184 to <2 x double>*
  %wide.load74 = load <2 x double>, <2 x double>* %185, align 8, !noalias !75
  %186 = fsub fast <2 x double> %wide.load74, %wide.load73
  %187 = bitcast double* %184 to <2 x double>*
  store <2 x double> %186, <2 x double>* %187, align 8, !noalias !75
  %188 = sub nuw nsw i64 -3, %index69
  %189 = add i64 %188, %wide.trip.count.i119.i
  %190 = getelementptr inbounds double, double* %131, i64 %189
  %191 = bitcast double* %190 to <2 x double>*
  %wide.load73.1 = load <2 x double>, <2 x double>* %191, align 8, !noalias !75
  %192 = bitcast double* %190 to <2 x double>*
  store <2 x double> zeroinitializer, <2 x double>* %192, align 8, !noalias !75
  %193 = getelementptr inbounds double, double* %178, i64 %189
  %194 = bitcast double* %193 to <2 x double>*
  %wide.load74.1 = load <2 x double>, <2 x double>* %194, align 8, !noalias !75
  %195 = fsub fast <2 x double> %wide.load74.1, %wide.load73.1
  %196 = bitcast double* %193 to <2 x double>*
  store <2 x double> %195, <2 x double>* %196, align 8, !noalias !75
  %index.next70.1 = add i64 %index69, 4
  %niter.nsub.1 = add i64 %niter, -2
  %niter.ncmp.1 = icmp eq i64 %niter.nsub.1, 0
  br i1 %niter.ncmp.1, label %middle.block62.unr-lcssa, label %vector.body64, !llvm.loop !102

middle.block62.unr-lcssa:                         ; preds = %vector.body64, %vector.ph66
  %index69.unr = phi i64 [ 0, %vector.ph66 ], [ %index.next70.1, %vector.body64 ]
  br i1 %lcmp.mod224.not, label %middle.block62, label %vector.body64.epil

vector.body64.epil:                               ; preds = %middle.block62.unr-lcssa
  %197 = xor i64 %index69.unr, -1
  %198 = add i64 %197, %wide.trip.count.i119.i
  %199 = getelementptr inbounds double, double* %133, i64 %198
  %200 = bitcast double* %199 to <2 x double>*
  %wide.load73.epil = load <2 x double>, <2 x double>* %200, align 8, !noalias !75
  %201 = bitcast double* %199 to <2 x double>*
  store <2 x double> zeroinitializer, <2 x double>* %201, align 8, !noalias !75
  %202 = getelementptr inbounds double, double* %"arrayidx29'ipg_unwrap.i", i64 -1
  %203 = getelementptr inbounds double, double* %202, i64 %198
  %204 = bitcast double* %203 to <2 x double>*
  %wide.load74.epil = load <2 x double>, <2 x double>* %204, align 8, !noalias !75
  %205 = fsub fast <2 x double> %wide.load74.epil, %wide.load73.epil
  %206 = bitcast double* %203 to <2 x double>*
  store <2 x double> %205, <2 x double>* %206, align 8, !noalias !75
  br label %middle.block62

middle.block62:                                   ; preds = %middle.block62.unr-lcssa, %vector.body64.epil
  br i1 %cmp.n72, label %invertfor.body23.i, label %invertfor.body.i128.i.preheader

invertfor.body.i128.i.preheader:                  ; preds = %invertfor.body.i128.preheader.i, %middle.block62
  %"iv7'ac.0.in.i.ph" = phi i64 [ %wide.trip.count.i119.i, %invertfor.body.i128.preheader.i ], [ %ind.end, %middle.block62 ]
  br label %invertfor.body.i128.i

mergeinvertfor.body13.i_for.cond5.loopexit.i.loopexit.i: ; preds = %invertfor.cond5.loopexit.i.i
  %207 = xor i64 %"iv11'ac.0.in.i", -1
  %_unwrap57.i = add i64 %207, %wide.trip.count.i119.i
  %208 = getelementptr inbounds double, double* %279, i64 %"iv11'ac.0.i"
  %209 = load double, double* %208, align 8, !noalias !75, !invariant.group !100
  %_unwrap72.i = trunc i64 %"iv11'ac.0.i" to i32
  %_unwrap73.i = xor i32 %_unwrap72.i, -1
  %sub9.i_unwrap.i = add i32 %mul8.i.i, %_unwrap73.i
  %mul10.i_unwrap.i = mul nsw i32 %sub9.i_unwrap.i, %_unwrap72.i
  %div.i_unwrap.i = sdiv i32 %mul10.i_unwrap.i, 2
  %_unwrap75.i = sext i32 %div.i_unwrap.i to i64
  %min.iters.check105 = icmp ult i64 %indvar, 4
  br i1 %min.iters.check105, label %invertfor.body13.i.i.preheader, label %vector.ph106

vector.ph106:                                     ; preds = %mergeinvertfor.body13.i_for.cond5.loopexit.i.loopexit.i
  %n.vec108 = and i64 %indvar, -4
  %ind.end112 = sub i64 %_unwrap57.i, %n.vec108
  %broadcast.splatinsert120 = insertelement <2 x double> poison, double %209, i32 0
  %broadcast.splat121 = shufflevector <2 x double> %broadcast.splatinsert120, <2 x double> poison, <2 x i32> zeroinitializer
  %broadcast.splatinsert122 = insertelement <2 x double> poison, double %209, i32 0
  %broadcast.splat123 = shufflevector <2 x double> %broadcast.splatinsert122, <2 x double> poison, <2 x i32> zeroinitializer
  br label %vector.body104

vector.body104:                                   ; preds = %vector.body104, %vector.ph106
  %index109 = phi i64 [ 0, %vector.ph106 ], [ %index.next110, %vector.body104 ]
  %vec.phi = phi <2 x double> [ zeroinitializer, %vector.ph106 ], [ %230, %vector.body104 ]
  %vec.phi114 = phi <2 x double> [ zeroinitializer, %vector.ph106 ], [ %231, %vector.body104 ]
  %offset.idx115 = sub i64 %_unwrap57.i, %index109
  %210 = add i64 %offset.idx115, %"iv11'ac.0.in.i"
  %211 = getelementptr inbounds double, double* %"'ipc40.i", i64 %210
  %212 = getelementptr inbounds double, double* %211, i64 -1
  %213 = bitcast double* %212 to <2 x double>*
  %wide.load116 = load <2 x double>, <2 x double>* %213, align 8, !noalias !75
  %reverse117 = shufflevector <2 x double> %wide.load116, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  %214 = getelementptr inbounds double, double* %211, i64 -2
  %215 = getelementptr inbounds double, double* %214, i64 -1
  %216 = bitcast double* %215 to <2 x double>*
  %wide.load118 = load <2 x double>, <2 x double>* %216, align 8, !noalias !75
  %reverse119 = shufflevector <2 x double> %wide.load118, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  %217 = fmul fast <2 x double> %reverse117, %broadcast.splat121
  %218 = fmul fast <2 x double> %reverse119, %broadcast.splat123
  %219 = add i64 %offset.idx115, %_unwrap75.i
  %220 = getelementptr inbounds double, double* %arrayidx35_unwrap.i, i64 %219
  %221 = getelementptr inbounds double, double* %220, i64 -1
  %222 = bitcast double* %221 to <2 x double>*
  %wide.load124 = load <2 x double>, <2 x double>* %222, align 8, !tbaa !3, !alias.scope !71, !noalias !85
  %223 = getelementptr inbounds double, double* %220, i64 -2
  %224 = getelementptr inbounds double, double* %223, i64 -1
  %225 = bitcast double* %224 to <2 x double>*
  %wide.load126 = load <2 x double>, <2 x double>* %225, align 8, !tbaa !3, !alias.scope !71, !noalias !85
  %226 = fmul fast <2 x double> %wide.load124, %wide.load116
  %227 = shufflevector <2 x double> %226, <2 x double> undef, <2 x i32> <i32 1, i32 0>
  %228 = fmul fast <2 x double> %wide.load126, %wide.load118
  %229 = shufflevector <2 x double> %228, <2 x double> undef, <2 x i32> <i32 1, i32 0>
  %230 = fadd fast <2 x double> %227, %vec.phi
  %231 = fadd fast <2 x double> %229, %vec.phi114
  %232 = getelementptr inbounds double, double* %"arrayidx35'ipg_unwrap.i", i64 %219
  %233 = getelementptr inbounds double, double* %232, i64 -1
  %234 = bitcast double* %233 to <2 x double>*
  %wide.load128 = load <2 x double>, <2 x double>* %234, align 8, !noalias !75
  %reverse129 = shufflevector <2 x double> %wide.load128, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  %235 = getelementptr inbounds double, double* %232, i64 -2
  %236 = getelementptr inbounds double, double* %235, i64 -1
  %237 = bitcast double* %236 to <2 x double>*
  %wide.load130 = load <2 x double>, <2 x double>* %237, align 8, !noalias !75
  %reverse131 = shufflevector <2 x double> %wide.load130, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  %238 = fadd fast <2 x double> %reverse129, %217
  %239 = fadd fast <2 x double> %reverse131, %218
  %reverse132 = shufflevector <2 x double> %238, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  %240 = bitcast double* %233 to <2 x double>*
  store <2 x double> %reverse132, <2 x double>* %240, align 8, !noalias !75
  %reverse133 = shufflevector <2 x double> %239, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  %241 = bitcast double* %236 to <2 x double>*
  store <2 x double> %reverse133, <2 x double>* %241, align 8, !noalias !75
  %index.next110 = add i64 %index109, 4
  %242 = icmp eq i64 %index.next110, %n.vec108
  br i1 %242, label %middle.block102, label %vector.body104, !llvm.loop !103

middle.block102:                                  ; preds = %vector.body104
  %bin.rdx = fadd fast <2 x double> %231, %230
  %243 = call fast double @llvm.vector.reduce.fadd.v2f64(double -0.000000e+00, <2 x double> %bin.rdx)
  %cmp.n113 = icmp eq i64 %indvar, %n.vec108
  br i1 %cmp.n113, label %invertfor.body13.lr.ph.i.i, label %invertfor.body13.i.i.preheader

invertfor.body13.i.i.preheader:                   ; preds = %mergeinvertfor.body13.i_for.cond5.loopexit.i.loopexit.i, %middle.block102
  %"'de59.4.i.ph" = phi double [ 0.000000e+00, %mergeinvertfor.body13.i_for.cond5.loopexit.i.loopexit.i ], [ %243, %middle.block102 ]
  %"iv13'ac.0.i.ph" = phi i64 [ %_unwrap57.i, %mergeinvertfor.body13.i_for.cond5.loopexit.i.loopexit.i ], [ %ind.end112, %middle.block102 ]
  br label %invertfor.body13.i.i

invertfor.cond5.loopexit.i.i:                     ; preds = %invertcQtimesx.exit.loopexit.i, %invertfor.body7.i.i
  %indvar = phi i64 [ 0, %invertcQtimesx.exit.loopexit.i ], [ %indvar.next, %invertfor.body7.i.i ]
  %"iv11'ac.0.in.i" = phi i64 [ %wide.trip.count.i119.i, %invertcQtimesx.exit.loopexit.i ], [ %"iv11'ac.0.i", %invertfor.body7.i.i ]
  %"iv11'ac.0.i" = add nsw i64 %"iv11'ac.0.in.i", -1
  %cmp1254.i_unwrap.i = icmp ult i64 %"iv11'ac.0.in.i", %wide.trip.count.i119.i
  br i1 %cmp1254.i_unwrap.i, label %mergeinvertfor.body13.i_for.cond5.loopexit.i.loopexit.i, label %invertfor.body7.i.i

invertfor.body7.i.i:                              ; preds = %invertfor.body13.lr.ph.i.i, %invertfor.cond5.loopexit.i.i
  %244 = icmp eq i64 %"iv11'ac.0.i", 0
  %indvar.next = add i64 %indvar, 1
  br i1 %244, label %invertfor.body.i114.preheader.i, label %invertfor.cond5.loopexit.i.i

invertfor.body.i114.preheader.i:                  ; preds = %invertfor.body7.i.i
  %mul28_unwrap48.i = mul nsw i64 %"iv5'ac.1.i", %conv7.i
  %arrayidx33_unwrap.i = getelementptr inbounds double, double* %0, i64 %mul28_unwrap48.i
  %"arrayidx33'ipg_unwrap.i" = getelementptr inbounds double, double* %"'ipc21.i", i64 %mul28_unwrap48.i
  br i1 %min.iters.check80, label %invertfor.body.i114.i.preheader, label %vector.ph81

vector.ph81:                                      ; preds = %invertfor.body.i114.preheader.i
  %245 = getelementptr inbounds double, double* %279, i64 -1
  %246 = getelementptr inbounds double, double* %arrayidx33_unwrap.i, i64 -1
  %247 = getelementptr inbounds double, double* %"arrayidx33'ipg_unwrap.i", i64 -1
  br label %vector.body79

vector.body79:                                    ; preds = %vector.body79, %vector.ph81
  %index84 = phi i64 [ 0, %vector.ph81 ], [ %index.next85, %vector.body79 ]
  %248 = xor i64 %index84, -1
  %249 = add i64 %248, %wide.trip.count.i119.i
  %250 = getelementptr inbounds double, double* %124, i64 %249
  %251 = bitcast double* %250 to <2 x double>*
  %wide.load90 = load <2 x double>, <2 x double>* %251, align 8, !noalias !75
  %252 = bitcast double* %250 to <2 x double>*
  store <2 x double> zeroinitializer, <2 x double>* %252, align 8, !noalias !75
  %253 = getelementptr inbounds double, double* %245, i64 %249
  %254 = bitcast double* %253 to <2 x double>*
  %wide.load92 = load <2 x double>, <2 x double>* %254, align 8, !noalias !75
  %255 = fmul fast <2 x double> %wide.load92, %wide.load90
  %256 = getelementptr inbounds double, double* %246, i64 %249
  %257 = bitcast double* %256 to <2 x double>*
  %wide.load94 = load <2 x double>, <2 x double>* %257, align 8, !tbaa !3, !noalias !75
  %258 = fmul fast <2 x double> %wide.load94, %wide.load90
  %259 = getelementptr inbounds double, double* %126, i64 %249
  %260 = bitcast double* %259 to <2 x double>*
  %wide.load96 = load <2 x double>, <2 x double>* %260, align 8, !noalias !75
  %261 = fadd fast <2 x double> %wide.load96, %258
  %262 = bitcast double* %259 to <2 x double>*
  store <2 x double> %261, <2 x double>* %262, align 8, !noalias !75
  %263 = getelementptr inbounds double, double* %247, i64 %249
  %264 = bitcast double* %263 to <2 x double>*
  %wide.load99 = load <2 x double>, <2 x double>* %264, align 8, !noalias !75
  %265 = fadd fast <2 x double> %wide.load99, %255
  %266 = bitcast double* %263 to <2 x double>*
  store <2 x double> %265, <2 x double>* %266, align 8, !noalias !75
  %index.next85 = add i64 %index84, 2
  %267 = icmp eq i64 %index.next85, %n.vec83
  br i1 %267, label %middle.block77, label %vector.body79, !llvm.loop !104

middle.block77:                                   ; preds = %vector.body79
  br i1 %cmp.n88, label %invertfor.body.i128.preheader.i, label %invertfor.body.i114.i.preheader

invertfor.body.i114.i.preheader:                  ; preds = %invertfor.body.i114.preheader.i, %middle.block77
  %"iv9'ac.0.in.i.ph" = phi i64 [ %wide.trip.count.i119.i, %invertfor.body.i114.preheader.i ], [ %ind.end87, %middle.block77 ]
  br label %invertfor.body.i114.i

invertfor.body13.lr.ph.i.i:                       ; preds = %invertfor.body13.i.i, %middle.block102
  %.lcssa24 = phi double [ %243, %middle.block102 ], [ %271, %invertfor.body13.i.i ]
  %"arrayidx19.i'ipg_unwrap.i" = getelementptr inbounds double, double* %"'ipc37.i", i64 %"iv11'ac.0.i"
  %268 = load double, double* %"arrayidx19.i'ipg_unwrap.i", align 8, !noalias !75
  %269 = fadd fast double %268, %.lcssa24
  store double %269, double* %"arrayidx19.i'ipg_unwrap.i", align 8, !noalias !75
  br label %invertfor.body7.i.i

invertfor.body13.i.i:                             ; preds = %invertfor.body13.i.i.preheader, %invertfor.body13.i.i
  %"'de59.4.i" = phi double [ %271, %invertfor.body13.i.i ], [ %"'de59.4.i.ph", %invertfor.body13.i.i.preheader ]
  %"iv13'ac.0.i" = phi i64 [ %275, %invertfor.body13.i.i ], [ %"iv13'ac.0.i.ph", %invertfor.body13.i.i.preheader ]
  %_unwrap61.i = add i64 %"iv13'ac.0.i", %"iv11'ac.0.in.i"
  %"arrayidx15.i'ipg_unwrap.i" = getelementptr inbounds double, double* %"'ipc40.i", i64 %_unwrap61.i
  %270 = load double, double* %"arrayidx15.i'ipg_unwrap.i", align 8, !noalias !75
  %m0diffe71.i = fmul fast double %270, %209
  %_unwrap76.i = add i64 %"iv13'ac.0.i", %_unwrap75.i
  %arrayidx17.i_unwrap.i = getelementptr inbounds double, double* %arrayidx35_unwrap.i, i64 %_unwrap76.i
  %_unwrap77.i = load double, double* %arrayidx17.i_unwrap.i, align 8, !tbaa !3, !alias.scope !71, !noalias !85, !invariant.group !87
  %m1diffe78.i = fmul fast double %_unwrap77.i, %270
  %271 = fadd fast double %m1diffe78.i, %"'de59.4.i"
  %"arrayidx17.i'ipg_unwrap.i" = getelementptr inbounds double, double* %"arrayidx35'ipg_unwrap.i", i64 %_unwrap76.i
  %272 = load double, double* %"arrayidx17.i'ipg_unwrap.i", align 8, !noalias !75
  %273 = fadd fast double %272, %m0diffe71.i
  store double %273, double* %"arrayidx17.i'ipg_unwrap.i", align 8, !noalias !75
  %274 = icmp eq i64 %"iv13'ac.0.i", 0
  %275 = add nsw i64 %"iv13'ac.0.i", -1
  br i1 %274, label %invertfor.body13.lr.ph.i.i, label %invertfor.body13.i.i, !llvm.loop !105

invertcQtimesx.exit.loopexit.i:                   ; preds = %invertcQtimesx.exit.i
  %276 = load double, double* %"'ipc40.i", align 8, !noalias !75
  %277 = fadd fast double %276, %282
  store double %277, double* %"'ipc40.i", align 8, !noalias !75
  %278 = mul nuw nsw i64 %316, %wide.trip.count.i119.i
  %mul34_unwrap.i = mul nsw i64 %"iv5'ac.1.i", %conv.i
  %arrayidx35_unwrap.i = getelementptr inbounds double, double* %Ls, i64 %mul34_unwrap.i
  %"arrayidx35'ipg_unwrap.i" = getelementptr inbounds double, double* %Lsb, i64 %mul34_unwrap.i
  %279 = getelementptr inbounds double, double* %_malloccache.i, i64 %278
  br label %invertfor.cond5.loopexit.i.i

invertcQtimesx.exit.i:                            ; preds = %invertfor.body.i109.i, %invertsqnorm.exit.i, %middle.block134
  %280 = getelementptr inbounds double, double* %_malloccache85.i, i64 %316
  %281 = load double, double* %280, align 8, !noalias !75, !invariant.group !89
  %factor.i = fmul fast double %281, %315
  %282 = fsub fast double %"'de35.1.i", %factor.i
  %"arrayidx39'ipg_unwrap.i" = getelementptr inbounds double, double* %"'ipc.i", i64 %"iv5'ac.1.i"
  %283 = load double, double* %"arrayidx39'ipg_unwrap.i", align 8, !noalias !75
  %284 = fadd fast double %283, %315
  store double %284, double* %"arrayidx39'ipg_unwrap.i", align 8, !noalias !75
  %"arrayidx38'ipg_unwrap.i" = getelementptr inbounds double, double* %alphasb, i64 %"iv5'ac.1.i"
  %285 = load double, double* %"arrayidx38'ipg_unwrap.i", align 8, !noalias !75
  %286 = fadd fast double %285, %315
  store double %286, double* %"arrayidx38'ipg_unwrap.i", align 8, !noalias !75
  %287 = select fast i1 %cmp10.i.i, double 0.000000e+00, double %282
  br i1 %cmp10.i.i, label %invertcQtimesx.exit.loopexit.i, label %invertfor.body23.i

invertfor.body.i109.i:                            ; preds = %invertfor.body.i109.i.preheader, %invertfor.body.i109.i
  %"iv15'ac.0.i" = phi i64 [ %294, %invertfor.body.i109.i ], [ %"iv15'ac.0.i.ph", %invertfor.body.i109.i.preheader ]
  %288 = getelementptr inbounds double, double* %296, i64 %"iv15'ac.0.i"
  %289 = load double, double* %288, align 8, !noalias !75, !invariant.group !106
  %iv.next16_unwrap.i = add nuw nsw i64 %"iv15'ac.0.i", 1
  %"arrayidx2.i'ipg_unwrap.i" = getelementptr inbounds double, double* %"'ipc40.i", i64 %iv.next16_unwrap.i
  %290 = load double, double* %"arrayidx2.i'ipg_unwrap.i", align 8, !noalias !75
  %291 = fmul fast double %289, %315
  %292 = fsub fast double %290, %291
  store double %292, double* %"arrayidx2.i'ipg_unwrap.i", align 8, !noalias !75
  %293 = icmp eq i64 %"iv15'ac.0.i", 0
  %294 = add nsw i64 %"iv15'ac.0.i", -1
  br i1 %293, label %invertcQtimesx.exit.i, label %invertfor.body.i109.i, !llvm.loop !107

mergeinvertfor.body.i109_sqnorm.exit.loopexit.i:  ; preds = %invertsqnorm.exit.i
  %295 = mul nuw nsw i64 %316, %_unwrap97.i
  %296 = getelementptr inbounds double, double* %_malloccache101.i, i64 %295
  br i1 %min.iters.check137, label %invertfor.body.i109.i.preheader, label %vector.ph138

vector.ph138:                                     ; preds = %mergeinvertfor.body.i109_sqnorm.exit.loopexit.i
  %broadcast.splatinsert155 = insertelement <2 x double> poison, double %315, i32 0
  %broadcast.splat156 = shufflevector <2 x double> %broadcast.splatinsert155, <2 x double> poison, <2 x i32> zeroinitializer
  %broadcast.splatinsert157 = insertelement <2 x double> poison, double %315, i32 0
  %broadcast.splat158 = shufflevector <2 x double> %broadcast.splatinsert157, <2 x double> poison, <2 x i32> zeroinitializer
  br label %vector.body136

vector.body136:                                   ; preds = %vector.body136, %vector.ph138
  %index141 = phi i64 [ 0, %vector.ph138 ], [ %index.next142, %vector.body136 ]
  %offset.idx146 = sub i64 %_unwrap115.i, %index141
  %297 = getelementptr inbounds double, double* %296, i64 %offset.idx146
  %298 = getelementptr inbounds double, double* %297, i64 -1
  %299 = bitcast double* %298 to <2 x double>*
  %wide.load147 = load <2 x double>, <2 x double>* %299, align 8, !noalias !75
  %reverse148 = shufflevector <2 x double> %wide.load147, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  %300 = getelementptr inbounds double, double* %297, i64 -2
  %301 = getelementptr inbounds double, double* %300, i64 -1
  %302 = bitcast double* %301 to <2 x double>*
  %wide.load149 = load <2 x double>, <2 x double>* %302, align 8, !noalias !75
  %reverse150 = shufflevector <2 x double> %wide.load149, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  %303 = add nuw nsw i64 %offset.idx146, 1
  %304 = getelementptr inbounds double, double* %"'ipc40.i", i64 %offset.idx146
  %305 = bitcast double* %304 to <2 x double>*
  %wide.load151 = load <2 x double>, <2 x double>* %305, align 8, !noalias !75
  %reverse152 = shufflevector <2 x double> %wide.load151, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  %306 = getelementptr inbounds double, double* %122, i64 %303
  %307 = bitcast double* %306 to <2 x double>*
  %wide.load153 = load <2 x double>, <2 x double>* %307, align 8, !noalias !75
  %reverse154 = shufflevector <2 x double> %wide.load153, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  %308 = fmul fast <2 x double> %reverse148, %broadcast.splat156
  %309 = fmul fast <2 x double> %reverse150, %broadcast.splat158
  %310 = fsub fast <2 x double> %reverse152, %308
  %311 = fsub fast <2 x double> %reverse154, %309
  %reverse159 = shufflevector <2 x double> %310, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  %312 = bitcast double* %304 to <2 x double>*
  store <2 x double> %reverse159, <2 x double>* %312, align 8, !noalias !75
  %reverse160 = shufflevector <2 x double> %311, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  %313 = bitcast double* %306 to <2 x double>*
  store <2 x double> %reverse160, <2 x double>* %313, align 8, !noalias !75
  %index.next142 = add i64 %index141, 4
  %314 = icmp eq i64 %index.next142, %n.vec140
  br i1 %314, label %middle.block134, label %vector.body136, !llvm.loop !108

middle.block134:                                  ; preds = %vector.body136
  br i1 %cmp.n145, label %invertcQtimesx.exit.i, label %invertfor.body.i109.i.preheader

invertfor.body.i109.i.preheader:                  ; preds = %mergeinvertfor.body.i109_sqnorm.exit.loopexit.i, %middle.block134
  %"iv15'ac.0.i.ph" = phi i64 [ %_unwrap115.i, %mergeinvertfor.body.i109_sqnorm.exit.loopexit.i ], [ %ind.end144, %middle.block134 ]
  br label %invertfor.body.i109.i

invertsqnorm.exit.i:                              ; preds = %invertfor.end.loopexit.i, %invertfor.body23.i
  %"'de35.1.i" = phi double [ %"'de33.0.i", %invertfor.end.loopexit.i ], [ %287, %invertfor.body23.i ]
  %"iv5'ac.1.in.i" = phi i64 [ %conv4.i, %invertfor.end.loopexit.i ], [ %"iv5'ac.1.i", %invertfor.body23.i ]
  %"iv5'ac.1.i" = add nsw i64 %"iv5'ac.1.in.i", -1
  %"arrayidx44'ipg_unwrap.i" = getelementptr inbounds double, double* %"'ipc116.i", i64 %"iv5'ac.1.i"
  %315 = load double, double* %"arrayidx44'ipg_unwrap.i", align 8, !noalias !75
  store double 0.000000e+00, double* %"arrayidx44'ipg_unwrap.i", align 8, !noalias !75
  %316 = add nuw nsw i64 %"iv5'ac.1.i", %319
  br i1 %cmp15.i.i, label %mergeinvertfor.body.i109_sqnorm.exit.loopexit.i, label %invertcQtimesx.exit.i

invertfor.end.loopexit.i:                         ; preds = %invertfor.end.i
  %317 = load double, double* %"'ipc116.i", align 8, !noalias !75
  %318 = fadd fast double %317, %"'de31.0.i"
  store double %318, double* %"'ipc116.i", align 8, !noalias !75
  %319 = mul nuw nsw i64 %"iv3'ac.0.i", %conv4.i
  br label %invertsqnorm.exit.i

invertfor.end.loopexit162.i:                      ; preds = %invertfor.body.i.i.i, %invertfor.body.i.i.i.prol.loopexit
  %320 = icmp eq i64 %_unwrap130.i, 0
  %321 = select fast i1 %320, double %"m.0.lcssa.i.i'de.0.i", double 0.000000e+00
  br label %invertfor.end.i

invertfor.end.i:                                  ; preds = %invertarr_max.exit.i.i, %invertfor.end.loopexit162.i
  %"m.0.lcssa.i.i'de.0.pn.i" = phi double [ %"m.0.lcssa.i.i'de.0.i", %invertarr_max.exit.i.i ], [ %321, %invertfor.end.loopexit162.i ]
  %"'de31.0.i" = fadd fast double %"m.0.lcssa.i.i'de.0.pn.i", %"'de31.2.i"
  %322 = select fast i1 %cmp37.i.i, double 0.000000e+00, double %"'de31.0.i"
  br i1 %cmp37.i.i, label %invertfor.end.loopexit.i, label %invertfor.cond19.preheader.i

invertfor.body.i.i.i:                             ; preds = %invertfor.body.i.i.i.prol.loopexit, %invertfor.body.i.i.i
  %"iv17'ac.0.i" = phi i64 [ %332, %invertfor.body.i.i.i ], [ %"iv17'ac.0.i.unr.ph", %invertfor.body.i.i.i.prol.loopexit ]
  %iv.next18_unwrap.i = add nuw nsw i64 %"iv17'ac.0.i", 1
  %323 = icmp eq i64 %_unwrap130.i, %iv.next18_unwrap.i
  %324 = select fast i1 %323, double %"m.0.lcssa.i.i'de.0.i", double 0.000000e+00
  %"arrayidx1.i.i'ipg_unwrap.i" = getelementptr inbounds double, double* %"'ipc116.i", i64 %iv.next18_unwrap.i
  %325 = load double, double* %"arrayidx1.i.i'ipg_unwrap.i", align 8, !noalias !75
  %326 = fadd fast double %324, %325
  store double %326, double* %"arrayidx1.i.i'ipg_unwrap.i", align 8, !noalias !75
  %327 = icmp eq i64 %_unwrap130.i, %"iv17'ac.0.i"
  %328 = select fast i1 %327, double %"m.0.lcssa.i.i'de.0.i", double 0.000000e+00
  %"arrayidx1.i.i'ipg_unwrap.i.1" = getelementptr inbounds double, double* %"'ipc116.i", i64 %"iv17'ac.0.i"
  %329 = load double, double* %"arrayidx1.i.i'ipg_unwrap.i.1", align 8, !noalias !75
  %330 = fadd fast double %328, %329
  store double %330, double* %"arrayidx1.i.i'ipg_unwrap.i.1", align 8, !noalias !75
  %331 = icmp eq i64 %"iv17'ac.0.i", 1
  %332 = add nsw i64 %"iv17'ac.0.i", -2
  br i1 %331, label %invertfor.end.loopexit162.i, label %invertfor.body.i.i.i

invertarr_max.exit.i.loopexit.i:                  ; preds = %invertarr_max.exit.i.i
  %333 = getelementptr inbounds i8, i8* %malloccall120.i, i64 %"iv3'ac.0.i"
  %334 = bitcast i8* %333 to i1*
  %335 = load i1, i1* %334, align 1, !noalias !75, !invariant.group !91
  %336 = getelementptr inbounds i64, i64* %"!manual_lcssa126_malloccache.i", i64 %"iv3'ac.0.i"
  %337 = load i64, i64* %336, align 8, !noalias !75, !invariant.group !92
  %_unwrap130.i = select i1 %335, i64 %15, i64 %337
  br i1 %lcmp.mod.not, label %invertfor.body.i.i.i.prol.loopexit, label %invertfor.body.i.i.i.prol

invertfor.body.i.i.i.prol:                        ; preds = %invertarr_max.exit.i.loopexit.i
  %338 = icmp eq i64 %_unwrap130.i, %iv.next18_unwrap.i.prol
  %339 = select fast i1 %338, double %"m.0.lcssa.i.i'de.0.i", double 0.000000e+00
  %340 = load double, double* %"arrayidx1.i.i'ipg_unwrap.i.prol", align 8, !noalias !75
  %341 = fadd fast double %339, %340
  store double %341, double* %"arrayidx1.i.i'ipg_unwrap.i.prol", align 8, !noalias !75
  br label %invertfor.body.i.i.i.prol.loopexit

invertfor.body.i.i.i.prol.loopexit:               ; preds = %invertfor.body.i.i.i.prol, %invertarr_max.exit.i.loopexit.i
  %"iv17'ac.0.i.unr.ph" = phi i64 [ %119, %invertfor.body.i.i.i.prol ], [ %_unwrap149.i, %invertarr_max.exit.i.loopexit.i ]
  br i1 %120, label %invertfor.end.loopexit162.i, label %invertfor.body.i.i.i

invertarr_max.exit.i.i:                           ; preds = %invertlog_sum_exp.exit.i, %invertfor.body.preheader.i.i
  %"add.i'de.0.i" = phi double [ %"add.i'de.1.i", %invertfor.body.preheader.i.i ], [ %388, %invertlog_sum_exp.exit.i ]
  %"add.i136'de.0.i" = phi double [ 0.000000e+00, %invertfor.body.preheader.i.i ], [ %390, %invertlog_sum_exp.exit.i ]
  %"m.0.lcssa.i.i'de.0.i" = phi double [ %345, %invertfor.body.preheader.i.i ], [ 1.000000e+00, %invertlog_sum_exp.exit.i ]
  %"'de31.2.i" = phi double [ %344, %invertfor.body.preheader.i.i ], [ %"'de31.3.i", %invertlog_sum_exp.exit.i ]
  br i1 %cmp13.i.i.i, label %invertarr_max.exit.i.loopexit.i, label %invertfor.end.i

invertfor.body.preheader.i.loopexit.i:            ; preds = %invertfor.body.for.body_crit_edge.i.i, %middle.block161
  %.lcssa = phi double [ %380, %middle.block161 ], [ %351, %invertfor.body.for.body_crit_edge.i.i ]
  %342 = fadd fast double %388, %"add.i136'de.3.i"
  br label %invertfor.body.preheader.i.i

invertfor.body.preheader.i.i:                     ; preds = %staging.i, %invertfor.body.preheader.i.loopexit.i
  %"add.i'de.1.i" = phi double [ %388, %staging.i ], [ 0.000000e+00, %invertfor.body.preheader.i.loopexit.i ]
  %"add.i136'de.1.i" = phi double [ %389, %staging.i ], [ %342, %invertfor.body.preheader.i.loopexit.i ]
  %"m.0.lcssa.i.i'de.1.i" = phi double [ 1.000000e+00, %staging.i ], [ %.lcssa, %invertfor.body.preheader.i.loopexit.i ]
  %343 = fmul fast double %"add.i136'de.1.i", %383
  %344 = fadd fast double %343, %"'de31.3.i"
  %345 = fsub fast double %"m.0.lcssa.i.i'de.1.i", %343
  br label %invertarr_max.exit.i.i

invertfor.body.for.body_crit_edge.i.i:            ; preds = %invertfor.body.for.body_crit_edge.i.i.preheader, %invertfor.body.for.body_crit_edge.i.i
  %"m.0.lcssa.i.i'de.2.i" = phi double [ %351, %invertfor.body.for.body_crit_edge.i.i ], [ %"m.0.lcssa.i.i'de.2.i.ph", %invertfor.body.for.body_crit_edge.i.i.preheader ]
  %"iv19'ac.2.i" = phi i64 [ %355, %invertfor.body.for.body_crit_edge.i.i ], [ %"iv19'ac.2.i.ph", %invertfor.body.for.body_crit_edge.i.i.preheader ]
  %346 = add nuw nsw i64 %"iv19'ac.2.i", %356
  %347 = getelementptr inbounds double, double* %sub.i_malloccache.i, i64 %346
  %348 = load double, double* %347, align 8, !noalias !75, !invariant.group !94
  %349 = tail call fast double @llvm.exp.f64(double %348) #14
  %350 = fmul fast double %349, %388
  %351 = fsub fast double %"m.0.lcssa.i.i'de.2.i", %350
  %iv.next20_unwrap.i = add nuw nsw i64 %"iv19'ac.2.i", 1
  %"arrayidx.phi.trans.insert.i'ipg_unwrap.i" = getelementptr inbounds double, double* %"'ipc116.i", i64 %iv.next20_unwrap.i
  %352 = load double, double* %"arrayidx.phi.trans.insert.i'ipg_unwrap.i", align 8, !noalias !75
  %353 = fadd fast double %350, %352
  store double %353, double* %"arrayidx.phi.trans.insert.i'ipg_unwrap.i", align 8, !noalias !75
  %354 = icmp eq i64 %"iv19'ac.2.i", 0
  %355 = add nsw i64 %"iv19'ac.2.i", -1
  br i1 %354, label %invertfor.body.preheader.i.loopexit.i, label %invertfor.body.for.body_crit_edge.i.i, !llvm.loop !109

mergeinvertfor.body.for.body_crit_edge.i_log_sum_exp.exit.loopexit.i: ; preds = %staging.i
  %356 = mul nuw nsw i64 %"iv3'ac.0.i", %15
  br i1 %min.iters.check164, label %invertfor.body.for.body_crit_edge.i.i.preheader, label %vector.ph165

vector.ph165:                                     ; preds = %mergeinvertfor.body.for.body_crit_edge.i_log_sum_exp.exit.loopexit.i
  %broadcast.splatinsert180 = insertelement <2 x double> poison, double %388, i32 0
  %broadcast.splat181 = shufflevector <2 x double> %broadcast.splatinsert180, <2 x double> poison, <2 x i32> zeroinitializer
  %broadcast.splatinsert182 = insertelement <2 x double> poison, double %388, i32 0
  %broadcast.splat183 = shufflevector <2 x double> %broadcast.splatinsert182, <2 x double> poison, <2 x i32> zeroinitializer
  br label %vector.body163

vector.body163:                                   ; preds = %vector.body163, %vector.ph165
  %index168 = phi i64 [ 0, %vector.ph165 ], [ %index.next169, %vector.body163 ]
  %vec.phi173 = phi <2 x double> [ <double 1.000000e+00, double 0.000000e+00>, %vector.ph165 ], [ %368, %vector.body163 ]
  %vec.phi174 = phi <2 x double> [ zeroinitializer, %vector.ph165 ], [ %369, %vector.body163 ]
  %offset.idx175 = sub i64 %_unwrap149.i, %index168
  %357 = add nuw nsw i64 %offset.idx175, %356
  %358 = getelementptr inbounds double, double* %sub.i_malloccache.i, i64 %357
  %359 = getelementptr inbounds double, double* %358, i64 -1
  %360 = bitcast double* %359 to <2 x double>*
  %wide.load176 = load <2 x double>, <2 x double>* %360, align 8, !noalias !75
  %reverse177 = shufflevector <2 x double> %wide.load176, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  %361 = getelementptr inbounds double, double* %358, i64 -2
  %362 = getelementptr inbounds double, double* %361, i64 -1
  %363 = bitcast double* %362 to <2 x double>*
  %wide.load178 = load <2 x double>, <2 x double>* %363, align 8, !noalias !75
  %reverse179 = shufflevector <2 x double> %wide.load178, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  %364 = call fast <2 x double> @llvm.exp.v2f64(<2 x double> %reverse177)
  %365 = call fast <2 x double> @llvm.exp.v2f64(<2 x double> %reverse179)
  %366 = fmul fast <2 x double> %364, %broadcast.splat181
  %367 = fmul fast <2 x double> %365, %broadcast.splat183
  %368 = fsub fast <2 x double> %vec.phi173, %366
  %369 = fsub fast <2 x double> %vec.phi174, %367
  %370 = add nuw nsw i64 %offset.idx175, 1
  %371 = getelementptr inbounds double, double* %"'ipc116.i", i64 %offset.idx175
  %372 = bitcast double* %371 to <2 x double>*
  %wide.load184 = load <2 x double>, <2 x double>* %372, align 8, !noalias !75
  %reverse185 = shufflevector <2 x double> %wide.load184, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  %373 = getelementptr inbounds double, double* %118, i64 %370
  %374 = bitcast double* %373 to <2 x double>*
  %wide.load186 = load <2 x double>, <2 x double>* %374, align 8, !noalias !75
  %reverse187 = shufflevector <2 x double> %wide.load186, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  %375 = fadd fast <2 x double> %366, %reverse185
  %376 = fadd fast <2 x double> %367, %reverse187
  %reverse188 = shufflevector <2 x double> %375, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  %377 = bitcast double* %371 to <2 x double>*
  store <2 x double> %reverse188, <2 x double>* %377, align 8, !noalias !75
  %reverse189 = shufflevector <2 x double> %376, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  %378 = bitcast double* %373 to <2 x double>*
  store <2 x double> %reverse189, <2 x double>* %378, align 8, !noalias !75
  %index.next169 = add i64 %index168, 4
  %379 = icmp eq i64 %index.next169, %n.vec167
  br i1 %379, label %middle.block161, label %vector.body163, !llvm.loop !110

middle.block161:                                  ; preds = %vector.body163
  %bin.rdx190 = fadd fast <2 x double> %369, %368
  %380 = call fast double @llvm.vector.reduce.fadd.v2f64(double -0.000000e+00, <2 x double> %bin.rdx190)
  br i1 %cmp.n172, label %invertfor.body.preheader.i.loopexit.i, label %invertfor.body.for.body_crit_edge.i.i.preheader

invertfor.body.for.body_crit_edge.i.i.preheader:  ; preds = %mergeinvertfor.body.for.body_crit_edge.i_log_sum_exp.exit.loopexit.i, %middle.block161
  %"m.0.lcssa.i.i'de.2.i.ph" = phi double [ 1.000000e+00, %mergeinvertfor.body.for.body_crit_edge.i_log_sum_exp.exit.loopexit.i ], [ %380, %middle.block161 ]
  %"iv19'ac.2.i.ph" = phi i64 [ %_unwrap149.i, %mergeinvertfor.body.for.body_crit_edge.i_log_sum_exp.exit.loopexit.i ], [ %ind.end171, %middle.block161 ]
  br label %invertfor.body.for.body_crit_edge.i.i

invertlog_sum_exp.exit.i:                         ; preds = %invertfor.cond19.preheader.i, %invertlog_sum_exp.exit.preheader.i
  %"add.i'de.3.i" = phi double [ %"add.i'de.0.i", %invertfor.cond19.preheader.i ], [ 0.000000e+00, %invertlog_sum_exp.exit.preheader.i ]
  %"add.i136'de.3.i" = phi double [ %"add.i136'de.0.i", %invertfor.cond19.preheader.i ], [ 0.000000e+00, %invertlog_sum_exp.exit.preheader.i ]
  %"'de33.0.i" = phi double [ %"'de32.0.i", %invertfor.cond19.preheader.i ], [ 0.000000e+00, %invertlog_sum_exp.exit.preheader.i ]
  %"'de31.3.i" = phi double [ %322, %invertfor.cond19.preheader.i ], [ 0.000000e+00, %invertlog_sum_exp.exit.preheader.i ]
  %"iv3'ac.0.in.i" = phi i64 [ %"iv3'ac.0.i", %invertfor.cond19.preheader.i ], [ %conv17.i, %invertlog_sum_exp.exit.preheader.i ]
  %"iv3'ac.0.i" = add nsw i64 %"iv3'ac.0.in.i", -1
  %381 = getelementptr inbounds double, double* %sub.i135_malloccache.i, i64 %"iv3'ac.0.i"
  %382 = load double, double* %381, align 8, !noalias !75, !invariant.group !93
  %383 = tail call double @llvm.exp.f64(double %382) #14
  %add.i136_unwrap.i = fadd double %383, 0.000000e+00
  %384 = getelementptr inbounds double, double* %"add.i!manual_lcssa154_malloccache.i", i64 %"iv3'ac.0.i"
  %385 = load double, double* %384, align 8, !noalias !75, !invariant.group !95
  %spec.select.i = select i1 %exitcond.not.i99137.i, double %add.i136_unwrap.i, double %385
  %spec.select.i.op = fdiv fast double 1.000000e+00, %spec.select.i
  %386 = select i1 %cmp37.i.i, double %spec.select.i.op, double 0x7FF0000000000000
  %387 = fadd fast double %386, %"add.i'de.3.i"
  %388 = select fast i1 %cmp13.i.i.i, double %387, double %"add.i'de.3.i"
  %389 = fadd fast double %386, %"add.i136'de.3.i"
  %390 = select fast i1 %exitcond.not.i99137.i, double %389, double %"add.i136'de.3.i"
  br i1 %cmp37.i.i, label %staging.i, label %invertarr_max.exit.i.i

staging.i:                                        ; preds = %invertlog_sum_exp.exit.i
  br i1 %exitcond.not.i99137.i, label %invertfor.body.preheader.i.i, label %mergeinvertfor.body.for.body_crit_edge.i_log_sum_exp.exit.loopexit.i

diffeec_main_term.exit:                           ; preds = %invertfor.body.i.loopexit.us.i, %invertpreprocess_qs.exit.i
  tail call void @free(i8* nonnull %"call15'mi.i") #14, !noalias !75
  tail call void @free(i8* nonnull %"call12'mi.i") #14, !noalias !75
  tail call void @free(i8* nonnull %"call9'mi.i") #14, !noalias !75
  tail call void @free(i8* nonnull %"call6'mi.i") #14, !noalias !75
  tail call void @free(i8* nonnull %"call'mi.i") #14, !noalias !75
  tail call void @free(i8* %call.i) #14, !noalias !75
  ret void
}
