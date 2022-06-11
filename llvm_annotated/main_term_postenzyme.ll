; ModuleID = '<stdin>'
source_filename = "-"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

; Function Attrs: norecurse nounwind readonly ssp uwtable
define dso_local double @arr_max(i32 %n, double* nocapture readonly %x) local_unnamed_addr #0 {
entry:
  %0 = load double, double* %x, align 8, !tbaa !3
  %cmp13 = icmp sgt i32 %n, 1
  br i1 %cmp13, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.preheader
  %indvars.iv = phi i64 [ 1, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %m.015 = phi double [ %0, %for.body.preheader ], [ %m.1, %for.body ]
  %arrayidx1 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  %1 = load double, double* %arrayidx1, align 8, !tbaa !3
  %cmp2 = fcmp olt double %m.015, %1
  %m.1 = select i1 %cmp2, double %1, double %m.015
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !7

for.end:                                          ; preds = %for.body, %entry
  %m.0.lcssa = phi double [ %0, %entry ], [ %m.1, %for.body ]
  ret double %m.0.lcssa
}

; Function Attrs: norecurse nounwind readonly ssp uwtable
define dso_local double @sqnorm(i32 %n, double* nocapture readonly %x) local_unnamed_addr #0 {
entry:
  %0 = load double, double* %x, align 8, !tbaa !3
  %mul = fmul double %0, %0
  %cmp15 = icmp sgt i32 %n, 1
  br i1 %cmp15, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.preheader
  %indvars.iv = phi i64 [ 1, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %res.017 = phi double [ %mul, %for.body.preheader ], [ %add, %for.body ]
  %arrayidx2 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  %1 = load double, double* %arrayidx2, align 8, !tbaa !3
  %mul5 = fmul double %1, %1
  %add = fadd double %res.017, %mul5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !10

for.end:                                          ; preds = %for.body, %entry
  %res.0.lcssa = phi double [ %mul, %entry ], [ %add, %for.body ]
  ret double %res.0.lcssa
}

; Function Attrs: nofree norecurse nounwind ssp uwtable
define dso_local void @subtract(i32 %d, double* nocapture readonly %x, double* nocapture readonly %y, double* nocapture %out) local_unnamed_addr #1 {
entry:
  %cmp10 = icmp sgt i32 %d, 0
  br i1 %cmp10, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %d to i64
  %min.iters.check = icmp ult i32 %d, 4
  br i1 %min.iters.check, label %for.body.preheader15, label %vector.memcheck

vector.memcheck:                                  ; preds = %for.body.preheader
  %scevgep = getelementptr double, double* %out, i64 %wide.trip.count
  %scevgep4 = getelementptr double, double* %x, i64 %wide.trip.count
  %scevgep7 = getelementptr double, double* %y, i64 %wide.trip.count
  %bound0 = icmp ugt double* %scevgep4, %out
  %bound1 = icmp ugt double* %scevgep, %x
  %found.conflict = and i1 %bound0, %bound1
  %bound09 = icmp ugt double* %scevgep7, %out
  %bound110 = icmp ugt double* %scevgep, %y
  %found.conflict11 = and i1 %bound09, %bound110
  %conflict.rdx = or i1 %found.conflict, %found.conflict11
  br i1 %conflict.rdx, label %for.body.preheader15, label %vector.ph

vector.ph:                                        ; preds = %vector.memcheck
  %n.vec = and i64 %wide.trip.count, 4294967292
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds double, double* %x, i64 %index
  %1 = bitcast double* %0 to <2 x double>*
  %wide.load = load <2 x double>, <2 x double>* %1, align 8, !tbaa !3, !alias.scope !11
  %2 = getelementptr inbounds double, double* %0, i64 2
  %3 = bitcast double* %2 to <2 x double>*
  %wide.load12 = load <2 x double>, <2 x double>* %3, align 8, !tbaa !3, !alias.scope !11
  %4 = getelementptr inbounds double, double* %y, i64 %index
  %5 = bitcast double* %4 to <2 x double>*
  %wide.load13 = load <2 x double>, <2 x double>* %5, align 8, !tbaa !3, !alias.scope !14
  %6 = getelementptr inbounds double, double* %4, i64 2
  %7 = bitcast double* %6 to <2 x double>*
  %wide.load14 = load <2 x double>, <2 x double>* %7, align 8, !tbaa !3, !alias.scope !14
  %8 = fsub <2 x double> %wide.load, %wide.load13
  %9 = fsub <2 x double> %wide.load12, %wide.load14
  %10 = getelementptr inbounds double, double* %out, i64 %index
  %11 = bitcast double* %10 to <2 x double>*
  store <2 x double> %8, <2 x double>* %11, align 8, !tbaa !3, !alias.scope !16, !noalias !18
  %12 = getelementptr inbounds double, double* %10, i64 2
  %13 = bitcast double* %12 to <2 x double>*
  store <2 x double> %9, <2 x double>* %13, align 8, !tbaa !3, !alias.scope !16, !noalias !18
  %index.next = add i64 %index, 4
  %14 = icmp eq i64 %index.next, %n.vec
  br i1 %14, label %middle.block, label %vector.body, !llvm.loop !19

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %n.vec, %wide.trip.count
  br i1 %cmp.n, label %for.end, label %for.body.preheader15

for.body.preheader15:                             ; preds = %vector.memcheck, %for.body.preheader, %middle.block
  %indvars.iv.ph = phi i64 [ 0, %vector.memcheck ], [ 0, %for.body.preheader ], [ %n.vec, %middle.block ]
  br label %for.body

for.body:                                         ; preds = %for.body.preheader15, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ %indvars.iv.ph, %for.body.preheader15 ]
  %arrayidx = getelementptr inbounds double, double* %x, i64 %indvars.iv
  %15 = load double, double* %arrayidx, align 8, !tbaa !3
  %arrayidx2 = getelementptr inbounds double, double* %y, i64 %indvars.iv
  %16 = load double, double* %arrayidx2, align 8, !tbaa !3
  %sub = fsub double %15, %16
  %arrayidx4 = getelementptr inbounds double, double* %out, i64 %indvars.iv
  store double %sub, double* %arrayidx4, align 8, !tbaa !3
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !21

for.end:                                          ; preds = %for.body, %middle.block, %entry
  ret void
}

; Function Attrs: nounwind readonly ssp uwtable
define dso_local double @log_sum_exp(i32 %n, double* nocapture readonly %x) local_unnamed_addr #2 {
entry:
  %0 = load double, double* %x, align 8, !tbaa !3
  %cmp13.i = icmp sgt i32 %n, 1
  br i1 %cmp13.i, label %for.body.preheader.i, label %arr_max.exit

for.body.preheader.i:                             ; preds = %entry
  %wide.trip.count.i = zext i32 %n to i64
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %for.body.preheader.i
  %indvars.iv.i = phi i64 [ 1, %for.body.preheader.i ], [ %indvars.iv.next.i, %for.body.i ]
  %m.015.i = phi double [ %0, %for.body.preheader.i ], [ %m.1.i, %for.body.i ]
  %arrayidx1.i = getelementptr inbounds double, double* %x, i64 %indvars.iv.i
  %1 = load double, double* %arrayidx1.i, align 8, !tbaa !3
  %cmp2.i = fcmp olt double %m.015.i, %1
  %m.1.i = select i1 %cmp2.i, double %1, double %m.015.i
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
  %exitcond.not.i = icmp eq i64 %indvars.iv.next.i, %wide.trip.count.i
  br i1 %exitcond.not.i, label %arr_max.exit, label %for.body.i, !llvm.loop !7

arr_max.exit:                                     ; preds = %for.body.i, %entry
  %m.0.lcssa.i = phi double [ %0, %entry ], [ %m.1.i, %for.body.i ]
  %cmp11 = icmp sgt i32 %n, 0
  br i1 %cmp11, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %arr_max.exit
  %wide.trip.count = zext i32 %n to i64
  %sub14 = fsub double %0, %m.0.lcssa.i
  %2 = tail call double @llvm.exp.f64(double %sub14)
  %add15 = fadd double %2, 0.000000e+00
  %exitcond.not16 = icmp eq i32 %n, 1
  br i1 %exitcond.not16, label %for.end, label %for.body.for.body_crit_edge, !llvm.loop !22

for.body.for.body_crit_edge:                      ; preds = %for.body.preheader, %for.body.for.body_crit_edge
  %indvars.iv.next18 = phi i64 [ %indvars.iv.next, %for.body.for.body_crit_edge ], [ 1, %for.body.preheader ]
  %add17 = phi double [ %add, %for.body.for.body_crit_edge ], [ %add15, %for.body.preheader ]
  %arrayidx.phi.trans.insert = getelementptr inbounds double, double* %x, i64 %indvars.iv.next18
  %.pre = load double, double* %arrayidx.phi.trans.insert, align 8, !tbaa !3
  %sub = fsub double %.pre, %m.0.lcssa.i
  %3 = tail call double @llvm.exp.f64(double %sub)
  %add = fadd double %add17, %3
  %indvars.iv.next = add nuw nsw i64 %indvars.iv.next18, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.end, label %for.body.for.body_crit_edge, !llvm.loop !22

for.end:                                          ; preds = %for.body.for.body_crit_edge, %for.body.preheader, %arr_max.exit
  %semx.0.lcssa = phi double [ 0.000000e+00, %arr_max.exit ], [ %add15, %for.body.preheader ], [ %add, %for.body.for.body_crit_edge ]
  %4 = tail call double @llvm.log.f64(double %semx.0.lcssa)
  %add1 = fadd double %m.0.lcssa.i, %4
  ret double %add1
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare double @llvm.exp.f64(double) #3

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare double @llvm.log.f64(double) #3

; Function Attrs: nofree nounwind ssp uwtable
define dso_local void @preprocess_qs(i32 %d, i32 %k, double* nocapture readonly %Qs, double* nocapture %sum_qs, double* nocapture %Qdiags) local_unnamed_addr #4 {
entry:
  %sum_qs5 = bitcast double* %sum_qs to i8*
  %cmp37 = icmp sgt i32 %k, 0
  br i1 %cmp37, label %for.body.lr.ph, label %for.end17

for.body.lr.ph:                                   ; preds = %entry
  %cmp235 = icmp sgt i32 %d, 0
  %wide.trip.count44 = zext i32 %k to i64
  %wide.trip.count = zext i32 %d to i64
  br i1 %cmp235, label %for.body.lr.ph.split.us, label %for.body.preheader

for.body.preheader:                               ; preds = %for.body.lr.ph
  %0 = shl nuw nsw i64 %wide.trip.count44, 3
  call void @llvm.memset.p0i8.i64(i8* align 8 %sum_qs5, i8 0, i64 %0, i1 false)
  br label %for.end17

for.body.lr.ph.split.us:                          ; preds = %for.body.lr.ph
  %exitcond.not49 = icmp eq i32 %d, 1
  br i1 %exitcond.not49, label %for.body.us.us.preheader, label %for.body.us

for.body.us.us.preheader:                         ; preds = %for.body.lr.ph.split.us
  %min.iters.check = icmp ult i32 %k, 2
  %1 = add nsw i64 %wide.trip.count44, -1
  %2 = icmp ugt i64 %1, 2147483647
  %or.cond = or i1 %min.iters.check, %2
  br i1 %or.cond, label %for.body.us.us.preheader23, label %vector.memcheck

vector.memcheck:                                  ; preds = %for.body.us.us.preheader
  %scevgep = getelementptr double, double* %sum_qs, i64 %wide.trip.count44
  %scevgep10 = getelementptr double, double* %Qdiags, i64 %wide.trip.count44
  %scevgep13 = getelementptr double, double* %Qs, i64 %wide.trip.count44
  %bound0 = icmp ugt double* %scevgep10, %sum_qs
  %bound1 = icmp ugt double* %scevgep, %Qdiags
  %found.conflict = and i1 %bound0, %bound1
  %bound015 = icmp ugt double* %scevgep13, %sum_qs
  %bound116 = icmp ugt double* %scevgep, %Qs
  %found.conflict17 = and i1 %bound015, %bound116
  %conflict.rdx = or i1 %found.conflict, %found.conflict17
  %bound018 = icmp ugt double* %scevgep13, %Qdiags
  %bound119 = icmp ugt double* %scevgep10, %Qs
  %found.conflict20 = and i1 %bound018, %bound119
  %conflict.rdx21 = or i1 %conflict.rdx, %found.conflict20
  br i1 %conflict.rdx21, label %for.body.us.us.preheader23, label %vector.ph

vector.ph:                                        ; preds = %vector.memcheck
  %n.vec = and i64 %wide.trip.count44, 4294967294
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %3 = getelementptr inbounds double, double* %sum_qs, i64 %index
  %4 = bitcast double* %3 to <2 x double>*
  store <2 x double> zeroinitializer, <2 x double>* %4, align 8, !tbaa !3, !alias.scope !23, !noalias !26
  %5 = shl i64 %index, 32
  %6 = ashr exact i64 %5, 32
  %7 = getelementptr inbounds double, double* %Qs, i64 %6
  %8 = bitcast double* %7 to <2 x double>*
  %wide.load = load <2 x double>, <2 x double>* %8, align 8, !tbaa !3, !alias.scope !29
  %9 = fadd <2 x double> %wide.load, zeroinitializer
  %10 = bitcast double* %3 to <2 x double>*
  store <2 x double> %9, <2 x double>* %10, align 8, !tbaa !3, !alias.scope !23, !noalias !26
  %11 = call <2 x double> @llvm.exp.v2f64(<2 x double> %wide.load)
  %12 = getelementptr inbounds double, double* %Qdiags, i64 %6
  %13 = bitcast double* %12 to <2 x double>*
  store <2 x double> %11, <2 x double>* %13, align 8, !tbaa !3, !alias.scope !30, !noalias !29
  %index.next = add i64 %index, 2
  %14 = icmp eq i64 %index.next, %n.vec
  br i1 %14, label %middle.block, label %vector.body, !llvm.loop !31

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %n.vec, %wide.trip.count44
  br i1 %cmp.n, label %for.end17, label %for.body.us.us.preheader23

for.body.us.us.preheader23:                       ; preds = %vector.memcheck, %for.body.us.us.preheader, %middle.block
  %indvars.iv42.us.us.ph = phi i64 [ 0, %vector.memcheck ], [ 0, %for.body.us.us.preheader ], [ %n.vec, %middle.block ]
  br label %for.body.us.us

for.body.us.us:                                   ; preds = %for.body.us.us.preheader23, %for.body.us.us
  %indvars.iv42.us.us = phi i64 [ %indvars.iv.next43.us.us, %for.body.us.us ], [ %indvars.iv42.us.us.ph, %for.body.us.us.preheader23 ]
  %arrayidx.us.us = getelementptr inbounds double, double* %sum_qs, i64 %indvars.iv42.us.us
  store double 0.000000e+00, double* %arrayidx.us.us, align 8, !tbaa !3
  %sext = shl i64 %indvars.iv42.us.us, 32
  %15 = ashr exact i64 %sext, 32
  %arrayidx546.us.us = getelementptr inbounds double, double* %Qs, i64 %15
  %16 = load double, double* %arrayidx546.us.us, align 8, !tbaa !3
  %add847.us.us = fadd double %16, 0.000000e+00
  store double %add847.us.us, double* %arrayidx.us.us, align 8, !tbaa !3
  %17 = tail call double @llvm.exp.f64(double %16)
  %arrayidx1448.us.us = getelementptr inbounds double, double* %Qdiags, i64 %15
  store double %17, double* %arrayidx1448.us.us, align 8, !tbaa !3
  %indvars.iv.next43.us.us = add nuw nsw i64 %indvars.iv42.us.us, 1
  %exitcond45.not.us.us = icmp eq i64 %indvars.iv.next43.us.us, %wide.trip.count44
  br i1 %exitcond45.not.us.us, label %for.end17, label %for.body.us.us, !llvm.loop !32

for.body.us:                                      ; preds = %for.body.lr.ph.split.us, %for.inc15.loopexit.us
  %indvars.iv42.us = phi i64 [ %indvars.iv.next43.us, %for.inc15.loopexit.us ], [ 0, %for.body.lr.ph.split.us ]
  %arrayidx.us = getelementptr inbounds double, double* %sum_qs, i64 %indvars.iv42.us
  store double 0.000000e+00, double* %arrayidx.us, align 8, !tbaa !3
  %18 = trunc i64 %indvars.iv42.us to i32
  %mul.us = mul nsw i32 %18, %d
  %19 = sext i32 %mul.us to i64
  %arrayidx546.us = getelementptr inbounds double, double* %Qs, i64 %19
  %20 = load double, double* %arrayidx546.us, align 8, !tbaa !3
  %add847.us = fadd double %20, 0.000000e+00
  store double %add847.us, double* %arrayidx.us, align 8, !tbaa !3
  %21 = tail call double @llvm.exp.f64(double %20)
  %arrayidx1448.us = getelementptr inbounds double, double* %Qdiags, i64 %19
  store double %21, double* %arrayidx1448.us, align 8, !tbaa !3
  br label %for.body3.for.body3_crit_edge.us

for.body3.for.body3_crit_edge.us:                 ; preds = %for.body.us, %for.body3.for.body3_crit_edge.us
  %indvars.iv.next50.us = phi i64 [ %indvars.iv.next.us, %for.body3.for.body3_crit_edge.us ], [ 1, %for.body.us ]
  %.pre.us = load double, double* %arrayidx.us, align 8, !tbaa !3
  %22 = add nsw i64 %indvars.iv.next50.us, %19
  %arrayidx5.us = getelementptr inbounds double, double* %Qs, i64 %22
  %23 = load double, double* %arrayidx5.us, align 8, !tbaa !3
  %add8.us = fadd double %.pre.us, %23
  store double %add8.us, double* %arrayidx.us, align 8, !tbaa !3
  %24 = tail call double @llvm.exp.f64(double %23)
  %arrayidx14.us = getelementptr inbounds double, double* %Qdiags, i64 %22
  store double %24, double* %arrayidx14.us, align 8, !tbaa !3
  %indvars.iv.next.us = add nuw nsw i64 %indvars.iv.next50.us, 1
  %exitcond.not.us = icmp eq i64 %indvars.iv.next.us, %wide.trip.count
  br i1 %exitcond.not.us, label %for.inc15.loopexit.us, label %for.body3.for.body3_crit_edge.us, !llvm.loop !33

for.inc15.loopexit.us:                            ; preds = %for.body3.for.body3_crit_edge.us
  %indvars.iv.next43.us = add nuw nsw i64 %indvars.iv42.us, 1
  %exitcond45.not.us = icmp eq i64 %indvars.iv.next43.us, %wide.trip.count44
  br i1 %exitcond45.not.us, label %for.end17, label %for.body.us, !llvm.loop !34

for.end17:                                        ; preds = %for.inc15.loopexit.us, %for.body.us.us, %middle.block, %for.body.preheader, %entry
  ret void
}

; Function Attrs: nofree norecurse nounwind ssp uwtable
define dso_local void @cQtimesx(i32 %d, double* nocapture readonly %Qdiag, double* nocapture readonly %ltri, double* nocapture readonly %x, double* nocapture %out) local_unnamed_addr #1 {
entry:
  %cmp59 = icmp sgt i32 %d, 0
  br i1 %cmp59, label %for.body.preheader, label %for.end30

for.body.preheader:                               ; preds = %entry
  %wide.trip.count71 = zext i32 %d to i64
  %min.iters.check = icmp ult i32 %d, 4
  br i1 %min.iters.check, label %for.body.preheader56, label %vector.memcheck

vector.memcheck:                                  ; preds = %for.body.preheader
  %scevgep = getelementptr double, double* %out, i64 %wide.trip.count71
  %scevgep4 = getelementptr double, double* %Qdiag, i64 %wide.trip.count71
  %scevgep7 = getelementptr double, double* %x, i64 %wide.trip.count71
  %bound0 = icmp ugt double* %scevgep4, %out
  %bound1 = icmp ugt double* %scevgep, %Qdiag
  %found.conflict = and i1 %bound0, %bound1
  %bound09 = icmp ugt double* %scevgep7, %out
  %bound110 = icmp ugt double* %scevgep, %x
  %found.conflict11 = and i1 %bound09, %bound110
  %conflict.rdx = or i1 %found.conflict, %found.conflict11
  br i1 %conflict.rdx, label %for.body.preheader56, label %vector.ph

vector.ph:                                        ; preds = %vector.memcheck
  %n.vec = and i64 %wide.trip.count71, 4294967292
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds double, double* %Qdiag, i64 %index
  %1 = bitcast double* %0 to <2 x double>*
  %wide.load = load <2 x double>, <2 x double>* %1, align 8, !tbaa !3, !alias.scope !35
  %2 = getelementptr inbounds double, double* %0, i64 2
  %3 = bitcast double* %2 to <2 x double>*
  %wide.load12 = load <2 x double>, <2 x double>* %3, align 8, !tbaa !3, !alias.scope !35
  %4 = getelementptr inbounds double, double* %x, i64 %index
  %5 = bitcast double* %4 to <2 x double>*
  %wide.load13 = load <2 x double>, <2 x double>* %5, align 8, !tbaa !3, !alias.scope !38
  %6 = getelementptr inbounds double, double* %4, i64 2
  %7 = bitcast double* %6 to <2 x double>*
  %wide.load14 = load <2 x double>, <2 x double>* %7, align 8, !tbaa !3, !alias.scope !38
  %8 = fmul <2 x double> %wide.load, %wide.load13
  %9 = fmul <2 x double> %wide.load12, %wide.load14
  %10 = getelementptr inbounds double, double* %out, i64 %index
  %11 = bitcast double* %10 to <2 x double>*
  store <2 x double> %8, <2 x double>* %11, align 8, !tbaa !3, !alias.scope !40, !noalias !42
  %12 = getelementptr inbounds double, double* %10, i64 2
  %13 = bitcast double* %12 to <2 x double>*
  store <2 x double> %9, <2 x double>* %13, align 8, !tbaa !3, !alias.scope !40, !noalias !42
  %index.next = add i64 %index, 4
  %14 = icmp eq i64 %index.next, %n.vec
  br i1 %14, label %middle.block, label %vector.body, !llvm.loop !43

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %n.vec, %wide.trip.count71
  br i1 %cmp.n, label %for.body7.lr.ph, label %for.body.preheader56

for.body.preheader56:                             ; preds = %vector.memcheck, %for.body.preheader, %middle.block
  %indvars.iv69.ph = phi i64 [ 0, %vector.memcheck ], [ 0, %for.body.preheader ], [ %n.vec, %middle.block ]
  br label %for.body

for.body7.lr.ph:                                  ; preds = %for.body, %middle.block
  %mul8 = shl nuw nsw i32 %d, 1
  %scevgep24 = getelementptr double, double* %out, i64 %wide.trip.count71
  br label %for.body7

for.body:                                         ; preds = %for.body.preheader56, %for.body
  %indvars.iv69 = phi i64 [ %indvars.iv.next70, %for.body ], [ %indvars.iv69.ph, %for.body.preheader56 ]
  %arrayidx = getelementptr inbounds double, double* %Qdiag, i64 %indvars.iv69
  %15 = load double, double* %arrayidx, align 8, !tbaa !3
  %arrayidx2 = getelementptr inbounds double, double* %x, i64 %indvars.iv69
  %16 = load double, double* %arrayidx2, align 8, !tbaa !3
  %mul = fmul double %15, %16
  %arrayidx4 = getelementptr inbounds double, double* %out, i64 %indvars.iv69
  store double %mul, double* %arrayidx4, align 8, !tbaa !3
  %indvars.iv.next70 = add nuw nsw i64 %indvars.iv69, 1
  %exitcond72.not = icmp eq i64 %indvars.iv.next70, %wide.trip.count71
  br i1 %exitcond72.not, label %for.body7.lr.ph, label %for.body, !llvm.loop !44

for.cond5.loopexit:                               ; preds = %for.body13, %middle.block15, %for.body7
  %indvars.iv.next62 = add nuw nsw i64 %indvars.iv61, 1
  %exitcond68.not = icmp eq i64 %indvars.iv.next66, %wide.trip.count71
  br i1 %exitcond68.not, label %for.end30, label %for.body7, !llvm.loop !45

for.body7:                                        ; preds = %for.cond5.loopexit, %for.body7.lr.ph
  %indvars.iv65 = phi i64 [ 0, %for.body7.lr.ph ], [ %indvars.iv.next66, %for.cond5.loopexit ]
  %indvars.iv61 = phi i64 [ 1, %for.body7.lr.ph ], [ %indvars.iv.next62, %for.cond5.loopexit ]
  %i.158 = phi i32 [ 0, %for.body7.lr.ph ], [ %add, %for.cond5.loopexit ]
  %17 = add nuw i64 %indvars.iv65, 1
  %scevgep22 = getelementptr double, double* %out, i64 %17
  %scevgep2223 = bitcast double* %scevgep22 to i8*
  %18 = xor i64 %indvars.iv65, -1
  %19 = add nsw i64 %18, %wide.trip.count71
  %scevgep28 = getelementptr double, double* %ltri, i64 %19
  %scevgep31 = getelementptr double, double* %x, i64 %indvars.iv65
  %scevgep3132 = bitcast double* %scevgep31 to i8*
  %uglygep = getelementptr i8, i8* %scevgep3132, i64 1
  %20 = xor i64 %indvars.iv65, -1
  %21 = add nsw i64 %20, %wide.trip.count71
  %indvars.iv.next66 = add nuw nsw i64 %indvars.iv65, 1
  %add = add nuw nsw i32 %i.158, 1
  %cmp1254 = icmp ult i64 %indvars.iv.next66, %wide.trip.count71
  br i1 %cmp1254, label %for.body13.lr.ph, label %for.cond5.loopexit

for.body13.lr.ph:                                 ; preds = %for.body7
  %22 = xor i32 %i.158, -1
  %sub9 = add i32 %mul8, %22
  %23 = trunc i64 %indvars.iv65 to i32
  %mul10 = mul nsw i32 %sub9, %23
  %div = sdiv i32 %mul10, 2
  %arrayidx19 = getelementptr inbounds double, double* %x, i64 %indvars.iv65
  %24 = sext i32 %div to i64
  %min.iters.check18 = icmp ult i64 %21, 4
  br i1 %min.iters.check18, label %for.body13.preheader, label %vector.memcheck20

vector.memcheck20:                                ; preds = %for.body13.lr.ph
  %scevgep26 = getelementptr double, double* %ltri, i64 %24
  %scevgep29 = getelementptr double, double* %scevgep28, i64 %24
  %bound033 = icmp ult double* %scevgep22, %scevgep29
  %bound134 = icmp ult double* %scevgep26, %scevgep24
  %found.conflict35 = and i1 %bound033, %bound134
  %bound036 = icmp ugt i8* %uglygep, %scevgep2223
  %bound137 = icmp ult double* %arrayidx19, %scevgep24
  %found.conflict38 = and i1 %bound036, %bound137
  %conflict.rdx39 = or i1 %found.conflict35, %found.conflict38
  br i1 %conflict.rdx39, label %for.body13.preheader, label %vector.ph21

vector.ph21:                                      ; preds = %vector.memcheck20
  %n.vec42 = and i64 %21, -4
  %ind.end = add i64 %indvars.iv61, %n.vec42
  %ind.end47 = add i64 %n.vec42, %24
  %25 = load double, double* %arrayidx19, align 8, !tbaa !3, !alias.scope !46
  %broadcast.splatinsert = insertelement <2 x double> poison, double %25, i32 0
  %broadcast.splat = shufflevector <2 x double> %broadcast.splatinsert, <2 x double> poison, <2 x i32> zeroinitializer
  %broadcast.splatinsert54 = insertelement <2 x double> poison, double %25, i32 0
  %broadcast.splat55 = shufflevector <2 x double> %broadcast.splatinsert54, <2 x double> poison, <2 x i32> zeroinitializer
  br label %vector.body17

vector.body17:                                    ; preds = %vector.body17, %vector.ph21
  %index43 = phi i64 [ 0, %vector.ph21 ], [ %index.next44, %vector.body17 ]
  %offset.idx = add i64 %indvars.iv61, %index43
  %offset.idx49 = add i64 %index43, %24
  %26 = getelementptr inbounds double, double* %out, i64 %offset.idx
  %27 = bitcast double* %26 to <2 x double>*
  %wide.load50 = load <2 x double>, <2 x double>* %27, align 8, !tbaa !3, !alias.scope !49, !noalias !51
  %28 = getelementptr inbounds double, double* %26, i64 2
  %29 = bitcast double* %28 to <2 x double>*
  %wide.load51 = load <2 x double>, <2 x double>* %29, align 8, !tbaa !3, !alias.scope !49, !noalias !51
  %30 = getelementptr inbounds double, double* %ltri, i64 %offset.idx49
  %31 = bitcast double* %30 to <2 x double>*
  %wide.load52 = load <2 x double>, <2 x double>* %31, align 8, !tbaa !3, !alias.scope !53
  %32 = getelementptr inbounds double, double* %30, i64 2
  %33 = bitcast double* %32 to <2 x double>*
  %wide.load53 = load <2 x double>, <2 x double>* %33, align 8, !tbaa !3, !alias.scope !53
  %34 = fmul <2 x double> %wide.load52, %broadcast.splat
  %35 = fmul <2 x double> %wide.load53, %broadcast.splat55
  %36 = fadd <2 x double> %wide.load50, %34
  %37 = fadd <2 x double> %wide.load51, %35
  %38 = bitcast double* %26 to <2 x double>*
  store <2 x double> %36, <2 x double>* %38, align 8, !tbaa !3, !alias.scope !49, !noalias !51
  %39 = bitcast double* %28 to <2 x double>*
  store <2 x double> %37, <2 x double>* %39, align 8, !tbaa !3, !alias.scope !49, !noalias !51
  %index.next44 = add i64 %index43, 4
  %40 = icmp eq i64 %index.next44, %n.vec42
  br i1 %40, label %middle.block15, label %vector.body17, !llvm.loop !54

middle.block15:                                   ; preds = %vector.body17
  %cmp.n48 = icmp eq i64 %21, %n.vec42
  br i1 %cmp.n48, label %for.cond5.loopexit, label %for.body13.preheader

for.body13.preheader:                             ; preds = %vector.memcheck20, %for.body13.lr.ph, %middle.block15
  %indvars.iv63.ph = phi i64 [ %indvars.iv61, %vector.memcheck20 ], [ %indvars.iv61, %for.body13.lr.ph ], [ %ind.end, %middle.block15 ]
  %indvars.iv.ph = phi i64 [ %24, %vector.memcheck20 ], [ %24, %for.body13.lr.ph ], [ %ind.end47, %middle.block15 ]
  br label %for.body13

for.body13:                                       ; preds = %for.body13.preheader, %for.body13
  %indvars.iv63 = phi i64 [ %indvars.iv.next64, %for.body13 ], [ %indvars.iv63.ph, %for.body13.preheader ]
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body13 ], [ %indvars.iv.ph, %for.body13.preheader ]
  %arrayidx15 = getelementptr inbounds double, double* %out, i64 %indvars.iv63
  %41 = load double, double* %arrayidx15, align 8, !tbaa !3
  %arrayidx17 = getelementptr inbounds double, double* %ltri, i64 %indvars.iv
  %42 = load double, double* %arrayidx17, align 8, !tbaa !3
  %43 = load double, double* %arrayidx19, align 8, !tbaa !3
  %mul20 = fmul double %42, %43
  %add21 = fadd double %41, %mul20
  store double %add21, double* %arrayidx15, align 8, !tbaa !3
  %indvars.iv.next = add nsw i64 %indvars.iv, 1
  %indvars.iv.next64 = add nuw nsw i64 %indvars.iv63, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next64, %wide.trip.count71
  br i1 %exitcond.not, label %for.cond5.loopexit, label %for.body13, !llvm.loop !55

for.end30:                                        ; preds = %for.cond5.loopexit, %entry
  ret void
}

; Function Attrs: nounwind ssp uwtable
define dso_local void @ec_main_term(i32 %d, i32 %k, i32 %n, double* noalias nocapture readonly %alphas, double* noalias nocapture readonly %means, double* noalias nocapture readonly %Qs, double* noalias nocapture readonly %Ls, double* noalias nocapture readonly %x, double* nocapture %err) local_unnamed_addr #5 {
entry:
  %sub = add nsw i32 %d, -1
  %mul = mul nsw i32 %sub, %d
  %div = sdiv i32 %mul, 2
  %conv = sext i32 %div to i64
  %mul1 = mul nsw i32 %k, %d
  %conv2 = sext i32 %mul1 to i64
  %mul3 = shl nsw i64 %conv2, 3
  %call = tail call i8* @malloc(i64 %mul3) #13
  %0 = bitcast i8* %call to double*
  %conv4 = sext i32 %k to i64
  %mul5 = shl nsw i64 %conv4, 3
  %call6 = tail call i8* @malloc(i64 %mul5) #13
  %1 = bitcast i8* %call6 to double*
  %conv7 = sext i32 %d to i64
  %mul8 = shl nsw i64 %conv7, 3
  %call9 = tail call i8* @malloc(i64 %mul8) #13
  %2 = bitcast i8* %call9 to double*
  %call12 = tail call i8* @malloc(i64 %mul8) #13
  %3 = bitcast i8* %call12 to double*
  %call15 = tail call i8* @malloc(i64 %mul5) #13
  %4 = bitcast i8* %call15 to double*
  %cmp37.i = icmp sgt i32 %k, 0
  br i1 %cmp37.i, label %for.body.lr.ph.i, label %preprocess_qs.exit

for.body.lr.ph.i:                                 ; preds = %entry
  %cmp235.i = icmp sgt i32 %d, 0
  %wide.trip.count44.i = zext i32 %k to i64
  %wide.trip.count.i = zext i32 %d to i64
  br i1 %cmp235.i, label %for.body.i.us, label %for.body.i.preheader

for.body.i.preheader:                             ; preds = %for.body.lr.ph.i
  %5 = shl nuw nsw i64 %wide.trip.count44.i, 3
  call void @llvm.memset.p0i8.i64(i8* align 8 %call6, i8 0, i64 %5, i1 false)
  br label %preprocess_qs.exit

for.body.i.us:                                    ; preds = %for.body.lr.ph.i, %for.inc15.i.loopexit.us
  %indvars.iv42.i.us = phi i64 [ %indvars.iv.next43.i.us, %for.inc15.i.loopexit.us ], [ 0, %for.body.lr.ph.i ]
  %arrayidx.i.us = getelementptr inbounds double, double* %1, i64 %indvars.iv42.i.us
  store double 0.000000e+00, double* %arrayidx.i.us, align 8, !tbaa !3
  %6 = trunc i64 %indvars.iv42.i.us to i32
  %mul.i.us = mul nsw i32 %6, %d
  %7 = sext i32 %mul.i.us to i64
  br label %for.body3.i.us

for.body3.i.us:                                   ; preds = %for.body3.i.us, %for.body.i.us
  %8 = phi double [ 0.000000e+00, %for.body.i.us ], [ %add8.i.us, %for.body3.i.us ]
  %indvars.iv.i.us = phi i64 [ 0, %for.body.i.us ], [ %indvars.iv.next.i.us, %for.body3.i.us ]
  %9 = add nsw i64 %indvars.iv.i.us, %7
  %arrayidx5.i.us = getelementptr inbounds double, double* %Qs, i64 %9
  %10 = load double, double* %arrayidx5.i.us, align 8, !tbaa !3
  %add8.i.us = fadd double %8, %10
  %11 = tail call double @llvm.exp.f64(double %10) #14
  %arrayidx14.i.us = getelementptr inbounds double, double* %0, i64 %9
  store double %11, double* %arrayidx14.i.us, align 8, !tbaa !3
  %indvars.iv.next.i.us = add nuw nsw i64 %indvars.iv.i.us, 1
  %exitcond.not.i.us = icmp eq i64 %indvars.iv.next.i.us, %wide.trip.count.i
  br i1 %exitcond.not.i.us, label %for.inc15.i.loopexit.us, label %for.body3.i.us, !llvm.loop !33

for.inc15.i.loopexit.us:                          ; preds = %for.body3.i.us
  store double %add8.i.us, double* %arrayidx.i.us, align 8, !tbaa !3
  %indvars.iv.next43.i.us = add nuw nsw i64 %indvars.iv42.i.us, 1
  %exitcond45.not.i.us = icmp eq i64 %indvars.iv.next43.i.us, %wide.trip.count44.i
  br i1 %exitcond45.not.i.us, label %preprocess_qs.exit, label %for.body.i.us, !llvm.loop !34

preprocess_qs.exit:                               ; preds = %for.inc15.i.loopexit.us, %for.body.i.preheader, %entry
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
  %min.iters.check31 = icmp ult i32 %d, 4
  %n.vec34 = and i64 %wide.trip.count.i119, 4294967292
  %cmp.n38 = icmp eq i64 %n.vec34, %wide.trip.count.i119
  %min.iters.check16 = icmp ult i32 %d, 4
  %n.vec19 = and i64 %wide.trip.count.i119, 4294967292
  %cmp.n23 = icmp eq i64 %n.vec19, %wide.trip.count.i119
  br label %for.cond19.preheader

for.cond19.preheader:                             ; preds = %log_sum_exp.exit, %for.cond19.preheader.lr.ph
  %12 = phi double [ undef, %for.cond19.preheader.lr.ph ], [ %77, %log_sum_exp.exit ]
  %13 = phi double [ undef, %for.cond19.preheader.lr.ph ], [ %78, %log_sum_exp.exit ]
  %slse.0143 = phi double [ 0.000000e+00, %for.cond19.preheader.lr.ph ], [ %add47, %log_sum_exp.exit ]
  %ix.0141 = phi i64 [ 0, %for.cond19.preheader.lr.ph ], [ %inc49, %log_sum_exp.exit ]
  br i1 %cmp37.i, label %for.body23.lr.ph, label %for.end

for.body23.lr.ph:                                 ; preds = %for.cond19.preheader
  %mul25 = mul nsw i64 %ix.0141, %conv7
  %arrayidx26 = getelementptr inbounds double, double* %x, i64 %mul25
  br i1 %cmp10.i, label %for.body23.us, label %for.body23.preheader

for.body23.preheader:                             ; preds = %for.body23.lr.ph
  %mul.i102 = fmul double %13, %13
  br label %for.body23

for.body23.us:                                    ; preds = %for.body23.lr.ph, %sqnorm.exit.us
  %ik.0133.us = phi i64 [ %inc.us, %sqnorm.exit.us ], [ 0, %for.body23.lr.ph ]
  %mul28.us = mul nsw i64 %ik.0133.us, %conv7
  %arrayidx29.us = getelementptr inbounds double, double* %means, i64 %mul28.us
  br i1 %min.iters.check31, label %for.body.i128.us.preheader, label %vector.body30

vector.body30:                                    ; preds = %for.body23.us, %vector.body30
  %index35 = phi i64 [ %index.next36, %vector.body30 ], [ 0, %for.body23.us ]
  %14 = getelementptr inbounds double, double* %arrayidx26, i64 %index35
  %15 = bitcast double* %14 to <2 x double>*
  %wide.load39 = load <2 x double>, <2 x double>* %15, align 8, !tbaa !3
  %16 = getelementptr inbounds double, double* %14, i64 2
  %17 = bitcast double* %16 to <2 x double>*
  %wide.load40 = load <2 x double>, <2 x double>* %17, align 8, !tbaa !3
  %18 = getelementptr inbounds double, double* %arrayidx29.us, i64 %index35
  %19 = bitcast double* %18 to <2 x double>*
  %wide.load41 = load <2 x double>, <2 x double>* %19, align 8, !tbaa !3
  %20 = getelementptr inbounds double, double* %18, i64 2
  %21 = bitcast double* %20 to <2 x double>*
  %wide.load42 = load <2 x double>, <2 x double>* %21, align 8, !tbaa !3
  %22 = fsub <2 x double> %wide.load39, %wide.load41
  %23 = fsub <2 x double> %wide.load40, %wide.load42
  %24 = getelementptr inbounds double, double* %2, i64 %index35
  %25 = bitcast double* %24 to <2 x double>*
  store <2 x double> %22, <2 x double>* %25, align 8, !tbaa !3
  %26 = getelementptr inbounds double, double* %24, i64 2
  %27 = bitcast double* %26 to <2 x double>*
  store <2 x double> %23, <2 x double>* %27, align 8, !tbaa !3
  %index.next36 = add i64 %index35, 4
  %28 = icmp eq i64 %index.next36, %n.vec34
  br i1 %28, label %middle.block28, label %vector.body30, !llvm.loop !56

middle.block28:                                   ; preds = %vector.body30
  br i1 %cmp.n38, label %for.body.i114.preheader.us, label %for.body.i128.us.preheader

for.body.i128.us.preheader:                       ; preds = %for.body23.us, %middle.block28
  %indvars.iv.i121.us.ph = phi i64 [ 0, %for.body23.us ], [ %n.vec34, %middle.block28 ]
  br label %for.body.i128.us

for.body.i128.us:                                 ; preds = %for.body.i128.us.preheader, %for.body.i128.us
  %indvars.iv.i121.us = phi i64 [ %indvars.iv.next.i126.us, %for.body.i128.us ], [ %indvars.iv.i121.us.ph, %for.body.i128.us.preheader ]
  %arrayidx.i122.us = getelementptr inbounds double, double* %arrayidx26, i64 %indvars.iv.i121.us
  %29 = load double, double* %arrayidx.i122.us, align 8, !tbaa !3
  %arrayidx2.i123.us = getelementptr inbounds double, double* %arrayidx29.us, i64 %indvars.iv.i121.us
  %30 = load double, double* %arrayidx2.i123.us, align 8, !tbaa !3
  %sub.i124.us = fsub double %29, %30
  %arrayidx4.i125.us = getelementptr inbounds double, double* %2, i64 %indvars.iv.i121.us
  store double %sub.i124.us, double* %arrayidx4.i125.us, align 8, !tbaa !3
  %indvars.iv.next.i126.us = add nuw nsw i64 %indvars.iv.i121.us, 1
  %exitcond.not.i127.us = icmp eq i64 %indvars.iv.next.i126.us, %wide.trip.count.i119
  br i1 %exitcond.not.i127.us, label %for.body.i114.preheader.us, label %for.body.i128.us, !llvm.loop !57

for.body.i114.preheader.us:                       ; preds = %for.body.i128.us, %middle.block28
  %arrayidx33.us = getelementptr inbounds double, double* %0, i64 %mul28.us
  %mul34.us = mul nsw i64 %ik.0133.us, %conv
  br i1 %min.iters.check16, label %for.body.i114.us.preheader, label %vector.body15

vector.body15:                                    ; preds = %for.body.i114.preheader.us, %vector.body15
  %index20 = phi i64 [ %index.next21, %vector.body15 ], [ 0, %for.body.i114.preheader.us ]
  %31 = getelementptr inbounds double, double* %arrayidx33.us, i64 %index20
  %32 = bitcast double* %31 to <2 x double>*
  %wide.load24 = load <2 x double>, <2 x double>* %32, align 8, !tbaa !3
  %33 = getelementptr inbounds double, double* %31, i64 2
  %34 = bitcast double* %33 to <2 x double>*
  %wide.load25 = load <2 x double>, <2 x double>* %34, align 8, !tbaa !3
  %35 = getelementptr inbounds double, double* %2, i64 %index20
  %36 = bitcast double* %35 to <2 x double>*
  %wide.load26 = load <2 x double>, <2 x double>* %36, align 8, !tbaa !3
  %37 = getelementptr inbounds double, double* %35, i64 2
  %38 = bitcast double* %37 to <2 x double>*
  %wide.load27 = load <2 x double>, <2 x double>* %38, align 8, !tbaa !3
  %39 = fmul <2 x double> %wide.load24, %wide.load26
  %40 = fmul <2 x double> %wide.load25, %wide.load27
  %41 = getelementptr inbounds double, double* %3, i64 %index20
  %42 = bitcast double* %41 to <2 x double>*
  store <2 x double> %39, <2 x double>* %42, align 8, !tbaa !3
  %43 = getelementptr inbounds double, double* %41, i64 2
  %44 = bitcast double* %43 to <2 x double>*
  store <2 x double> %40, <2 x double>* %44, align 8, !tbaa !3
  %index.next21 = add i64 %index20, 4
  %45 = icmp eq i64 %index.next21, %n.vec19
  br i1 %45, label %middle.block13, label %vector.body15, !llvm.loop !58

middle.block13:                                   ; preds = %vector.body15
  br i1 %cmp.n23, label %for.body7.i.us.preheader, label %for.body.i114.us.preheader

for.body.i114.us.preheader:                       ; preds = %for.body.i114.preheader.us, %middle.block13
  %indvars.iv69.i.us.ph = phi i64 [ 0, %for.body.i114.preheader.us ], [ %n.vec19, %middle.block13 ]
  br label %for.body.i114.us

for.body.i114.us:                                 ; preds = %for.body.i114.us.preheader, %for.body.i114.us
  %indvars.iv69.i.us = phi i64 [ %indvars.iv.next70.i.us, %for.body.i114.us ], [ %indvars.iv69.i.us.ph, %for.body.i114.us.preheader ]
  %arrayidx.i111.us = getelementptr inbounds double, double* %arrayidx33.us, i64 %indvars.iv69.i.us
  %46 = load double, double* %arrayidx.i111.us, align 8, !tbaa !3
  %arrayidx2.i112.us = getelementptr inbounds double, double* %2, i64 %indvars.iv69.i.us
  %47 = load double, double* %arrayidx2.i112.us, align 8, !tbaa !3
  %mul.i113.us = fmul double %46, %47
  %arrayidx4.i.us = getelementptr inbounds double, double* %3, i64 %indvars.iv69.i.us
  store double %mul.i113.us, double* %arrayidx4.i.us, align 8, !tbaa !3
  %indvars.iv.next70.i.us = add nuw nsw i64 %indvars.iv69.i.us, 1
  %exitcond72.not.i.us = icmp eq i64 %indvars.iv.next70.i.us, %wide.trip.count.i119
  br i1 %exitcond72.not.i.us, label %for.body7.i.us.preheader, label %for.body.i114.us, !llvm.loop !59

for.body7.i.us.preheader:                         ; preds = %for.body.i114.us, %middle.block13
  %arrayidx35.us = getelementptr inbounds double, double* %Ls, i64 %mul34.us
  br label %for.body7.i.us

for.body7.i.us:                                   ; preds = %for.body7.i.us.preheader, %for.cond5.loopexit.i.us
  %indvars.iv65.i.us = phi i64 [ %indvars.iv.next66.i.us, %for.cond5.loopexit.i.us ], [ 0, %for.body7.i.us.preheader ]
  %indvars.iv61.i.us = phi i64 [ %indvars.iv.next62.i.us, %for.cond5.loopexit.i.us ], [ 1, %for.body7.i.us.preheader ]
  %i.158.i.us = phi i32 [ %add.i115.us, %for.cond5.loopexit.i.us ], [ 0, %for.body7.i.us.preheader ]
  %48 = xor i64 %indvars.iv65.i.us, -1
  %49 = add nsw i64 %48, %wide.trip.count.i119
  %indvars.iv.next66.i.us = add nuw nsw i64 %indvars.iv65.i.us, 1
  %add.i115.us = add nuw nsw i32 %i.158.i.us, 1
  %cmp1254.i.us = icmp ult i64 %indvars.iv.next66.i.us, %wide.trip.count.i119
  br i1 %cmp1254.i.us, label %for.body13.lr.ph.i.us, label %for.cond5.loopexit.i.us

for.body13.lr.ph.i.us:                            ; preds = %for.body7.i.us
  %50 = xor i32 %i.158.i.us, -1
  %sub9.i.us = add i32 %mul8.i, %50
  %51 = trunc i64 %indvars.iv65.i.us to i32
  %mul10.i.us = mul nsw i32 %sub9.i.us, %51
  %div.i.us = sdiv i32 %mul10.i.us, 2
  %arrayidx19.i.us = getelementptr inbounds double, double* %2, i64 %indvars.iv65.i.us
  %52 = sext i32 %div.i.us to i64
  %53 = load double, double* %arrayidx19.i.us, align 8, !tbaa !3
  %min.iters.check = icmp ult i64 %49, 4
  br i1 %min.iters.check, label %for.body13.i.us.preheader, label %vector.ph

vector.ph:                                        ; preds = %for.body13.lr.ph.i.us
  %n.vec = and i64 %49, -4
  %ind.end = add i64 %indvars.iv61.i.us, %n.vec
  %ind.end6 = add i64 %n.vec, %52
  %broadcast.splatinsert = insertelement <2 x double> poison, double %53, i32 0
  %broadcast.splat = shufflevector <2 x double> %broadcast.splatinsert, <2 x double> poison, <2 x i32> zeroinitializer
  %broadcast.splatinsert11 = insertelement <2 x double> poison, double %53, i32 0
  %broadcast.splat12 = shufflevector <2 x double> %broadcast.splatinsert11, <2 x double> poison, <2 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %offset.idx = add i64 %indvars.iv61.i.us, %index
  %offset.idx7 = add i64 %index, %52
  %54 = getelementptr inbounds double, double* %3, i64 %offset.idx
  %55 = bitcast double* %54 to <2 x double>*
  %wide.load = load <2 x double>, <2 x double>* %55, align 8, !tbaa !3
  %56 = getelementptr inbounds double, double* %54, i64 2
  %57 = bitcast double* %56 to <2 x double>*
  %wide.load8 = load <2 x double>, <2 x double>* %57, align 8, !tbaa !3
  %58 = getelementptr inbounds double, double* %arrayidx35.us, i64 %offset.idx7
  %59 = bitcast double* %58 to <2 x double>*
  %wide.load9 = load <2 x double>, <2 x double>* %59, align 8, !tbaa !3
  %60 = getelementptr inbounds double, double* %58, i64 2
  %61 = bitcast double* %60 to <2 x double>*
  %wide.load10 = load <2 x double>, <2 x double>* %61, align 8, !tbaa !3
  %62 = fmul <2 x double> %broadcast.splat, %wide.load9
  %63 = fmul <2 x double> %broadcast.splat12, %wide.load10
  %64 = fadd <2 x double> %wide.load, %62
  %65 = fadd <2 x double> %wide.load8, %63
  %66 = bitcast double* %54 to <2 x double>*
  store <2 x double> %64, <2 x double>* %66, align 8, !tbaa !3
  %67 = bitcast double* %56 to <2 x double>*
  store <2 x double> %65, <2 x double>* %67, align 8, !tbaa !3
  %index.next = add i64 %index, 4
  %68 = icmp eq i64 %index.next, %n.vec
  br i1 %68, label %middle.block, label %vector.body, !llvm.loop !60

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %49, %n.vec
  br i1 %cmp.n, label %for.cond5.loopexit.i.us, label %for.body13.i.us.preheader

for.body13.i.us.preheader:                        ; preds = %for.body13.lr.ph.i.us, %middle.block
  %indvars.iv63.i.us.ph = phi i64 [ %indvars.iv61.i.us, %for.body13.lr.ph.i.us ], [ %ind.end, %middle.block ]
  %indvars.iv.i116.us.ph = phi i64 [ %52, %for.body13.lr.ph.i.us ], [ %ind.end6, %middle.block ]
  br label %for.body13.i.us

for.body13.i.us:                                  ; preds = %for.body13.i.us.preheader, %for.body13.i.us
  %indvars.iv63.i.us = phi i64 [ %indvars.iv.next64.i.us, %for.body13.i.us ], [ %indvars.iv63.i.us.ph, %for.body13.i.us.preheader ]
  %indvars.iv.i116.us = phi i64 [ %indvars.iv.next.i117.us, %for.body13.i.us ], [ %indvars.iv.i116.us.ph, %for.body13.i.us.preheader ]
  %arrayidx15.i.us = getelementptr inbounds double, double* %3, i64 %indvars.iv63.i.us
  %69 = load double, double* %arrayidx15.i.us, align 8, !tbaa !3
  %arrayidx17.i.us = getelementptr inbounds double, double* %arrayidx35.us, i64 %indvars.iv.i116.us
  %70 = load double, double* %arrayidx17.i.us, align 8, !tbaa !3
  %mul20.i.us = fmul double %53, %70
  %add21.i.us = fadd double %69, %mul20.i.us
  store double %add21.i.us, double* %arrayidx15.i.us, align 8, !tbaa !3
  %indvars.iv.next.i117.us = add nsw i64 %indvars.iv.i116.us, 1
  %indvars.iv.next64.i.us = add nuw nsw i64 %indvars.iv63.i.us, 1
  %exitcond.not.i118.us = icmp eq i64 %indvars.iv.next64.i.us, %wide.trip.count.i119
  br i1 %exitcond.not.i118.us, label %for.cond5.loopexit.i.us, label %for.body13.i.us, !llvm.loop !61

for.cond5.loopexit.i.us:                          ; preds = %for.body13.i.us, %middle.block, %for.body7.i.us
  %indvars.iv.next62.i.us = add nuw nsw i64 %indvars.iv61.i.us, 1
  %exitcond68.not.i.us = icmp eq i64 %indvars.iv.next66.i.us, %wide.trip.count.i119
  br i1 %exitcond68.not.i.us, label %cQtimesx.exit.loopexit.us, label %for.body7.i.us, !llvm.loop !45

cQtimesx.exit.loopexit.us:                        ; preds = %for.cond5.loopexit.i.us
  %.pre.us = load double, double* %3, align 8, !tbaa !3
  %arrayidx38.us = getelementptr inbounds double, double* %alphas, i64 %ik.0133.us
  %71 = load double, double* %arrayidx38.us, align 8, !tbaa !3
  %arrayidx39.us = getelementptr inbounds double, double* %1, i64 %ik.0133.us
  %72 = load double, double* %arrayidx39.us, align 8, !tbaa !3
  %add.us = fadd double %71, %72
  %mul.i102.us = fmul double %.pre.us, %.pre.us
  br i1 %cmp15.i, label %for.body.i109.us, label %sqnorm.exit.us

for.body.i109.us:                                 ; preds = %cQtimesx.exit.loopexit.us, %for.body.i109.us
  %indvars.iv.i105.us = phi i64 [ %indvars.iv.next.i107.us, %for.body.i109.us ], [ 1, %cQtimesx.exit.loopexit.us ]
  %res.017.i.us = phi double [ %add.i106.us, %for.body.i109.us ], [ %mul.i102.us, %cQtimesx.exit.loopexit.us ]
  %arrayidx2.i.us = getelementptr inbounds double, double* %3, i64 %indvars.iv.i105.us
  %73 = load double, double* %arrayidx2.i.us, align 8, !tbaa !3
  %mul5.i.us = fmul double %73, %73
  %add.i106.us = fadd double %res.017.i.us, %mul5.i.us
  %indvars.iv.next.i107.us = add nuw nsw i64 %indvars.iv.i105.us, 1
  %exitcond.not.i108.us = icmp eq i64 %indvars.iv.next.i107.us, %wide.trip.count.i119
  br i1 %exitcond.not.i108.us, label %sqnorm.exit.us, label %for.body.i109.us, !llvm.loop !10

sqnorm.exit.us:                                   ; preds = %for.body.i109.us, %cQtimesx.exit.loopexit.us
  %res.0.lcssa.i.us = phi double [ %mul.i102.us, %cQtimesx.exit.loopexit.us ], [ %add.i106.us, %for.body.i109.us ]
  %mul42.us = fmul double %res.0.lcssa.i.us, 5.000000e-01
  %sub43.us = fsub double %add.us, %mul42.us
  %arrayidx44.us = getelementptr inbounds double, double* %4, i64 %ik.0133.us
  store double %sub43.us, double* %arrayidx44.us, align 8, !tbaa !3
  %inc.us = add nuw nsw i64 %ik.0133.us, 1
  %exitcond.not.us = icmp eq i64 %inc.us, %conv4
  br i1 %exitcond.not.us, label %for.end.loopexit, label %for.body23.us, !llvm.loop !62

for.body23:                                       ; preds = %for.body23.preheader, %sqnorm.exit
  %ik.0133 = phi i64 [ %inc, %sqnorm.exit ], [ 0, %for.body23.preheader ]
  %arrayidx38 = getelementptr inbounds double, double* %alphas, i64 %ik.0133
  %74 = load double, double* %arrayidx38, align 8, !tbaa !3
  %arrayidx39 = getelementptr inbounds double, double* %1, i64 %ik.0133
  %75 = load double, double* %arrayidx39, align 8, !tbaa !3
  %add = fadd double %74, %75
  br i1 %cmp15.i, label %for.body.i109, label %sqnorm.exit

for.body.i109:                                    ; preds = %for.body23, %for.body.i109
  %indvars.iv.i105 = phi i64 [ %indvars.iv.next.i107, %for.body.i109 ], [ 1, %for.body23 ]
  %res.017.i = phi double [ %add.i106, %for.body.i109 ], [ %mul.i102, %for.body23 ]
  %arrayidx2.i = getelementptr inbounds double, double* %3, i64 %indvars.iv.i105
  %76 = load double, double* %arrayidx2.i, align 8, !tbaa !3
  %mul5.i = fmul double %76, %76
  %add.i106 = fadd double %res.017.i, %mul5.i
  %indvars.iv.next.i107 = add nuw nsw i64 %indvars.iv.i105, 1
  %exitcond.not.i108 = icmp eq i64 %indvars.iv.next.i107, %wide.trip.count.i119
  br i1 %exitcond.not.i108, label %sqnorm.exit, label %for.body.i109, !llvm.loop !10

sqnorm.exit:                                      ; preds = %for.body.i109, %for.body23
  %res.0.lcssa.i = phi double [ %mul.i102, %for.body23 ], [ %add.i106, %for.body.i109 ]
  %mul42 = fmul double %res.0.lcssa.i, 5.000000e-01
  %sub43 = fsub double %add, %mul42
  %arrayidx44 = getelementptr inbounds double, double* %4, i64 %ik.0133
  store double %sub43, double* %arrayidx44, align 8, !tbaa !3
  %inc = add nuw nsw i64 %ik.0133, 1
  %exitcond.not = icmp eq i64 %inc, %conv4
  br i1 %exitcond.not, label %for.end.loopexit, label %for.body23, !llvm.loop !62

for.end.loopexit:                                 ; preds = %sqnorm.exit, %sqnorm.exit.us
  %.lcssa = phi double [ %.pre.us, %sqnorm.exit.us ], [ %13, %sqnorm.exit ]
  %.pre146 = load double, double* %4, align 8, !tbaa !3
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %for.cond19.preheader
  %77 = phi double [ %.pre146, %for.end.loopexit ], [ %12, %for.cond19.preheader ]
  %78 = phi double [ %.lcssa, %for.end.loopexit ], [ %13, %for.cond19.preheader ]
  br i1 %cmp13.i.i, label %for.body.i.i, label %arr_max.exit.i

for.body.i.i:                                     ; preds = %for.end, %for.body.i.i
  %indvars.iv.i.i = phi i64 [ %indvars.iv.next.i.i, %for.body.i.i ], [ 1, %for.end ]
  %m.015.i.i = phi double [ %m.1.i.i, %for.body.i.i ], [ %77, %for.end ]
  %arrayidx1.i.i = getelementptr inbounds double, double* %4, i64 %indvars.iv.i.i
  %79 = load double, double* %arrayidx1.i.i, align 8, !tbaa !3
  %cmp2.i.i = fcmp olt double %m.015.i.i, %79
  %m.1.i.i = select i1 %cmp2.i.i, double %79, double %m.015.i.i
  %indvars.iv.next.i.i = add nuw nsw i64 %indvars.iv.i.i, 1
  %exitcond.not.i.i = icmp eq i64 %indvars.iv.next.i.i, %wide.trip.count.i.i
  br i1 %exitcond.not.i.i, label %arr_max.exit.i, label %for.body.i.i, !llvm.loop !7

arr_max.exit.i:                                   ; preds = %for.body.i.i, %for.end
  %m.0.lcssa.i.i = phi double [ %77, %for.end ], [ %m.1.i.i, %for.body.i.i ]
  br i1 %cmp37.i, label %for.body.preheader.i, label %log_sum_exp.exit

for.body.preheader.i:                             ; preds = %arr_max.exit.i
  %sub.i135 = fsub double %77, %m.0.lcssa.i.i
  %80 = tail call double @llvm.exp.f64(double %sub.i135) #14
  %add.i136 = fadd double %80, 0.000000e+00
  br i1 %exitcond.not.i99137, label %log_sum_exp.exit, label %for.body.for.body_crit_edge.i, !llvm.loop !22

for.body.for.body_crit_edge.i:                    ; preds = %for.body.preheader.i, %for.body.for.body_crit_edge.i
  %indvars.iv.next.i98139 = phi i64 [ %indvars.iv.next.i98, %for.body.for.body_crit_edge.i ], [ 1, %for.body.preheader.i ]
  %add.i138 = phi double [ %add.i, %for.body.for.body_crit_edge.i ], [ %add.i136, %for.body.preheader.i ]
  %arrayidx.phi.trans.insert.i = getelementptr inbounds double, double* %4, i64 %indvars.iv.next.i98139
  %.pre.i101 = load double, double* %arrayidx.phi.trans.insert.i, align 8, !tbaa !3
  %sub.i = fsub double %.pre.i101, %m.0.lcssa.i.i
  %81 = tail call double @llvm.exp.f64(double %sub.i) #14
  %add.i = fadd double %add.i138, %81
  %indvars.iv.next.i98 = add nuw nsw i64 %indvars.iv.next.i98139, 1
  %exitcond.not.i99 = icmp eq i64 %indvars.iv.next.i98, %wide.trip.count.i.i
  br i1 %exitcond.not.i99, label %log_sum_exp.exit, label %for.body.for.body_crit_edge.i, !llvm.loop !22

log_sum_exp.exit:                                 ; preds = %for.body.for.body_crit_edge.i, %for.body.preheader.i, %arr_max.exit.i
  %semx.0.lcssa.i = phi double [ 0.000000e+00, %arr_max.exit.i ], [ %add.i136, %for.body.preheader.i ], [ %add.i, %for.body.for.body_crit_edge.i ]
  %82 = tail call double @llvm.log.f64(double %semx.0.lcssa.i) #14
  %add1.i = fadd double %m.0.lcssa.i.i, %82
  %add47 = fadd double %slse.0143, %add1.i
  %inc49 = add nuw nsw i64 %ix.0141, 1
  %exitcond145.not = icmp eq i64 %inc49, %conv17
  br i1 %exitcond145.not, label %for.end50, label %for.cond19.preheader, !llvm.loop !63

for.end50:                                        ; preds = %log_sum_exp.exit, %preprocess_qs.exit
  %slse.0.lcssa = phi double [ 0.000000e+00, %preprocess_qs.exit ], [ %add47, %log_sum_exp.exit ]
  store double %slse.0.lcssa, double* %err, align 8, !tbaa !3
  tail call void @free(i8* %call)
  tail call void @free(i8* %call6)
  tail call void @free(i8* %call9)
  tail call void @free(i8* %call12)
  tail call void @free(i8* %call15)
  ret void
}

; Function Attrs: inaccessiblememonly nofree nounwind willreturn allocsize(0)
declare noalias noundef i8* @malloc(i64) local_unnamed_addr #6

; Function Attrs: inaccessiblemem_or_argmemonly nounwind willreturn
declare void @free(i8* nocapture noundef) local_unnamed_addr #7

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

; Function Attrs: nofree norecurse nounwind ssp uwtable
define dso_local void @cvecmat(i32 %d, double* nocapture readonly %x, double* nocapture readonly %ltri, double* nocapture %out) local_unnamed_addr #1 {
entry:
  %cmp37 = icmp sgt i32 %d, 0
  br i1 %cmp37, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %mul = shl nuw nsw i32 %d, 1
  %0 = zext i32 %d to i64
  br label %for.body

for.cond.loopexit:                                ; preds = %for.body6, %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond46.not = icmp eq i64 %indvars.iv.next44, %0
  br i1 %exitcond46.not, label %for.cond.cleanup, label %for.body, !llvm.loop !111

for.cond.cleanup:                                 ; preds = %for.cond.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.cond.loopexit, %for.body.lr.ph
  %indvars.iv43 = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next44, %for.cond.loopexit ]
  %indvars.iv = phi i64 [ 1, %for.body.lr.ph ], [ %indvars.iv.next, %for.cond.loopexit ]
  %i.038 = phi i32 [ 0, %for.body.lr.ph ], [ %add, %for.cond.loopexit ]
  %indvars.iv.next44 = add nuw nsw i64 %indvars.iv43, 1
  %add = add nuw nsw i32 %i.038, 1
  %cmp434 = icmp ult i64 %indvars.iv.next44, %0
  br i1 %cmp434, label %for.body6.lr.ph, label %for.cond.loopexit

for.body6.lr.ph:                                  ; preds = %for.body
  %1 = xor i32 %i.038, -1
  %sub1 = add i32 %mul, %1
  %2 = trunc i64 %indvars.iv43 to i32
  %mul2 = mul nsw i32 %sub1, %2
  %div = sdiv i32 %mul2, 2
  %arrayidx = getelementptr inbounds double, double* %out, i64 %indvars.iv43
  %3 = sext i32 %div to i64
  %.pre = load double, double* %arrayidx, align 8, !tbaa !3
  br label %for.body6

for.body6:                                        ; preds = %for.body6, %for.body6.lr.ph
  %4 = phi double [ %.pre, %for.body6.lr.ph ], [ %add12, %for.body6 ]
  %indvars.iv41 = phi i64 [ %3, %for.body6.lr.ph ], [ %indvars.iv.next42, %for.body6 ]
  %indvars.iv39 = phi i64 [ %indvars.iv, %for.body6.lr.ph ], [ %indvars.iv.next40, %for.body6 ]
  %arrayidx8 = getelementptr inbounds double, double* %ltri, i64 %indvars.iv41
  %5 = load double, double* %arrayidx8, align 8, !tbaa !3
  %arrayidx10 = getelementptr inbounds double, double* %x, i64 %indvars.iv39
  %6 = load double, double* %arrayidx10, align 8, !tbaa !3
  %mul11 = fmul double %5, %6
  %add12 = fadd double %4, %mul11
  store double %add12, double* %arrayidx, align 8, !tbaa !3
  %indvars.iv.next42 = add nsw i64 %indvars.iv41, 1
  %indvars.iv.next40 = add nuw nsw i64 %indvars.iv39, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next40, %0
  br i1 %exitcond.not, label %for.cond.loopexit, label %for.body6, !llvm.loop !112
}

; Function Attrs: nofree norecurse nounwind ssp uwtable
define dso_local void @couter(i32 %d, double* nocapture readonly %x, double* nocapture readonly %y, double* nocapture %out) local_unnamed_addr #1 {
entry:
  %cmp33 = icmp sgt i32 %d, 0
  br i1 %cmp33, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %mul = shl nuw nsw i32 %d, 1
  %0 = zext i32 %d to i64
  %scevgep7 = getelementptr double, double* %x, i64 %0
  br label %for.body

for.cond.loopexit:                                ; preds = %for.body6, %middle.block, %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond42.not = icmp eq i64 %indvars.iv.next40, %0
  br i1 %exitcond42.not, label %for.cond.cleanup, label %for.body, !llvm.loop !113

for.cond.cleanup:                                 ; preds = %for.cond.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.cond.loopexit, %for.body.lr.ph
  %indvars.iv39 = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next40, %for.cond.loopexit ]
  %indvars.iv = phi i64 [ 1, %for.body.lr.ph ], [ %indvars.iv.next, %for.cond.loopexit ]
  %i.034 = phi i32 [ 0, %for.body.lr.ph ], [ %add, %for.cond.loopexit ]
  %1 = xor i64 %indvars.iv39, -1
  %2 = add nsw i64 %1, %0
  %scevgep2 = getelementptr double, double* %out, i64 %2
  %3 = add nuw i64 %indvars.iv39, 1
  %scevgep5 = getelementptr double, double* %x, i64 %3
  %scevgep9 = getelementptr double, double* %y, i64 %indvars.iv39
  %scevgep910 = bitcast double* %scevgep9 to i8*
  %uglygep = getelementptr i8, i8* %scevgep910, i64 1
  %4 = xor i64 %indvars.iv39, -1
  %5 = add nsw i64 %4, %0
  %indvars.iv.next40 = add nuw nsw i64 %indvars.iv39, 1
  %add = add nuw nsw i32 %i.034, 1
  %cmp430 = icmp ult i64 %indvars.iv.next40, %0
  br i1 %cmp430, label %for.body6.lr.ph, label %for.cond.loopexit

for.body6.lr.ph:                                  ; preds = %for.body
  %6 = xor i32 %i.034, -1
  %sub1 = add i32 %mul, %6
  %7 = trunc i64 %indvars.iv39 to i32
  %mul2 = mul nsw i32 %sub1, %7
  %div = sdiv i32 %mul2, 2
  %arrayidx8 = getelementptr inbounds double, double* %y, i64 %indvars.iv39
  %8 = sext i32 %div to i64
  %min.iters.check = icmp ult i64 %5, 4
  br i1 %min.iters.check, label %for.body6.preheader, label %vector.memcheck

vector.memcheck:                                  ; preds = %for.body6.lr.ph
  %scevgep = getelementptr double, double* %out, i64 %8
  %scevgep1 = bitcast double* %scevgep to i8*
  %scevgep3 = getelementptr double, double* %scevgep2, i64 %8
  %bound0 = icmp ult double* %scevgep, %scevgep7
  %bound1 = icmp ult double* %scevgep5, %scevgep3
  %found.conflict = and i1 %bound0, %bound1
  %bound011 = icmp ugt i8* %uglygep, %scevgep1
  %bound112 = icmp ult double* %arrayidx8, %scevgep3
  %found.conflict13 = and i1 %bound011, %bound112
  %conflict.rdx = or i1 %found.conflict, %found.conflict13
  br i1 %conflict.rdx, label %for.body6.preheader, label %vector.ph

vector.ph:                                        ; preds = %vector.memcheck
  %n.vec = and i64 %5, -4
  %ind.end = add i64 %n.vec, %8
  %ind.end15 = add i64 %indvars.iv, %n.vec
  %9 = load double, double* %arrayidx8, align 8, !tbaa !3, !alias.scope !114
  %broadcast.splatinsert = insertelement <2 x double> poison, double %9, i32 0
  %broadcast.splat = shufflevector <2 x double> %broadcast.splatinsert, <2 x double> poison, <2 x i32> zeroinitializer
  %broadcast.splatinsert18 = insertelement <2 x double> poison, double %9, i32 0
  %broadcast.splat19 = shufflevector <2 x double> %broadcast.splatinsert18, <2 x double> poison, <2 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %offset.idx = add i64 %index, %8
  %offset.idx16 = add i64 %indvars.iv, %index
  %10 = getelementptr inbounds double, double* %x, i64 %offset.idx16
  %11 = bitcast double* %10 to <2 x double>*
  %wide.load = load <2 x double>, <2 x double>* %11, align 8, !tbaa !3, !alias.scope !117
  %12 = getelementptr inbounds double, double* %10, i64 2
  %13 = bitcast double* %12 to <2 x double>*
  %wide.load17 = load <2 x double>, <2 x double>* %13, align 8, !tbaa !3, !alias.scope !117
  %14 = fmul <2 x double> %wide.load, %broadcast.splat
  %15 = fmul <2 x double> %wide.load17, %broadcast.splat19
  %16 = getelementptr inbounds double, double* %out, i64 %offset.idx
  %17 = bitcast double* %16 to <2 x double>*
  %wide.load20 = load <2 x double>, <2 x double>* %17, align 8, !tbaa !3, !alias.scope !119, !noalias !121
  %18 = getelementptr inbounds double, double* %16, i64 2
  %19 = bitcast double* %18 to <2 x double>*
  %wide.load21 = load <2 x double>, <2 x double>* %19, align 8, !tbaa !3, !alias.scope !119, !noalias !121
  %20 = fadd <2 x double> %wide.load20, %14
  %21 = fadd <2 x double> %wide.load21, %15
  %22 = bitcast double* %16 to <2 x double>*
  store <2 x double> %20, <2 x double>* %22, align 8, !tbaa !3, !alias.scope !119, !noalias !121
  %23 = bitcast double* %18 to <2 x double>*
  store <2 x double> %21, <2 x double>* %23, align 8, !tbaa !3, !alias.scope !119, !noalias !121
  %index.next = add i64 %index, 4
  %24 = icmp eq i64 %index.next, %n.vec
  br i1 %24, label %middle.block, label %vector.body, !llvm.loop !122

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %5, %n.vec
  br i1 %cmp.n, label %for.cond.loopexit, label %for.body6.preheader

for.body6.preheader:                              ; preds = %vector.memcheck, %for.body6.lr.ph, %middle.block
  %indvars.iv37.ph = phi i64 [ %8, %vector.memcheck ], [ %8, %for.body6.lr.ph ], [ %ind.end, %middle.block ]
  %indvars.iv35.ph = phi i64 [ %indvars.iv, %vector.memcheck ], [ %indvars.iv, %for.body6.lr.ph ], [ %ind.end15, %middle.block ]
  br label %for.body6

for.body6:                                        ; preds = %for.body6.preheader, %for.body6
  %indvars.iv37 = phi i64 [ %indvars.iv.next38, %for.body6 ], [ %indvars.iv37.ph, %for.body6.preheader ]
  %indvars.iv35 = phi i64 [ %indvars.iv.next36, %for.body6 ], [ %indvars.iv35.ph, %for.body6.preheader ]
  %arrayidx = getelementptr inbounds double, double* %x, i64 %indvars.iv35
  %25 = load double, double* %arrayidx, align 8, !tbaa !3
  %26 = load double, double* %arrayidx8, align 8, !tbaa !3
  %mul9 = fmul double %25, %26
  %arrayidx11 = getelementptr inbounds double, double* %out, i64 %indvars.iv37
  %27 = load double, double* %arrayidx11, align 8, !tbaa !3
  %add12 = fadd double %27, %mul9
  store double %add12, double* %arrayidx11, align 8, !tbaa !3
  %indvars.iv.next38 = add nsw i64 %indvars.iv37, 1
  %indvars.iv.next36 = add nuw nsw i64 %indvars.iv35, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next36, %0
  br i1 %exitcond.not, label %for.cond.loopexit, label %for.body6, !llvm.loop !123
}

; Function Attrs: nofree nounwind ssp uwtable
define dso_local void @cgrad_log_sum_exp(i32 %d, double* nocapture readonly %x, double %g, double* nocapture %dx) local_unnamed_addr #4 {
entry:
  %0 = load double, double* %x, align 8, !tbaa !3
  %cmp13.i.i = icmp sgt i32 %d, 1
  br i1 %cmp13.i.i, label %for.body.preheader.i.i, label %arr_max.exit.i

for.body.preheader.i.i:                           ; preds = %entry
  %wide.trip.count.i.i = zext i32 %d to i64
  br label %for.body.i.i

for.body.i.i:                                     ; preds = %for.body.i.i, %for.body.preheader.i.i
  %indvars.iv.i.i = phi i64 [ 1, %for.body.preheader.i.i ], [ %indvars.iv.next.i.i, %for.body.i.i ]
  %m.015.i.i = phi double [ %0, %for.body.preheader.i.i ], [ %m.1.i.i, %for.body.i.i ]
  %arrayidx1.i.i = getelementptr inbounds double, double* %x, i64 %indvars.iv.i.i
  %1 = load double, double* %arrayidx1.i.i, align 8, !tbaa !3
  %cmp2.i.i = fcmp olt double %m.015.i.i, %1
  %m.1.i.i = select i1 %cmp2.i.i, double %1, double %m.015.i.i
  %indvars.iv.next.i.i = add nuw nsw i64 %indvars.iv.i.i, 1
  %exitcond.not.i.i = icmp eq i64 %indvars.iv.next.i.i, %wide.trip.count.i.i
  br i1 %exitcond.not.i.i, label %arr_max.exit.i, label %for.body.i.i, !llvm.loop !7

arr_max.exit.i:                                   ; preds = %for.body.i.i, %entry
  %m.0.lcssa.i.i = phi double [ %0, %entry ], [ %m.1.i.i, %for.body.i.i ]
  %cmp11.i = icmp sgt i32 %d, 0
  br i1 %cmp11.i, label %for.body.preheader.i, label %log_sum_exp.exit

for.body.preheader.i:                             ; preds = %arr_max.exit.i
  %wide.trip.count.i = zext i32 %d to i64
  %sub.i12 = fsub double %0, %m.0.lcssa.i.i
  %2 = tail call double @llvm.exp.f64(double %sub.i12) #14
  %add.i13 = fadd double %2, 0.000000e+00
  %exitcond.not.i14 = icmp eq i32 %d, 1
  br i1 %exitcond.not.i14, label %log_sum_exp.exit.thread, label %for.body.for.body_crit_edge.i, !llvm.loop !22

log_sum_exp.exit.thread:                          ; preds = %for.body.preheader.i
  %3 = tail call double @llvm.log.f64(double %add.i13) #14
  br label %for.body.preheader

for.body.for.body_crit_edge.i:                    ; preds = %for.body.preheader.i, %for.body.for.body_crit_edge.i
  %indvars.iv.next.i16 = phi i64 [ %indvars.iv.next.i, %for.body.for.body_crit_edge.i ], [ 1, %for.body.preheader.i ]
  %add.i15 = phi double [ %add.i, %for.body.for.body_crit_edge.i ], [ %add.i13, %for.body.preheader.i ]
  %arrayidx.phi.trans.insert.i = getelementptr inbounds double, double* %x, i64 %indvars.iv.next.i16
  %.pre.i = load double, double* %arrayidx.phi.trans.insert.i, align 8, !tbaa !3
  %sub.i = fsub double %.pre.i, %m.0.lcssa.i.i
  %4 = tail call double @llvm.exp.f64(double %sub.i) #14
  %add.i = fadd double %add.i15, %4
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.next.i16, 1
  %exitcond.not.i = icmp eq i64 %indvars.iv.next.i, %wide.trip.count.i
  br i1 %exitcond.not.i, label %log_sum_exp.exit, label %for.body.for.body_crit_edge.i, !llvm.loop !22

log_sum_exp.exit:                                 ; preds = %for.body.for.body_crit_edge.i, %arr_max.exit.i
  %semx.0.lcssa.i = phi double [ 0.000000e+00, %arr_max.exit.i ], [ %add.i, %for.body.for.body_crit_edge.i ]
  %5 = tail call double @llvm.log.f64(double %semx.0.lcssa.i) #14
  %conv = sext i32 %d to i64
  %cmp10.not = icmp eq i32 %d, 0
  br i1 %cmp10.not, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %log_sum_exp.exit, %log_sum_exp.exit.thread
  %conv22 = phi i64 [ 1, %log_sum_exp.exit.thread ], [ %conv, %log_sum_exp.exit ]
  %.pn = phi double [ %3, %log_sum_exp.exit.thread ], [ %5, %log_sum_exp.exit ]
  %add1.i21 = fadd double %m.0.lcssa.i.i, %.pn
  %6 = icmp ult i64 %conv22, 2
  %sub24 = fsub double %0, %add1.i21
  %7 = tail call double @llvm.exp.f64(double %sub24)
  %mul25 = fmul double %7, %g
  store double %mul25, double* %dx, align 8, !tbaa !3
  br i1 %6, label %for.cond.cleanup, label %for.body.for.body_crit_edge.preheader, !llvm.loop !124

for.body.for.body_crit_edge.preheader:            ; preds = %for.body.preheader
  %8 = add nsw i64 %conv22, -1
  %min.iters.check = icmp ult i64 %8, 2
  br i1 %min.iters.check, label %for.body.for.body_crit_edge.preheader10, label %vector.memcheck

vector.memcheck:                                  ; preds = %for.body.for.body_crit_edge.preheader
  %scevgep = getelementptr double, double* %dx, i64 1
  %scevgep2 = getelementptr double, double* %dx, i64 %conv22
  %scevgep4 = getelementptr double, double* %x, i64 1
  %scevgep6 = getelementptr double, double* %x, i64 %conv22
  %bound0 = icmp ult double* %scevgep, %scevgep6
  %bound1 = icmp ult double* %scevgep4, %scevgep2
  %found.conflict = and i1 %bound0, %bound1
  br i1 %found.conflict, label %for.body.for.body_crit_edge.preheader10, label %vector.ph

vector.ph:                                        ; preds = %vector.memcheck
  %n.vec = and i64 %8, -2
  %ind.end = or i64 %8, 1
  %broadcast.splatinsert = insertelement <2 x double> poison, double %add1.i21, i32 0
  %broadcast.splat = shufflevector <2 x double> %broadcast.splatinsert, <2 x double> poison, <2 x i32> zeroinitializer
  %broadcast.splatinsert8 = insertelement <2 x double> poison, double %g, i32 0
  %broadcast.splat9 = shufflevector <2 x double> %broadcast.splatinsert8, <2 x double> poison, <2 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %offset.idx = or i64 %index, 1
  %9 = getelementptr inbounds double, double* %x, i64 %offset.idx
  %10 = bitcast double* %9 to <2 x double>*
  %wide.load = load <2 x double>, <2 x double>* %10, align 8, !tbaa !3, !alias.scope !125
  %11 = fsub <2 x double> %wide.load, %broadcast.splat
  %12 = call <2 x double> @llvm.exp.v2f64(<2 x double> %11)
  %13 = fmul <2 x double> %12, %broadcast.splat9
  %14 = getelementptr inbounds double, double* %dx, i64 %offset.idx
  %15 = bitcast double* %14 to <2 x double>*
  store <2 x double> %13, <2 x double>* %15, align 8, !tbaa !3, !alias.scope !128, !noalias !125
  %index.next = add i64 %index, 2
  %16 = icmp eq i64 %index.next, %n.vec
  br i1 %16, label %middle.block, label %vector.body, !llvm.loop !130

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %8, %n.vec
  br i1 %cmp.n, label %for.cond.cleanup, label %for.body.for.body_crit_edge.preheader10

for.body.for.body_crit_edge.preheader10:          ; preds = %vector.memcheck, %for.body.for.body_crit_edge.preheader, %middle.block
  %inc27.ph = phi i64 [ 1, %vector.memcheck ], [ 1, %for.body.for.body_crit_edge.preheader ], [ %ind.end, %middle.block ]
  br label %for.body.for.body_crit_edge

for.cond.cleanup:                                 ; preds = %for.body.for.body_crit_edge, %middle.block, %for.body.preheader, %log_sum_exp.exit
  ret void

for.body.for.body_crit_edge:                      ; preds = %for.body.for.body_crit_edge.preheader10, %for.body.for.body_crit_edge
  %inc27 = phi i64 [ %inc, %for.body.for.body_crit_edge ], [ %inc27.ph, %for.body.for.body_crit_edge.preheader10 ]
  %arrayidx.phi.trans.insert = getelementptr inbounds double, double* %x, i64 %inc27
  %.pre = load double, double* %arrayidx.phi.trans.insert, align 8, !tbaa !3
  %sub = fsub double %.pre, %add1.i21
  %17 = tail call double @llvm.exp.f64(double %sub)
  %mul = fmul double %17, %g
  %arrayidx2 = getelementptr inbounds double, double* %dx, i64 %inc27
  store double %mul, double* %arrayidx2, align 8, !tbaa !3
  %inc = add nuw i64 %inc27, 1
  %exitcond.not = icmp eq i64 %inc, %conv22
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body.for.body_crit_edge, !llvm.loop !131
}

; Function Attrs: nounwind ssp uwtable
define dso_local void @manual_c_main_term(i32 %d, i32 %k, i32 %n, double* nocapture readonly %alphas, double* nocapture %alphasb, double* nocapture readonly %means, double* nocapture %meansb, double* nocapture readonly %Qs, double* nocapture %Qsb, double* nocapture readonly %Ls, double* nocapture %Lsb, double* nocapture readonly %x) local_unnamed_addr #5 {
entry:
  %sub = add nsw i32 %d, -1
  %mul = mul nsw i32 %sub, %d
  %div = sdiv i32 %mul, 2
  %conv = sext i32 %div to i64
  %mul1 = mul nsw i32 %k, %d
  %conv2 = sext i32 %mul1 to i64
  %mul3 = shl nsw i64 %conv2, 3
  %call = tail call i8* @malloc(i64 %mul3) #13
  %0 = bitcast i8* %call to double*
  %call6 = tail call i8* @calloc(i64 %conv2, i64 8) #16
  %1 = bitcast i8* %call6 to double*
  %conv7 = sext i32 %k to i64
  %mul8 = shl nsw i64 %conv7, 3
  %call9 = tail call i8* @malloc(i64 %mul8) #13
  %2 = bitcast i8* %call9 to double*
  %call11 = tail call i8* @calloc(i64 %conv7, i64 8) #16
  %3 = bitcast i8* %call11 to double*
  %conv12 = sext i32 %d to i64
  %mul13 = shl nsw i64 %conv12, 3
  %call14 = tail call i8* @malloc(i64 %mul13) #13
  %4 = bitcast i8* %call14 to double*
  %call17 = tail call i8* @malloc(i64 %mul13) #13
  %5 = bitcast i8* %call17 to double*
  %call20 = tail call i8* @malloc(i64 %mul13) #13
  %6 = bitcast i8* %call20 to double*
  %call23 = tail call i8* @malloc(i64 %mul13) #13
  %7 = bitcast i8* %call23 to double*
  %call26 = tail call i8* @malloc(i64 %mul8) #13
  %8 = bitcast i8* %call26 to double*
  %call29 = tail call i8* @malloc(i64 %mul8) #13
  %9 = bitcast i8* %call29 to double*
  %cmp37.i = icmp sgt i32 %k, 0
  br i1 %cmp37.i, label %for.body.lr.ph.i, label %preprocess_qs.exit

for.body.lr.ph.i:                                 ; preds = %entry
  %cmp235.i = icmp sgt i32 %d, 0
  %wide.trip.count44.i = zext i32 %k to i64
  %wide.trip.count.i = zext i32 %d to i64
  br i1 %cmp235.i, label %for.body.i.us, label %for.body.i.preheader

for.body.i.preheader:                             ; preds = %for.body.lr.ph.i
  %10 = shl nuw nsw i64 %wide.trip.count44.i, 3
  call void @llvm.memset.p0i8.i64(i8* align 8 %call9, i8 0, i64 %10, i1 false)
  br label %preprocess_qs.exit

for.body.i.us:                                    ; preds = %for.body.lr.ph.i, %for.inc15.i.loopexit.us
  %indvars.iv42.i.us = phi i64 [ %indvars.iv.next43.i.us, %for.inc15.i.loopexit.us ], [ 0, %for.body.lr.ph.i ]
  %arrayidx.i.us = getelementptr inbounds double, double* %2, i64 %indvars.iv42.i.us
  store double 0.000000e+00, double* %arrayidx.i.us, align 8, !tbaa !3
  %11 = trunc i64 %indvars.iv42.i.us to i32
  %mul.i.us = mul nsw i32 %11, %d
  %12 = sext i32 %mul.i.us to i64
  br label %for.body3.i.us

for.body3.i.us:                                   ; preds = %for.body3.i.us, %for.body.i.us
  %13 = phi double [ 0.000000e+00, %for.body.i.us ], [ %add8.i.us, %for.body3.i.us ]
  %indvars.iv.i.us = phi i64 [ 0, %for.body.i.us ], [ %indvars.iv.next.i.us, %for.body3.i.us ]
  %14 = add nsw i64 %indvars.iv.i.us, %12
  %arrayidx5.i.us = getelementptr inbounds double, double* %Qs, i64 %14
  %15 = load double, double* %arrayidx5.i.us, align 8, !tbaa !3
  %add8.i.us = fadd double %13, %15
  %16 = tail call double @llvm.exp.f64(double %15) #14
  %arrayidx14.i.us = getelementptr inbounds double, double* %0, i64 %14
  store double %16, double* %arrayidx14.i.us, align 8, !tbaa !3
  %indvars.iv.next.i.us = add nuw nsw i64 %indvars.iv.i.us, 1
  %exitcond.not.i.us = icmp eq i64 %indvars.iv.next.i.us, %wide.trip.count.i
  br i1 %exitcond.not.i.us, label %for.inc15.i.loopexit.us, label %for.body3.i.us, !llvm.loop !33

for.inc15.i.loopexit.us:                          ; preds = %for.body3.i.us
  store double %add8.i.us, double* %arrayidx.i.us, align 8, !tbaa !3
  %indvars.iv.next43.i.us = add nuw nsw i64 %indvars.iv42.i.us, 1
  %exitcond45.not.i.us = icmp eq i64 %indvars.iv.next43.i.us, %wide.trip.count44.i
  br i1 %exitcond45.not.i.us, label %preprocess_qs.exit, label %for.body.i.us, !llvm.loop !34

preprocess_qs.exit:                               ; preds = %for.inc15.i.loopexit.us, %for.body.i.preheader, %entry
  %call34 = tail call i8* @malloc(i64 %mul3) #13
  %17 = bitcast i8* %call34 to double*
  %call38 = tail call i8* @malloc(i64 %mul3) #13
  %18 = bitcast i8* %call38 to double*
  %conv39 = sext i32 %n to i64
  %cmp461 = icmp sgt i32 %n, 0
  br i1 %cmp461, label %for.cond41.preheader.lr.ph, label %for.cond173.preheader

for.cond41.preheader.lr.ph:                       ; preds = %preprocess_qs.exit
  %cmp10.i = icmp sgt i32 %d, 0
  %wide.trip.count.i423 = zext i32 %d to i64
  %mul8.i = shl nuw nsw i32 %d, 1
  %cmp15.i = icmp sgt i32 %d, 1
  %cmp13.i.i.i = icmp sgt i32 %k, 1
  %wide.trip.count.i.i.i = zext i32 %k to i64
  %exitcond.not.i14.i = icmp eq i32 %k, 1
  %cmp10.not.i = icmp eq i32 %k, 0
  %19 = icmp ugt i64 %conv7, 1
  %spec.select = select i1 %19, i64 %conv7, i64 1
  %exitcond.not.i401450 = icmp ult i32 %k, 2
  %20 = add nsw i64 %spec.select, -1
  %min.iters.check114 = icmp ult i32 %d, 4
  %n.vec117 = and i64 %wide.trip.count.i423, 4294967292
  %cmp.n121 = icmp eq i64 %n.vec117, %wide.trip.count.i423
  %min.iters.check99 = icmp ult i32 %d, 4
  %n.vec102 = and i64 %wide.trip.count.i423, 4294967292
  %cmp.n106 = icmp eq i64 %n.vec102, %wide.trip.count.i423
  %min.iters.check59 = icmp ult i64 %20, 2
  %n.vec62 = and i64 %20, -2
  %ind.end66 = or i64 %20, 1
  %cmp.n67 = icmp eq i64 %20, %n.vec62
  %min.iters.check45 = icmp ult i32 %k, 2
  %n.vec48 = and i64 %conv7, -2
  %cmp.n52 = icmp eq i64 %n.vec48, %conv7
  %min.iters.check30 = icmp eq i32 %d, 1
  %n.vec33 = and i64 %conv12, -2
  %cmp.n37 = icmp eq i64 %n.vec33, %conv12
  %min.iters.check = icmp ult i32 %d, 2
  %n.vec = and i64 %conv12, -2
  %cmp.n = icmp eq i64 %n.vec, %conv12
  br label %for.cond41.preheader

for.cond41.preheader:                             ; preds = %for.inc169, %for.cond41.preheader.lr.ph
  %21 = phi double [ undef, %for.cond41.preheader.lr.ph ], [ %106, %for.inc169 ]
  %ix.0462 = phi i64 [ 0, %for.cond41.preheader.lr.ph ], [ %inc170, %for.inc169 ]
  br i1 %cmp37.i, label %for.body45.lr.ph, label %for.end

for.body45.lr.ph:                                 ; preds = %for.cond41.preheader
  %mul47 = mul nsw i64 %ix.0462, %conv12
  %arrayidx48 = getelementptr inbounds double, double* %x, i64 %mul47
  br label %for.body45

for.cond173.preheader:                            ; preds = %for.inc169, %preprocess_qs.exit
  %cmp181441 = icmp sgt i32 %d, 0
  %or.cond = and i1 %cmp37.i, %cmp181441
  br i1 %or.cond, label %for.cond179.preheader.us.preheader, label %for.cond.cleanup177

for.cond179.preheader.us.preheader:               ; preds = %for.cond173.preheader
  %min.iters.check129 = icmp ult i32 %d, 4
  %n.vec132 = and i64 %conv12, -4
  %cmp.n136 = icmp eq i64 %n.vec132, %conv12
  br label %for.cond179.preheader.us

for.cond179.preheader.us:                         ; preds = %for.cond179.preheader.us.preheader, %for.cond.cleanup183.loopexit.us
  %i172.0444.us = phi i64 [ %inc204.us, %for.cond.cleanup183.loopexit.us ], [ 0, %for.cond179.preheader.us.preheader ]
  %arrayidx185.us = getelementptr inbounds double, double* %3, i64 %i172.0444.us
  %22 = load double, double* %arrayidx185.us, align 8, !tbaa !3
  %mul187.us = mul nsw i64 %i172.0444.us, %conv12
  br i1 %min.iters.check129, label %for.body184.us.preheader, label %vector.ph130

vector.ph130:                                     ; preds = %for.cond179.preheader.us
  %broadcast.splatinsert141 = insertelement <2 x double> poison, double %22, i32 0
  %broadcast.splat142 = shufflevector <2 x double> %broadcast.splatinsert141, <2 x double> poison, <2 x i32> zeroinitializer
  %broadcast.splatinsert143 = insertelement <2 x double> poison, double %22, i32 0
  %broadcast.splat144 = shufflevector <2 x double> %broadcast.splatinsert143, <2 x double> poison, <2 x i32> zeroinitializer
  br label %vector.body128

vector.body128:                                   ; preds = %vector.body128, %vector.ph130
  %index133 = phi i64 [ 0, %vector.ph130 ], [ %index.next134, %vector.body128 ]
  %23 = add nsw i64 %index133, %mul187.us
  %24 = getelementptr inbounds double, double* %1, i64 %23
  %25 = bitcast double* %24 to <2 x double>*
  %wide.load137 = load <2 x double>, <2 x double>* %25, align 8, !tbaa !3
  %26 = getelementptr inbounds double, double* %24, i64 2
  %27 = bitcast double* %26 to <2 x double>*
  %wide.load138 = load <2 x double>, <2 x double>* %27, align 8, !tbaa !3
  %28 = getelementptr inbounds double, double* %0, i64 %23
  %29 = bitcast double* %28 to <2 x double>*
  %wide.load139 = load <2 x double>, <2 x double>* %29, align 8, !tbaa !3
  %30 = getelementptr inbounds double, double* %28, i64 2
  %31 = bitcast double* %30 to <2 x double>*
  %wide.load140 = load <2 x double>, <2 x double>* %31, align 8, !tbaa !3
  %32 = fmul <2 x double> %wide.load137, %wide.load139
  %33 = fmul <2 x double> %wide.load138, %wide.load140
  %34 = fadd <2 x double> %broadcast.splat142, %32
  %35 = fadd <2 x double> %broadcast.splat144, %33
  %36 = getelementptr inbounds double, double* %Qsb, i64 %23
  %37 = bitcast double* %36 to <2 x double>*
  store <2 x double> %34, <2 x double>* %37, align 8, !tbaa !3
  %38 = getelementptr inbounds double, double* %36, i64 2
  %39 = bitcast double* %38 to <2 x double>*
  store <2 x double> %35, <2 x double>* %39, align 8, !tbaa !3
  %index.next134 = add i64 %index133, 4
  %40 = icmp eq i64 %index.next134, %n.vec132
  br i1 %40, label %middle.block126, label %vector.body128, !llvm.loop !132

middle.block126:                                  ; preds = %vector.body128
  br i1 %cmp.n136, label %for.cond.cleanup183.loopexit.us, label %for.body184.us.preheader

for.body184.us.preheader:                         ; preds = %for.cond179.preheader.us, %middle.block126
  %j.0442.us.ph = phi i64 [ 0, %for.cond179.preheader.us ], [ %n.vec132, %middle.block126 ]
  br label %for.body184.us

for.body184.us:                                   ; preds = %for.body184.us.preheader, %for.body184.us
  %j.0442.us = phi i64 [ %inc201.us, %for.body184.us ], [ %j.0442.us.ph, %for.body184.us.preheader ]
  %add188.us = add nsw i64 %j.0442.us, %mul187.us
  %arrayidx189.us = getelementptr inbounds double, double* %1, i64 %add188.us
  %41 = load double, double* %arrayidx189.us, align 8, !tbaa !3
  %arrayidx193.us = getelementptr inbounds double, double* %0, i64 %add188.us
  %42 = load double, double* %arrayidx193.us, align 8, !tbaa !3
  %mul194.us = fmul double %41, %42
  %add195.us = fadd double %22, %mul194.us
  %arrayidx199.us = getelementptr inbounds double, double* %Qsb, i64 %add188.us
  store double %add195.us, double* %arrayidx199.us, align 8, !tbaa !3
  %inc201.us = add nuw nsw i64 %j.0442.us, 1
  %exitcond.not.us = icmp eq i64 %inc201.us, %conv12
  br i1 %exitcond.not.us, label %for.cond.cleanup183.loopexit.us, label %for.body184.us, !llvm.loop !133

for.cond.cleanup183.loopexit.us:                  ; preds = %for.body184.us, %middle.block126
  %inc204.us = add nuw nsw i64 %i172.0444.us, 1
  %exitcond464.not.us = icmp eq i64 %inc204.us, %conv7
  br i1 %exitcond464.not.us, label %for.cond.cleanup177, label %for.cond179.preheader.us, !llvm.loop !134

for.body45:                                       ; preds = %sqnorm.exit, %for.body45.lr.ph
  %ik.0447 = phi i64 [ 0, %for.body45.lr.ph ], [ %inc, %sqnorm.exit ]
  %mul50 = mul nsw i64 %ik.0447, %conv12
  %arrayidx51 = getelementptr inbounds double, double* %means, i64 %mul50
  br i1 %cmp10.i, label %for.body.i432.preheader, label %cQtimesx.exit

for.body.i432.preheader:                          ; preds = %for.body45
  br i1 %min.iters.check114, label %for.body.i432.preheader147, label %vector.body113

vector.body113:                                   ; preds = %for.body.i432.preheader, %vector.body113
  %index118 = phi i64 [ %index.next119, %vector.body113 ], [ 0, %for.body.i432.preheader ]
  %43 = getelementptr inbounds double, double* %arrayidx48, i64 %index118
  %44 = bitcast double* %43 to <2 x double>*
  %wide.load122 = load <2 x double>, <2 x double>* %44, align 8, !tbaa !3
  %45 = getelementptr inbounds double, double* %43, i64 2
  %46 = bitcast double* %45 to <2 x double>*
  %wide.load123 = load <2 x double>, <2 x double>* %46, align 8, !tbaa !3
  %47 = getelementptr inbounds double, double* %arrayidx51, i64 %index118
  %48 = bitcast double* %47 to <2 x double>*
  %wide.load124 = load <2 x double>, <2 x double>* %48, align 8, !tbaa !3
  %49 = getelementptr inbounds double, double* %47, i64 2
  %50 = bitcast double* %49 to <2 x double>*
  %wide.load125 = load <2 x double>, <2 x double>* %50, align 8, !tbaa !3
  %51 = fsub <2 x double> %wide.load122, %wide.load124
  %52 = fsub <2 x double> %wide.load123, %wide.load125
  %53 = getelementptr inbounds double, double* %4, i64 %index118
  %54 = bitcast double* %53 to <2 x double>*
  store <2 x double> %51, <2 x double>* %54, align 8, !tbaa !3
  %55 = getelementptr inbounds double, double* %53, i64 2
  %56 = bitcast double* %55 to <2 x double>*
  store <2 x double> %52, <2 x double>* %56, align 8, !tbaa !3
  %index.next119 = add i64 %index118, 4
  %57 = icmp eq i64 %index.next119, %n.vec117
  br i1 %57, label %middle.block111, label %vector.body113, !llvm.loop !135

middle.block111:                                  ; preds = %vector.body113
  br i1 %cmp.n121, label %for.body.i417.preheader, label %for.body.i432.preheader147

for.body.i432.preheader147:                       ; preds = %for.body.i432.preheader, %middle.block111
  %indvars.iv.i425.ph = phi i64 [ 0, %for.body.i432.preheader ], [ %n.vec117, %middle.block111 ]
  br label %for.body.i432

for.body.i432:                                    ; preds = %for.body.i432.preheader147, %for.body.i432
  %indvars.iv.i425 = phi i64 [ %indvars.iv.next.i430, %for.body.i432 ], [ %indvars.iv.i425.ph, %for.body.i432.preheader147 ]
  %arrayidx.i426 = getelementptr inbounds double, double* %arrayidx48, i64 %indvars.iv.i425
  %58 = load double, double* %arrayidx.i426, align 8, !tbaa !3
  %arrayidx2.i427 = getelementptr inbounds double, double* %arrayidx51, i64 %indvars.iv.i425
  %59 = load double, double* %arrayidx2.i427, align 8, !tbaa !3
  %sub.i428 = fsub double %58, %59
  %arrayidx4.i429 = getelementptr inbounds double, double* %4, i64 %indvars.iv.i425
  store double %sub.i428, double* %arrayidx4.i429, align 8, !tbaa !3
  %indvars.iv.next.i430 = add nuw nsw i64 %indvars.iv.i425, 1
  %exitcond.not.i431 = icmp eq i64 %indvars.iv.next.i430, %wide.trip.count.i423
  br i1 %exitcond.not.i431, label %for.body.i417.preheader, label %for.body.i432, !llvm.loop !136

for.body.i417.preheader:                          ; preds = %for.body.i432, %middle.block111
  %arrayidx55 = getelementptr inbounds double, double* %0, i64 %mul50
  %mul56 = mul nsw i64 %ik.0447, %conv
  br i1 %min.iters.check99, label %for.body.i417.preheader146, label %vector.body98

vector.body98:                                    ; preds = %for.body.i417.preheader, %vector.body98
  %index103 = phi i64 [ %index.next104, %vector.body98 ], [ 0, %for.body.i417.preheader ]
  %60 = getelementptr inbounds double, double* %arrayidx55, i64 %index103
  %61 = bitcast double* %60 to <2 x double>*
  %wide.load107 = load <2 x double>, <2 x double>* %61, align 8, !tbaa !3
  %62 = getelementptr inbounds double, double* %60, i64 2
  %63 = bitcast double* %62 to <2 x double>*
  %wide.load108 = load <2 x double>, <2 x double>* %63, align 8, !tbaa !3
  %64 = getelementptr inbounds double, double* %4, i64 %index103
  %65 = bitcast double* %64 to <2 x double>*
  %wide.load109 = load <2 x double>, <2 x double>* %65, align 8, !tbaa !3
  %66 = getelementptr inbounds double, double* %64, i64 2
  %67 = bitcast double* %66 to <2 x double>*
  %wide.load110 = load <2 x double>, <2 x double>* %67, align 8, !tbaa !3
  %68 = fmul <2 x double> %wide.load107, %wide.load109
  %69 = fmul <2 x double> %wide.load108, %wide.load110
  %70 = getelementptr inbounds double, double* %6, i64 %index103
  %71 = bitcast double* %70 to <2 x double>*
  store <2 x double> %68, <2 x double>* %71, align 8, !tbaa !3
  %72 = getelementptr inbounds double, double* %70, i64 2
  %73 = bitcast double* %72 to <2 x double>*
  store <2 x double> %69, <2 x double>* %73, align 8, !tbaa !3
  %index.next104 = add i64 %index103, 4
  %74 = icmp eq i64 %index.next104, %n.vec102
  br i1 %74, label %middle.block96, label %vector.body98, !llvm.loop !137

middle.block96:                                   ; preds = %vector.body98
  br i1 %cmp.n106, label %for.body7.i.preheader, label %for.body.i417.preheader146

for.body.i417.preheader146:                       ; preds = %for.body.i417.preheader, %middle.block96
  %indvars.iv69.i.ph = phi i64 [ 0, %for.body.i417.preheader ], [ %n.vec102, %middle.block96 ]
  br label %for.body.i417

for.body.i417:                                    ; preds = %for.body.i417.preheader146, %for.body.i417
  %indvars.iv69.i = phi i64 [ %indvars.iv.next70.i, %for.body.i417 ], [ %indvars.iv69.i.ph, %for.body.i417.preheader146 ]
  %arrayidx.i414 = getelementptr inbounds double, double* %arrayidx55, i64 %indvars.iv69.i
  %75 = load double, double* %arrayidx.i414, align 8, !tbaa !3
  %arrayidx2.i415 = getelementptr inbounds double, double* %4, i64 %indvars.iv69.i
  %76 = load double, double* %arrayidx2.i415, align 8, !tbaa !3
  %mul.i416 = fmul double %75, %76
  %arrayidx4.i = getelementptr inbounds double, double* %6, i64 %indvars.iv69.i
  store double %mul.i416, double* %arrayidx4.i, align 8, !tbaa !3
  %indvars.iv.next70.i = add nuw nsw i64 %indvars.iv69.i, 1
  %exitcond72.not.i = icmp eq i64 %indvars.iv.next70.i, %wide.trip.count.i423
  br i1 %exitcond72.not.i, label %for.body7.i.preheader, label %for.body.i417, !llvm.loop !138

for.body7.i.preheader:                            ; preds = %for.body.i417, %middle.block96
  %arrayidx57 = getelementptr inbounds double, double* %Ls, i64 %mul56
  br label %for.body7.i

for.cond5.loopexit.i:                             ; preds = %for.body13.i, %middle.block72, %for.body7.i
  %indvars.iv.next62.i = add nuw nsw i64 %indvars.iv61.i, 1
  %exitcond68.not.i = icmp eq i64 %indvars.iv.next66.i, %wide.trip.count.i423
  br i1 %exitcond68.not.i, label %cQtimesx.exit, label %for.body7.i, !llvm.loop !45

for.body7.i:                                      ; preds = %for.body7.i.preheader, %for.cond5.loopexit.i
  %indvars.iv65.i = phi i64 [ %indvars.iv.next66.i, %for.cond5.loopexit.i ], [ 0, %for.body7.i.preheader ]
  %indvars.iv61.i = phi i64 [ %indvars.iv.next62.i, %for.cond5.loopexit.i ], [ 1, %for.body7.i.preheader ]
  %i.158.i = phi i32 [ %add.i418, %for.cond5.loopexit.i ], [ 0, %for.body7.i.preheader ]
  %77 = xor i64 %indvars.iv65.i, -1
  %78 = add nsw i64 %77, %wide.trip.count.i423
  %indvars.iv.next66.i = add nuw nsw i64 %indvars.iv65.i, 1
  %add.i418 = add nuw nsw i32 %i.158.i, 1
  %cmp1254.i = icmp ult i64 %indvars.iv.next66.i, %wide.trip.count.i423
  br i1 %cmp1254.i, label %for.body13.lr.ph.i, label %for.cond5.loopexit.i

for.body13.lr.ph.i:                               ; preds = %for.body7.i
  %79 = xor i32 %i.158.i, -1
  %sub9.i = add i32 %mul8.i, %79
  %80 = trunc i64 %indvars.iv65.i to i32
  %mul10.i = mul nsw i32 %sub9.i, %80
  %div.i419 = sdiv i32 %mul10.i, 2
  %arrayidx19.i = getelementptr inbounds double, double* %4, i64 %indvars.iv65.i
  %81 = sext i32 %div.i419 to i64
  %82 = load double, double* %arrayidx19.i, align 8, !tbaa !3
  %min.iters.check75 = icmp ult i64 %78, 4
  br i1 %min.iters.check75, label %for.body13.i.preheader, label %vector.ph76

vector.ph76:                                      ; preds = %for.body13.lr.ph.i
  %n.vec78 = and i64 %78, -4
  %ind.end82 = add i64 %indvars.iv61.i, %n.vec78
  %ind.end84 = add i64 %n.vec78, %81
  %broadcast.splatinsert92 = insertelement <2 x double> poison, double %82, i32 0
  %broadcast.splat93 = shufflevector <2 x double> %broadcast.splatinsert92, <2 x double> poison, <2 x i32> zeroinitializer
  %broadcast.splatinsert94 = insertelement <2 x double> poison, double %82, i32 0
  %broadcast.splat95 = shufflevector <2 x double> %broadcast.splatinsert94, <2 x double> poison, <2 x i32> zeroinitializer
  br label %vector.body74

vector.body74:                                    ; preds = %vector.body74, %vector.ph76
  %index79 = phi i64 [ 0, %vector.ph76 ], [ %index.next80, %vector.body74 ]
  %offset.idx86 = add i64 %indvars.iv61.i, %index79
  %offset.idx87 = add i64 %index79, %81
  %83 = getelementptr inbounds double, double* %6, i64 %offset.idx86
  %84 = bitcast double* %83 to <2 x double>*
  %wide.load88 = load <2 x double>, <2 x double>* %84, align 8, !tbaa !3
  %85 = getelementptr inbounds double, double* %83, i64 2
  %86 = bitcast double* %85 to <2 x double>*
  %wide.load89 = load <2 x double>, <2 x double>* %86, align 8, !tbaa !3
  %87 = getelementptr inbounds double, double* %arrayidx57, i64 %offset.idx87
  %88 = bitcast double* %87 to <2 x double>*
  %wide.load90 = load <2 x double>, <2 x double>* %88, align 8, !tbaa !3
  %89 = getelementptr inbounds double, double* %87, i64 2
  %90 = bitcast double* %89 to <2 x double>*
  %wide.load91 = load <2 x double>, <2 x double>* %90, align 8, !tbaa !3
  %91 = fmul <2 x double> %broadcast.splat93, %wide.load90
  %92 = fmul <2 x double> %broadcast.splat95, %wide.load91
  %93 = fadd <2 x double> %wide.load88, %91
  %94 = fadd <2 x double> %wide.load89, %92
  %95 = bitcast double* %83 to <2 x double>*
  store <2 x double> %93, <2 x double>* %95, align 8, !tbaa !3
  %96 = bitcast double* %85 to <2 x double>*
  store <2 x double> %94, <2 x double>* %96, align 8, !tbaa !3
  %index.next80 = add i64 %index79, 4
  %97 = icmp eq i64 %index.next80, %n.vec78
  br i1 %97, label %middle.block72, label %vector.body74, !llvm.loop !139

middle.block72:                                   ; preds = %vector.body74
  %cmp.n85 = icmp eq i64 %78, %n.vec78
  br i1 %cmp.n85, label %for.cond5.loopexit.i, label %for.body13.i.preheader

for.body13.i.preheader:                           ; preds = %for.body13.lr.ph.i, %middle.block72
  %indvars.iv63.i.ph = phi i64 [ %indvars.iv61.i, %for.body13.lr.ph.i ], [ %ind.end82, %middle.block72 ]
  %indvars.iv.i420.ph = phi i64 [ %81, %for.body13.lr.ph.i ], [ %ind.end84, %middle.block72 ]
  br label %for.body13.i

for.body13.i:                                     ; preds = %for.body13.i.preheader, %for.body13.i
  %indvars.iv63.i = phi i64 [ %indvars.iv.next64.i, %for.body13.i ], [ %indvars.iv63.i.ph, %for.body13.i.preheader ]
  %indvars.iv.i420 = phi i64 [ %indvars.iv.next.i421, %for.body13.i ], [ %indvars.iv.i420.ph, %for.body13.i.preheader ]
  %arrayidx15.i = getelementptr inbounds double, double* %6, i64 %indvars.iv63.i
  %98 = load double, double* %arrayidx15.i, align 8, !tbaa !3
  %arrayidx17.i = getelementptr inbounds double, double* %arrayidx57, i64 %indvars.iv.i420
  %99 = load double, double* %arrayidx17.i, align 8, !tbaa !3
  %mul20.i = fmul double %82, %99
  %add21.i = fadd double %98, %mul20.i
  store double %add21.i, double* %arrayidx15.i, align 8, !tbaa !3
  %indvars.iv.next.i421 = add nsw i64 %indvars.iv.i420, 1
  %indvars.iv.next64.i = add nuw nsw i64 %indvars.iv63.i, 1
  %exitcond.not.i422 = icmp eq i64 %indvars.iv.next64.i, %wide.trip.count.i423
  br i1 %exitcond.not.i422, label %for.cond5.loopexit.i, label %for.body13.i, !llvm.loop !140

cQtimesx.exit:                                    ; preds = %for.cond5.loopexit.i, %for.body45
  %arrayidx62 = getelementptr inbounds double, double* %17, i64 %mul50
  %100 = bitcast double* %arrayidx62 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %100, i8* align 1 %call14, i64 %mul13, i1 false) #14
  %arrayidx71 = getelementptr inbounds double, double* %18, i64 %mul50
  %101 = bitcast double* %arrayidx71 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %101, i8* align 1 %call20, i64 %mul13, i1 false) #14
  %arrayidx78 = getelementptr inbounds double, double* %alphas, i64 %ik.0447
  %102 = load double, double* %arrayidx78, align 8, !tbaa !3
  %arrayidx79 = getelementptr inbounds double, double* %2, i64 %ik.0447
  %103 = load double, double* %arrayidx79, align 8, !tbaa !3
  %add = fadd double %102, %103
  %104 = load double, double* %6, align 8, !tbaa !3
  %mul.i404 = fmul double %104, %104
  br i1 %cmp15.i, label %for.body.i412, label %sqnorm.exit

for.body.i412:                                    ; preds = %cQtimesx.exit, %for.body.i412
  %indvars.iv.i407 = phi i64 [ %indvars.iv.next.i410, %for.body.i412 ], [ 1, %cQtimesx.exit ]
  %res.017.i = phi double [ %add.i409, %for.body.i412 ], [ %mul.i404, %cQtimesx.exit ]
  %arrayidx2.i408 = getelementptr inbounds double, double* %6, i64 %indvars.iv.i407
  %105 = load double, double* %arrayidx2.i408, align 8, !tbaa !3
  %mul5.i = fmul double %105, %105
  %add.i409 = fadd double %res.017.i, %mul5.i
  %indvars.iv.next.i410 = add nuw nsw i64 %indvars.iv.i407, 1
  %exitcond.not.i411 = icmp eq i64 %indvars.iv.next.i410, %wide.trip.count.i423
  br i1 %exitcond.not.i411, label %sqnorm.exit, label %for.body.i412, !llvm.loop !10

sqnorm.exit:                                      ; preds = %for.body.i412, %cQtimesx.exit
  %res.0.lcssa.i = phi double [ %mul.i404, %cQtimesx.exit ], [ %add.i409, %for.body.i412 ]
  %mul82 = fmul double %res.0.lcssa.i, 5.000000e-01
  %sub83 = fsub double %add, %mul82
  %arrayidx84 = getelementptr inbounds double, double* %8, i64 %ik.0447
  store double %sub83, double* %arrayidx84, align 8, !tbaa !3
  %inc = add nuw nsw i64 %ik.0447, 1
  %exitcond465.not = icmp eq i64 %inc, %conv7
  br i1 %exitcond465.not, label %for.end.loopexit, label %for.body45, !llvm.loop !141

for.end.loopexit:                                 ; preds = %sqnorm.exit
  %.pre = load double, double* %8, align 8, !tbaa !3
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %for.cond41.preheader
  %106 = phi double [ %.pre, %for.end.loopexit ], [ %21, %for.cond41.preheader ]
  br i1 %cmp13.i.i.i, label %for.body.i.i.i, label %arr_max.exit.i.i

for.body.i.i.i:                                   ; preds = %for.end, %for.body.i.i.i
  %indvars.iv.i.i.i = phi i64 [ %indvars.iv.next.i.i.i, %for.body.i.i.i ], [ 1, %for.end ]
  %m.015.i.i.i = phi double [ %m.1.i.i.i, %for.body.i.i.i ], [ %106, %for.end ]
  %arrayidx1.i.i.i = getelementptr inbounds double, double* %8, i64 %indvars.iv.i.i.i
  %107 = load double, double* %arrayidx1.i.i.i, align 8, !tbaa !3
  %cmp2.i.i.i = fcmp olt double %m.015.i.i.i, %107
  %m.1.i.i.i = select i1 %cmp2.i.i.i, double %107, double %m.015.i.i.i
  %indvars.iv.next.i.i.i = add nuw nsw i64 %indvars.iv.i.i.i, 1
  %exitcond.not.i.i.i = icmp eq i64 %indvars.iv.next.i.i.i, %wide.trip.count.i.i.i
  br i1 %exitcond.not.i.i.i, label %arr_max.exit.i.i, label %for.body.i.i.i, !llvm.loop !7

arr_max.exit.i.i:                                 ; preds = %for.body.i.i.i, %for.end
  %m.0.lcssa.i.i.i = phi double [ %106, %for.end ], [ %m.1.i.i.i, %for.body.i.i.i ]
  br i1 %cmp37.i, label %for.body.preheader.i.i, label %log_sum_exp.exit.i

for.body.preheader.i.i:                           ; preds = %arr_max.exit.i.i
  %sub.i12.i = fsub double %106, %m.0.lcssa.i.i.i
  %108 = tail call double @llvm.exp.f64(double %sub.i12.i) #14
  %add.i13.i = fadd double %108, 0.000000e+00
  br i1 %exitcond.not.i14.i, label %cgrad_log_sum_exp.exit.thread, label %for.body.for.body_crit_edge.i.i, !llvm.loop !22

cgrad_log_sum_exp.exit.thread:                    ; preds = %for.body.preheader.i.i
  %109 = tail call double @llvm.log.f64(double %add.i13.i) #14
  %add1.i21.i439472 = fadd double %m.0.lcssa.i.i.i, %109
  %sub.i449473 = fsub double %106, %add1.i21.i439472
  %110 = tail call double @llvm.exp.f64(double %sub.i449473) #14
  store double %110, double* %9, align 8, !tbaa !3
  br label %for.body89.preheader

for.body.for.body_crit_edge.i.i:                  ; preds = %for.body.preheader.i.i, %for.body.for.body_crit_edge.i.i
  %indvars.iv.next.i16.i = phi i64 [ %indvars.iv.next.i.i, %for.body.for.body_crit_edge.i.i ], [ 1, %for.body.preheader.i.i ]
  %add.i15.i = phi double [ %add.i.i, %for.body.for.body_crit_edge.i.i ], [ %add.i13.i, %for.body.preheader.i.i ]
  %arrayidx.phi.trans.insert.i.i = getelementptr inbounds double, double* %8, i64 %indvars.iv.next.i16.i
  %.pre.i.i = load double, double* %arrayidx.phi.trans.insert.i.i, align 8, !tbaa !3
  %sub.i.i = fsub double %.pre.i.i, %m.0.lcssa.i.i.i
  %111 = tail call double @llvm.exp.f64(double %sub.i.i) #14
  %add.i.i = fadd double %add.i15.i, %111
  %indvars.iv.next.i.i = add nuw nsw i64 %indvars.iv.next.i16.i, 1
  %exitcond.not.i.i = icmp eq i64 %indvars.iv.next.i.i, %wide.trip.count.i.i.i
  br i1 %exitcond.not.i.i, label %log_sum_exp.exit.i, label %for.body.for.body_crit_edge.i.i, !llvm.loop !22

log_sum_exp.exit.i:                               ; preds = %for.body.for.body_crit_edge.i.i, %arr_max.exit.i.i
  %semx.0.lcssa.i.i = phi double [ 0.000000e+00, %arr_max.exit.i.i ], [ %add.i.i, %for.body.for.body_crit_edge.i.i ]
  br i1 %cmp10.not.i, label %cgrad_log_sum_exp.exit, label %112

112:                                              ; preds = %log_sum_exp.exit.i
  %113 = tail call double @llvm.log.f64(double %semx.0.lcssa.i.i) #14
  %add1.i21.i439 = fadd double %m.0.lcssa.i.i.i, %113
  %sub.i449 = fsub double %106, %add1.i21.i439
  %114 = tail call double @llvm.exp.f64(double %sub.i449) #14
  store double %114, double* %9, align 8, !tbaa !3
  br i1 %exitcond.not.i401450, label %cgrad_log_sum_exp.exit, label %for.body.for.body_crit_edge.i.preheader, !llvm.loop !124

for.body.for.body_crit_edge.i.preheader:          ; preds = %112
  br i1 %min.iters.check59, label %for.body.for.body_crit_edge.i.preheader149, label %vector.ph60

vector.ph60:                                      ; preds = %for.body.for.body_crit_edge.i.preheader
  %broadcast.splatinsert70 = insertelement <2 x double> poison, double %add1.i21.i439, i32 0
  %broadcast.splat71 = shufflevector <2 x double> %broadcast.splatinsert70, <2 x double> poison, <2 x i32> zeroinitializer
  br label %vector.body58

vector.body58:                                    ; preds = %vector.body58, %vector.ph60
  %index63 = phi i64 [ 0, %vector.ph60 ], [ %index.next64, %vector.body58 ]
  %offset.idx68 = or i64 %index63, 1
  %115 = getelementptr inbounds double, double* %8, i64 %offset.idx68
  %116 = bitcast double* %115 to <2 x double>*
  %wide.load69 = load <2 x double>, <2 x double>* %116, align 8, !tbaa !3
  %117 = fsub <2 x double> %wide.load69, %broadcast.splat71
  %118 = call <2 x double> @llvm.exp.v2f64(<2 x double> %117)
  %119 = getelementptr inbounds double, double* %9, i64 %offset.idx68
  %120 = bitcast double* %119 to <2 x double>*
  store <2 x double> %118, <2 x double>* %120, align 8, !tbaa !3
  %index.next64 = add i64 %index63, 2
  %121 = icmp eq i64 %index.next64, %n.vec62
  br i1 %121, label %middle.block56, label %vector.body58, !llvm.loop !142

middle.block56:                                   ; preds = %vector.body58
  br i1 %cmp.n67, label %cgrad_log_sum_exp.exit, label %for.body.for.body_crit_edge.i.preheader149

for.body.for.body_crit_edge.i.preheader149:       ; preds = %for.body.for.body_crit_edge.i.preheader, %middle.block56
  %inc.i451.ph = phi i64 [ 1, %for.body.for.body_crit_edge.i.preheader ], [ %ind.end66, %middle.block56 ]
  br label %for.body.for.body_crit_edge.i

for.body.for.body_crit_edge.i:                    ; preds = %for.body.for.body_crit_edge.i.preheader149, %for.body.for.body_crit_edge.i
  %inc.i451 = phi i64 [ %inc.i, %for.body.for.body_crit_edge.i ], [ %inc.i451.ph, %for.body.for.body_crit_edge.i.preheader149 ]
  %arrayidx.phi.trans.insert.i = getelementptr inbounds double, double* %8, i64 %inc.i451
  %.pre.i403 = load double, double* %arrayidx.phi.trans.insert.i, align 8, !tbaa !3
  %sub.i = fsub double %.pre.i403, %add1.i21.i439
  %122 = tail call double @llvm.exp.f64(double %sub.i) #14
  %arrayidx2.i = getelementptr inbounds double, double* %9, i64 %inc.i451
  store double %122, double* %arrayidx2.i, align 8, !tbaa !3
  %inc.i = add nuw i64 %inc.i451, 1
  %exitcond.not.i401 = icmp eq i64 %inc.i, %spec.select
  br i1 %exitcond.not.i401, label %cgrad_log_sum_exp.exit, label %for.body.for.body_crit_edge.i, !llvm.loop !143

cgrad_log_sum_exp.exit:                           ; preds = %for.body.for.body_crit_edge.i, %middle.block56, %112, %log_sum_exp.exit.i
  br i1 %cmp37.i, label %for.body89.preheader, label %for.inc169

for.body89.preheader:                             ; preds = %cgrad_log_sum_exp.exit, %cgrad_log_sum_exp.exit.thread
  br i1 %min.iters.check45, label %for.body89.preheader148, label %vector.body44

vector.body44:                                    ; preds = %for.body89.preheader, %vector.body44
  %index49 = phi i64 [ %index.next50, %vector.body44 ], [ 0, %for.body89.preheader ]
  %123 = getelementptr inbounds double, double* %9, i64 %index49
  %124 = bitcast double* %123 to <2 x double>*
  %wide.load53 = load <2 x double>, <2 x double>* %124, align 8, !tbaa !3
  %125 = getelementptr inbounds double, double* %alphasb, i64 %index49
  %126 = bitcast double* %125 to <2 x double>*
  %wide.load54 = load <2 x double>, <2 x double>* %126, align 8, !tbaa !3
  %127 = fadd <2 x double> %wide.load53, %wide.load54
  %128 = bitcast double* %125 to <2 x double>*
  store <2 x double> %127, <2 x double>* %128, align 8, !tbaa !3
  %129 = getelementptr inbounds double, double* %3, i64 %index49
  %130 = bitcast double* %129 to <2 x double>*
  %wide.load55 = load <2 x double>, <2 x double>* %130, align 8, !tbaa !3
  %131 = fadd <2 x double> %wide.load53, %wide.load55
  %132 = bitcast double* %129 to <2 x double>*
  store <2 x double> %131, <2 x double>* %132, align 8, !tbaa !3
  %index.next50 = add i64 %index49, 2
  %133 = icmp eq i64 %index.next50, %n.vec48
  br i1 %133, label %middle.block42, label %vector.body44, !llvm.loop !144

middle.block42:                                   ; preds = %vector.body44
  br i1 %cmp.n52, label %for.cond99.preheader, label %for.body89.preheader148

for.body89.preheader148:                          ; preds = %for.body89.preheader, %middle.block42
  %ik.1453.ph = phi i64 [ 0, %for.body89.preheader ], [ %n.vec48, %middle.block42 ]
  br label %for.body89

for.cond99.preheader:                             ; preds = %for.body89, %middle.block42
  br i1 %cmp37.i, label %for.body103, label %for.inc169

for.body89:                                       ; preds = %for.body89.preheader148, %for.body89
  %ik.1453 = phi i64 [ %inc97, %for.body89 ], [ %ik.1453.ph, %for.body89.preheader148 ]
  %arrayidx90 = getelementptr inbounds double, double* %9, i64 %ik.1453
  %134 = load double, double* %arrayidx90, align 8, !tbaa !3
  %arrayidx91 = getelementptr inbounds double, double* %alphasb, i64 %ik.1453
  %135 = load double, double* %arrayidx91, align 8, !tbaa !3
  %add92 = fadd double %134, %135
  store double %add92, double* %arrayidx91, align 8, !tbaa !3
  %arrayidx94 = getelementptr inbounds double, double* %3, i64 %ik.1453
  %136 = load double, double* %arrayidx94, align 8, !tbaa !3
  %add95 = fadd double %134, %136
  store double %add95, double* %arrayidx94, align 8, !tbaa !3
  %inc97 = add nuw nsw i64 %ik.1453, 1
  %exitcond466.not = icmp eq i64 %inc97, %conv7
  br i1 %exitcond466.not, label %for.cond99.preheader, label %for.body89, !llvm.loop !145

for.body103:                                      ; preds = %for.cond99.preheader, %for.cond.cleanup147
  %ik.2459 = phi i64 [ %inc167, %for.cond.cleanup147 ], [ 0, %for.cond99.preheader ]
  %mul105 = mul nsw i64 %ik.2459, %conv12
  %arrayidx106 = getelementptr inbounds double, double* %17, i64 %mul105
  %137 = bitcast double* %arrayidx106 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %call14, i8* align 1 %137, i64 %mul13, i1 false) #14
  %arrayidx112 = getelementptr inbounds double, double* %18, i64 %mul105
  %138 = bitcast double* %arrayidx112 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %call20, i8* align 1 %138, i64 %mul13, i1 false) #14
  br i1 %cmp10.i, label %for.body120.lr.ph, label %for.cond.cleanup147

for.body120.lr.ph:                                ; preds = %for.body103
  %arrayidx121 = getelementptr inbounds double, double* %9, i64 %ik.2459
  %139 = load double, double* %arrayidx121, align 8, !tbaa !3
  %fneg = fneg double %139
  br i1 %min.iters.check30, label %for.body120.preheader, label %vector.ph31

vector.ph31:                                      ; preds = %for.body120.lr.ph
  %broadcast.splatinsert39 = insertelement <2 x double> poison, double %fneg, i32 0
  %broadcast.splat40 = shufflevector <2 x double> %broadcast.splatinsert39, <2 x double> poison, <2 x i32> zeroinitializer
  br label %vector.body29

vector.body29:                                    ; preds = %vector.body29, %vector.ph31
  %index34 = phi i64 [ 0, %vector.ph31 ], [ %index.next35, %vector.body29 ]
  %140 = getelementptr inbounds double, double* %6, i64 %index34
  %141 = bitcast double* %140 to <2 x double>*
  %wide.load38 = load <2 x double>, <2 x double>* %141, align 8, !tbaa !3
  %142 = fmul <2 x double> %wide.load38, %broadcast.splat40
  %143 = getelementptr inbounds double, double* %7, i64 %index34
  %144 = bitcast double* %143 to <2 x double>*
  store <2 x double> %142, <2 x double>* %144, align 8, !tbaa !3
  %145 = add nsw i64 %index34, %mul105
  %146 = getelementptr inbounds double, double* %0, i64 %145
  %147 = bitcast double* %146 to <2 x double>*
  %wide.load41 = load <2 x double>, <2 x double>* %147, align 8, !tbaa !3
  %148 = fmul <2 x double> %142, %wide.load41
  %149 = getelementptr inbounds double, double* %5, i64 %index34
  %150 = bitcast double* %149 to <2 x double>*
  store <2 x double> %148, <2 x double>* %150, align 8, !tbaa !3
  %index.next35 = add i64 %index34, 2
  %151 = icmp eq i64 %index.next35, %n.vec33
  br i1 %151, label %middle.block27, label %vector.body29, !llvm.loop !146

middle.block27:                                   ; preds = %vector.body29
  br i1 %cmp.n37, label %for.cond.cleanup, label %for.body120.preheader

for.body120.preheader:                            ; preds = %for.body120.lr.ph, %middle.block27
  %i.0455.ph = phi i64 [ 0, %for.body120.lr.ph ], [ %n.vec33, %middle.block27 ]
  br label %for.body120

for.cond.cleanup:                                 ; preds = %for.body120, %middle.block27
  %mul135 = mul nsw i64 %ik.2459, %conv
  %arrayidx136 = getelementptr inbounds double, double* %Ls, i64 %mul135
  br label %for.body.i388

for.cond.loopexit.i385.loopexit:                  ; preds = %for.body6.i400
  store double %add12.i397, double* %arrayidx.i392, align 8, !tbaa !3
  br label %for.cond.loopexit.i385

for.cond.loopexit.i385:                           ; preds = %for.body.i388, %for.cond.loopexit.i385.loopexit
  %indvars.iv.next.i384 = add nuw nsw i64 %indvars.iv.i386, 1
  %exitcond46.not.i = icmp eq i64 %indvars.iv.next44.i, %wide.trip.count.i423
  br i1 %exitcond46.not.i, label %for.body.i378.preheader, label %for.body.i388, !llvm.loop !111

for.body.i388:                                    ; preds = %for.cond.cleanup, %for.cond.loopexit.i385
  %indvars.iv43.i = phi i64 [ %indvars.iv.next44.i, %for.cond.loopexit.i385 ], [ 0, %for.cond.cleanup ]
  %indvars.iv.i386 = phi i64 [ %indvars.iv.next.i384, %for.cond.loopexit.i385 ], [ 1, %for.cond.cleanup ]
  %i.038.i = phi i32 [ %add.i387, %for.cond.loopexit.i385 ], [ 0, %for.cond.cleanup ]
  %indvars.iv.next44.i = add nuw nsw i64 %indvars.iv43.i, 1
  %add.i387 = add nuw nsw i32 %i.038.i, 1
  %cmp434.i = icmp ult i64 %indvars.iv.next44.i, %wide.trip.count.i423
  br i1 %cmp434.i, label %for.body6.lr.ph.i394, label %for.cond.loopexit.i385

for.body6.lr.ph.i394:                             ; preds = %for.body.i388
  %152 = xor i32 %i.038.i, -1
  %sub1.i389 = add i32 %mul8.i, %152
  %153 = trunc i64 %indvars.iv43.i to i32
  %mul2.i390 = mul nsw i32 %sub1.i389, %153
  %div.i391 = sdiv i32 %mul2.i390, 2
  %arrayidx.i392 = getelementptr inbounds double, double* %5, i64 %indvars.iv43.i
  %154 = sext i32 %div.i391 to i64
  %.pre.i393 = load double, double* %arrayidx.i392, align 8, !tbaa !3
  br label %for.body6.i400

for.body6.i400:                                   ; preds = %for.body6.i400, %for.body6.lr.ph.i394
  %155 = phi double [ %.pre.i393, %for.body6.lr.ph.i394 ], [ %add12.i397, %for.body6.i400 ]
  %indvars.iv41.i = phi i64 [ %154, %for.body6.lr.ph.i394 ], [ %indvars.iv.next42.i, %for.body6.i400 ]
  %indvars.iv39.i395 = phi i64 [ %indvars.iv.i386, %for.body6.lr.ph.i394 ], [ %indvars.iv.next40.i398, %for.body6.i400 ]
  %arrayidx8.i396 = getelementptr inbounds double, double* %arrayidx136, i64 %indvars.iv41.i
  %156 = load double, double* %arrayidx8.i396, align 8, !tbaa !3
  %arrayidx10.i = getelementptr inbounds double, double* %7, i64 %indvars.iv39.i395
  %157 = load double, double* %arrayidx10.i, align 8, !tbaa !3
  %mul11.i = fmul double %156, %157
  %add12.i397 = fadd double %155, %mul11.i
  %indvars.iv.next42.i = add nsw i64 %indvars.iv41.i, 1
  %indvars.iv.next40.i398 = add nuw nsw i64 %indvars.iv39.i395, 1
  %exitcond.not.i399 = icmp eq i64 %indvars.iv.next40.i398, %wide.trip.count.i423
  br i1 %exitcond.not.i399, label %for.cond.loopexit.i385.loopexit, label %for.body6.i400, !llvm.loop !112

for.body.i378.preheader:                          ; preds = %for.cond.loopexit.i385
  %arrayidx141 = getelementptr inbounds double, double* %Lsb, i64 %mul135
  br label %for.body.i378

for.cond.loopexit.i:                              ; preds = %for.body6.i, %middle.block7, %for.body.i378
  %indvars.iv.next.i376 = add nuw nsw i64 %indvars.iv.i377, 1
  %exitcond42.not.i = icmp eq i64 %indvars.iv.next40.i, %wide.trip.count.i423
  br i1 %exitcond42.not.i, label %for.body148.preheader, label %for.body.i378, !llvm.loop !113

for.body148.preheader:                            ; preds = %for.cond.loopexit.i
  br i1 %min.iters.check, label %for.body148.preheader145, label %vector.body

vector.body:                                      ; preds = %for.body148.preheader, %vector.body
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %for.body148.preheader ]
  %158 = getelementptr inbounds double, double* %5, i64 %index
  %159 = bitcast double* %158 to <2 x double>*
  %wide.load = load <2 x double>, <2 x double>* %159, align 8, !tbaa !3
  %160 = add nsw i64 %index, %mul105
  %161 = getelementptr inbounds double, double* %meansb, i64 %160
  %162 = bitcast double* %161 to <2 x double>*
  %wide.load3 = load <2 x double>, <2 x double>* %162, align 8, !tbaa !3
  %163 = fsub <2 x double> %wide.load3, %wide.load
  %164 = bitcast double* %161 to <2 x double>*
  store <2 x double> %163, <2 x double>* %164, align 8, !tbaa !3
  %165 = getelementptr inbounds double, double* %7, i64 %index
  %166 = bitcast double* %165 to <2 x double>*
  %wide.load4 = load <2 x double>, <2 x double>* %166, align 8, !tbaa !3
  %167 = getelementptr inbounds double, double* %4, i64 %index
  %168 = bitcast double* %167 to <2 x double>*
  %wide.load5 = load <2 x double>, <2 x double>* %168, align 8, !tbaa !3
  %169 = fmul <2 x double> %wide.load4, %wide.load5
  %170 = getelementptr inbounds double, double* %1, i64 %160
  %171 = bitcast double* %170 to <2 x double>*
  %wide.load6 = load <2 x double>, <2 x double>* %171, align 8, !tbaa !3
  %172 = fadd <2 x double> %wide.load6, %169
  %173 = bitcast double* %170 to <2 x double>*
  store <2 x double> %172, <2 x double>* %173, align 8, !tbaa !3
  %index.next = add i64 %index, 2
  %174 = icmp eq i64 %index.next, %n.vec
  br i1 %174, label %middle.block, label %vector.body, !llvm.loop !147

middle.block:                                     ; preds = %vector.body
  br i1 %cmp.n, label %for.cond.cleanup147, label %for.body148.preheader145

for.body148.preheader145:                         ; preds = %for.body148.preheader, %middle.block
  %i142.0457.ph = phi i64 [ 0, %for.body148.preheader ], [ %n.vec, %middle.block ]
  br label %for.body148

for.body.i378:                                    ; preds = %for.cond.loopexit.i, %for.body.i378.preheader
  %indvars.iv39.i = phi i64 [ %indvars.iv.next40.i, %for.cond.loopexit.i ], [ 0, %for.body.i378.preheader ]
  %indvars.iv.i377 = phi i64 [ %indvars.iv.next.i376, %for.cond.loopexit.i ], [ 1, %for.body.i378.preheader ]
  %i.034.i = phi i32 [ %add.i, %for.cond.loopexit.i ], [ 0, %for.body.i378.preheader ]
  %175 = xor i64 %indvars.iv39.i, -1
  %176 = add nsw i64 %175, %wide.trip.count.i423
  %indvars.iv.next40.i = add nuw nsw i64 %indvars.iv39.i, 1
  %add.i = add nuw nsw i32 %i.034.i, 1
  %cmp430.i = icmp ult i64 %indvars.iv.next40.i, %wide.trip.count.i423
  br i1 %cmp430.i, label %for.body6.lr.ph.i, label %for.cond.loopexit.i

for.body6.lr.ph.i:                                ; preds = %for.body.i378
  %177 = xor i32 %i.034.i, -1
  %sub1.i = add i32 %mul8.i, %177
  %178 = trunc i64 %indvars.iv39.i to i32
  %mul2.i = mul nsw i32 %sub1.i, %178
  %div.i = sdiv i32 %mul2.i, 2
  %arrayidx8.i = getelementptr inbounds double, double* %arrayidx106, i64 %indvars.iv39.i
  %179 = sext i32 %div.i to i64
  %180 = load double, double* %arrayidx8.i, align 8, !tbaa !3
  %min.iters.check10 = icmp ult i64 %176, 4
  br i1 %min.iters.check10, label %for.body6.i.preheader, label %vector.ph11

vector.ph11:                                      ; preds = %for.body6.lr.ph.i
  %n.vec13 = and i64 %176, -4
  %ind.end = add i64 %n.vec13, %179
  %ind.end18 = add i64 %indvars.iv.i377, %n.vec13
  %broadcast.splatinsert = insertelement <2 x double> poison, double %180, i32 0
  %broadcast.splat = shufflevector <2 x double> %broadcast.splatinsert, <2 x double> poison, <2 x i32> zeroinitializer
  %broadcast.splatinsert23 = insertelement <2 x double> poison, double %180, i32 0
  %broadcast.splat24 = shufflevector <2 x double> %broadcast.splatinsert23, <2 x double> poison, <2 x i32> zeroinitializer
  br label %vector.body9

vector.body9:                                     ; preds = %vector.body9, %vector.ph11
  %index14 = phi i64 [ 0, %vector.ph11 ], [ %index.next15, %vector.body9 ]
  %offset.idx = add i64 %index14, %179
  %offset.idx20 = add i64 %indvars.iv.i377, %index14
  %181 = getelementptr inbounds double, double* %7, i64 %offset.idx20
  %182 = bitcast double* %181 to <2 x double>*
  %wide.load21 = load <2 x double>, <2 x double>* %182, align 8, !tbaa !3
  %183 = getelementptr inbounds double, double* %181, i64 2
  %184 = bitcast double* %183 to <2 x double>*
  %wide.load22 = load <2 x double>, <2 x double>* %184, align 8, !tbaa !3
  %185 = fmul <2 x double> %broadcast.splat, %wide.load21
  %186 = fmul <2 x double> %broadcast.splat24, %wide.load22
  %187 = getelementptr inbounds double, double* %arrayidx141, i64 %offset.idx
  %188 = bitcast double* %187 to <2 x double>*
  %wide.load25 = load <2 x double>, <2 x double>* %188, align 8, !tbaa !3
  %189 = getelementptr inbounds double, double* %187, i64 2
  %190 = bitcast double* %189 to <2 x double>*
  %wide.load26 = load <2 x double>, <2 x double>* %190, align 8, !tbaa !3
  %191 = fadd <2 x double> %wide.load25, %185
  %192 = fadd <2 x double> %wide.load26, %186
  %193 = bitcast double* %187 to <2 x double>*
  store <2 x double> %191, <2 x double>* %193, align 8, !tbaa !3
  %194 = bitcast double* %189 to <2 x double>*
  store <2 x double> %192, <2 x double>* %194, align 8, !tbaa !3
  %index.next15 = add i64 %index14, 4
  %195 = icmp eq i64 %index.next15, %n.vec13
  br i1 %195, label %middle.block7, label %vector.body9, !llvm.loop !148

middle.block7:                                    ; preds = %vector.body9
  %cmp.n19 = icmp eq i64 %176, %n.vec13
  br i1 %cmp.n19, label %for.cond.loopexit.i, label %for.body6.i.preheader

for.body6.i.preheader:                            ; preds = %for.body6.lr.ph.i, %middle.block7
  %indvars.iv37.i.ph = phi i64 [ %179, %for.body6.lr.ph.i ], [ %ind.end, %middle.block7 ]
  %indvars.iv35.i.ph = phi i64 [ %indvars.iv.i377, %for.body6.lr.ph.i ], [ %ind.end18, %middle.block7 ]
  br label %for.body6.i

for.body6.i:                                      ; preds = %for.body6.i.preheader, %for.body6.i
  %indvars.iv37.i = phi i64 [ %indvars.iv.next38.i, %for.body6.i ], [ %indvars.iv37.i.ph, %for.body6.i.preheader ]
  %indvars.iv35.i = phi i64 [ %indvars.iv.next36.i, %for.body6.i ], [ %indvars.iv35.i.ph, %for.body6.i.preheader ]
  %arrayidx.i379 = getelementptr inbounds double, double* %7, i64 %indvars.iv35.i
  %196 = load double, double* %arrayidx.i379, align 8, !tbaa !3
  %mul9.i = fmul double %180, %196
  %arrayidx11.i = getelementptr inbounds double, double* %arrayidx141, i64 %indvars.iv37.i
  %197 = load double, double* %arrayidx11.i, align 8, !tbaa !3
  %add12.i = fadd double %197, %mul9.i
  store double %add12.i, double* %arrayidx11.i, align 8, !tbaa !3
  %indvars.iv.next38.i = add nsw i64 %indvars.iv37.i, 1
  %indvars.iv.next36.i = add nuw nsw i64 %indvars.iv35.i, 1
  %exitcond.not.i380 = icmp eq i64 %indvars.iv.next36.i, %wide.trip.count.i423
  br i1 %exitcond.not.i380, label %for.cond.loopexit.i, label %for.body6.i, !llvm.loop !149

for.body120:                                      ; preds = %for.body120.preheader, %for.body120
  %i.0455 = phi i64 [ %inc133, %for.body120 ], [ %i.0455.ph, %for.body120.preheader ]
  %arrayidx122 = getelementptr inbounds double, double* %6, i64 %i.0455
  %198 = load double, double* %arrayidx122, align 8, !tbaa !3
  %mul123 = fmul double %198, %fneg
  %arrayidx124 = getelementptr inbounds double, double* %7, i64 %i.0455
  store double %mul123, double* %arrayidx124, align 8, !tbaa !3
  %add127 = add nsw i64 %i.0455, %mul105
  %arrayidx128 = getelementptr inbounds double, double* %0, i64 %add127
  %199 = load double, double* %arrayidx128, align 8, !tbaa !3
  %mul130 = fmul double %mul123, %199
  %arrayidx131 = getelementptr inbounds double, double* %5, i64 %i.0455
  store double %mul130, double* %arrayidx131, align 8, !tbaa !3
  %inc133 = add nuw nsw i64 %i.0455, 1
  %exitcond467.not = icmp eq i64 %inc133, %conv12
  br i1 %exitcond467.not, label %for.cond.cleanup, label %for.body120, !llvm.loop !150

for.cond.cleanup147:                              ; preds = %for.body148, %middle.block, %for.body103
  %inc167 = add nuw nsw i64 %ik.2459, 1
  %exitcond469.not = icmp eq i64 %inc167, %conv7
  br i1 %exitcond469.not, label %for.inc169, label %for.body103, !llvm.loop !151

for.body148:                                      ; preds = %for.body148.preheader145, %for.body148
  %i142.0457 = phi i64 [ %inc164, %for.body148 ], [ %i142.0457.ph, %for.body148.preheader145 ]
  %arrayidx149 = getelementptr inbounds double, double* %5, i64 %i142.0457
  %200 = load double, double* %arrayidx149, align 8, !tbaa !3
  %add152 = add nsw i64 %i142.0457, %mul105
  %arrayidx153 = getelementptr inbounds double, double* %meansb, i64 %add152
  %201 = load double, double* %arrayidx153, align 8, !tbaa !3
  %sub154 = fsub double %201, %200
  store double %sub154, double* %arrayidx153, align 8, !tbaa !3
  %arrayidx155 = getelementptr inbounds double, double* %7, i64 %i142.0457
  %202 = load double, double* %arrayidx155, align 8, !tbaa !3
  %arrayidx156 = getelementptr inbounds double, double* %4, i64 %i142.0457
  %203 = load double, double* %arrayidx156, align 8, !tbaa !3
  %mul157 = fmul double %202, %203
  %arrayidx161 = getelementptr inbounds double, double* %1, i64 %add152
  %204 = load double, double* %arrayidx161, align 8, !tbaa !3
  %add162 = fadd double %204, %mul157
  store double %add162, double* %arrayidx161, align 8, !tbaa !3
  %inc164 = add nuw nsw i64 %i142.0457, 1
  %exitcond468.not = icmp eq i64 %inc164, %conv12
  br i1 %exitcond468.not, label %for.cond.cleanup147, label %for.body148, !llvm.loop !152

for.inc169:                                       ; preds = %for.cond.cleanup147, %for.cond99.preheader, %cgrad_log_sum_exp.exit
  %inc170 = add nuw nsw i64 %ix.0462, 1
  %exitcond470.not = icmp eq i64 %inc170, %conv39
  br i1 %exitcond470.not, label %for.cond173.preheader, label %for.cond41.preheader, !llvm.loop !153

for.cond.cleanup177:                              ; preds = %for.cond.cleanup183.loopexit.us, %for.cond173.preheader
  tail call void @free(i8* %call34)
  tail call void @free(i8* %call38)
  tail call void @free(i8* %call)
  tail call void @free(i8* %call6)
  tail call void @free(i8* %call9)
  tail call void @free(i8* %call11)
  tail call void @free(i8* %call14)
  tail call void @free(i8* %call17)
  tail call void @free(i8* %call20)
  tail call void @free(i8* %call23)
  tail call void @free(i8* %call26)
  tail call void @free(i8* %call29)
  ret void
}

; Function Attrs: inaccessiblememonly nofree nounwind willreturn allocsize(0,1)
declare noalias noundef i8* @calloc(i64, i64) local_unnamed_addr #8

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #9

; Function Attrs: nounwind ssp uwtable
define dso_local void @preprocess_ec_main_term(i32 %d, i32 %k, i32 %n, double* noalias nocapture readonly %alphas, double* noalias nocapture readonly %means, double* noalias nocapture readonly %Qs, double* noalias nocapture readonly %Ls, double* noalias nocapture readonly %x, double* nocapture %err) local_unnamed_addr #5 {
entry:
  %sub = add nsw i32 %d, -1
  %mul = mul nsw i32 %sub, %d
  %div = sdiv i32 %mul, 2
  %conv = sext i32 %div to i64
  %mul1 = mul nsw i32 %k, %d
  %conv2 = sext i32 %mul1 to i64
  %mul3 = shl nsw i64 %conv2, 3
  %call = tail call i8* @malloc(i64 %mul3) #13
  %0 = bitcast i8* %call to double*
  %conv4 = sext i32 %k to i64
  %mul5 = shl nsw i64 %conv4, 3
  %call6 = tail call i8* @malloc(i64 %mul5) #13
  %1 = bitcast i8* %call6 to double*
  %conv7 = sext i32 %d to i64
  %mul8 = shl nsw i64 %conv7, 3
  %call9 = tail call i8* @malloc(i64 %mul8) #13
  %2 = bitcast i8* %call9 to double*
  %call12 = tail call i8* @malloc(i64 %mul8) #13
  %3 = bitcast i8* %call12 to double*
  %call15 = tail call i8* @malloc(i64 %mul5) #13
  %4 = bitcast i8* %call15 to double*
  %cmp37.i = icmp sgt i32 %k, 0
  br i1 %cmp37.i, label %for.body.lr.ph.i, label %preprocess_qs.exit

for.body.lr.ph.i:                                 ; preds = %entry
  %cmp235.i = icmp sgt i32 %d, 0
  %wide.trip.count44.i = zext i32 %k to i64
  %wide.trip.count.i = zext i32 %d to i64
  br i1 %cmp235.i, label %for.body.i.us, label %for.body.i.preheader

for.body.i.preheader:                             ; preds = %for.body.lr.ph.i
  %5 = shl nuw nsw i64 %wide.trip.count44.i, 3
  call void @llvm.memset.p0i8.i64(i8* align 8 %call6, i8 0, i64 %5, i1 false)
  br label %preprocess_qs.exit

for.body.i.us:                                    ; preds = %for.body.lr.ph.i, %for.inc15.i.loopexit.us
  %tiv1.us = phi i64 [ %tiv.next2.us, %for.inc15.i.loopexit.us ], [ 0, %for.body.lr.ph.i ]
  %arrayidx.i.us = getelementptr inbounds double, double* %1, i64 %tiv1.us
  store double 0.000000e+00, double* %arrayidx.i.us, align 8, !tbaa !3
  %6 = trunc i64 %tiv1.us to i32
  %mul.i.us = mul nsw i32 %6, %d
  %7 = sext i32 %mul.i.us to i64
  br label %for.body3.i.us

for.body3.i.us:                                   ; preds = %for.body3.i.us, %for.body.i.us
  %8 = phi double [ 0.000000e+00, %for.body.i.us ], [ %add8.i.us, %for.body3.i.us ]
  %indvars.iv.i.us = phi i64 [ 0, %for.body.i.us ], [ %indvars.iv.next.i.us, %for.body3.i.us ]
  %9 = add nsw i64 %indvars.iv.i.us, %7
  %arrayidx5.i.us = getelementptr inbounds double, double* %Qs, i64 %9
  %10 = load double, double* %arrayidx5.i.us, align 8, !tbaa !3
  %add8.i.us = fadd double %8, %10
  %11 = tail call double @llvm.exp.f64(double %10) #14
  %arrayidx14.i.us = getelementptr inbounds double, double* %0, i64 %9
  store double %11, double* %arrayidx14.i.us, align 8, !tbaa !3
  %indvars.iv.next.i.us = add nuw nsw i64 %indvars.iv.i.us, 1
  %exitcond.not.i.us = icmp eq i64 %indvars.iv.next.i.us, %wide.trip.count.i
  br i1 %exitcond.not.i.us, label %for.inc15.i.loopexit.us, label %for.body3.i.us, !llvm.loop !33

for.inc15.i.loopexit.us:                          ; preds = %for.body3.i.us
  %tiv.next2.us = add nuw nsw i64 %tiv1.us, 1
  store double %add8.i.us, double* %arrayidx.i.us, align 8, !tbaa !3
  %exitcond45.not.i.us = icmp eq i64 %tiv.next2.us, %wide.trip.count44.i
  br i1 %exitcond45.not.i.us, label %preprocess_qs.exit, label %for.body.i.us, !llvm.loop !34

preprocess_qs.exit:                               ; preds = %for.inc15.i.loopexit.us, %for.body.i.preheader, %entry
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
  %min.iters.check33 = icmp ult i32 %d, 4
  %n.vec36 = and i64 %wide.trip.count.i119, 4294967292
  %cmp.n40 = icmp eq i64 %n.vec36, %wide.trip.count.i119
  %min.iters.check18 = icmp ult i32 %d, 4
  %n.vec21 = and i64 %wide.trip.count.i119, 4294967292
  %cmp.n25 = icmp eq i64 %n.vec21, %wide.trip.count.i119
  br label %for.cond19.preheader

for.cond19.preheader:                             ; preds = %log_sum_exp.exit, %for.cond19.preheader.lr.ph
  %tiv = phi i64 [ %tiv.next, %log_sum_exp.exit ], [ 0, %for.cond19.preheader.lr.ph ]
  %12 = phi double [ %77, %log_sum_exp.exit ], [ undef, %for.cond19.preheader.lr.ph ]
  %13 = phi double [ %78, %log_sum_exp.exit ], [ undef, %for.cond19.preheader.lr.ph ]
  %slse.0143 = phi double [ %add47, %log_sum_exp.exit ], [ 0.000000e+00, %for.cond19.preheader.lr.ph ]
  %tiv.next = add nuw nsw i64 %tiv, 1
  br i1 %cmp37.i, label %for.body23.lr.ph, label %for.end

for.body23.lr.ph:                                 ; preds = %for.cond19.preheader
  %mul25 = mul nsw i64 %tiv, %conv7
  %arrayidx26 = getelementptr inbounds double, double* %x, i64 %mul25
  br i1 %cmp10.i, label %for.body23.us, label %for.body23.preheader

for.body23.preheader:                             ; preds = %for.body23.lr.ph
  %mul.i102 = fmul double %13, %13
  br label %for.body23

for.body23.us:                                    ; preds = %for.body23.lr.ph, %sqnorm.exit.us
  %ik.0133.us = phi i64 [ %inc.us, %sqnorm.exit.us ], [ 0, %for.body23.lr.ph ]
  %mul28.us = mul nsw i64 %ik.0133.us, %conv7
  %arrayidx29.us = getelementptr inbounds double, double* %means, i64 %mul28.us
  br i1 %min.iters.check33, label %for.body.i128.us.preheader, label %vector.body32

vector.body32:                                    ; preds = %for.body23.us, %vector.body32
  %index37 = phi i64 [ %index.next38, %vector.body32 ], [ 0, %for.body23.us ]
  %14 = getelementptr inbounds double, double* %arrayidx26, i64 %index37
  %15 = bitcast double* %14 to <2 x double>*
  %wide.load41 = load <2 x double>, <2 x double>* %15, align 8, !tbaa !3
  %16 = getelementptr inbounds double, double* %14, i64 2
  %17 = bitcast double* %16 to <2 x double>*
  %wide.load42 = load <2 x double>, <2 x double>* %17, align 8, !tbaa !3
  %18 = getelementptr inbounds double, double* %arrayidx29.us, i64 %index37
  %19 = bitcast double* %18 to <2 x double>*
  %wide.load43 = load <2 x double>, <2 x double>* %19, align 8, !tbaa !3
  %20 = getelementptr inbounds double, double* %18, i64 2
  %21 = bitcast double* %20 to <2 x double>*
  %wide.load44 = load <2 x double>, <2 x double>* %21, align 8, !tbaa !3
  %22 = fsub <2 x double> %wide.load41, %wide.load43
  %23 = fsub <2 x double> %wide.load42, %wide.load44
  %24 = getelementptr inbounds double, double* %2, i64 %index37
  %25 = bitcast double* %24 to <2 x double>*
  store <2 x double> %22, <2 x double>* %25, align 8, !tbaa !3
  %26 = getelementptr inbounds double, double* %24, i64 2
  %27 = bitcast double* %26 to <2 x double>*
  store <2 x double> %23, <2 x double>* %27, align 8, !tbaa !3
  %index.next38 = add i64 %index37, 4
  %28 = icmp eq i64 %index.next38, %n.vec36
  br i1 %28, label %middle.block30, label %vector.body32, !llvm.loop !154

middle.block30:                                   ; preds = %vector.body32
  br i1 %cmp.n40, label %for.body.i114.preheader.us, label %for.body.i128.us.preheader

for.body.i128.us.preheader:                       ; preds = %for.body23.us, %middle.block30
  %indvars.iv.i121.us.ph = phi i64 [ 0, %for.body23.us ], [ %n.vec36, %middle.block30 ]
  br label %for.body.i128.us

for.body.i128.us:                                 ; preds = %for.body.i128.us.preheader, %for.body.i128.us
  %indvars.iv.i121.us = phi i64 [ %indvars.iv.next.i126.us, %for.body.i128.us ], [ %indvars.iv.i121.us.ph, %for.body.i128.us.preheader ]
  %arrayidx.i122.us = getelementptr inbounds double, double* %arrayidx26, i64 %indvars.iv.i121.us
  %29 = load double, double* %arrayidx.i122.us, align 8, !tbaa !3
  %arrayidx2.i123.us = getelementptr inbounds double, double* %arrayidx29.us, i64 %indvars.iv.i121.us
  %30 = load double, double* %arrayidx2.i123.us, align 8, !tbaa !3
  %sub.i124.us = fsub double %29, %30
  %arrayidx4.i125.us = getelementptr inbounds double, double* %2, i64 %indvars.iv.i121.us
  store double %sub.i124.us, double* %arrayidx4.i125.us, align 8, !tbaa !3
  %indvars.iv.next.i126.us = add nuw nsw i64 %indvars.iv.i121.us, 1
  %exitcond.not.i127.us = icmp eq i64 %indvars.iv.next.i126.us, %wide.trip.count.i119
  br i1 %exitcond.not.i127.us, label %for.body.i114.preheader.us, label %for.body.i128.us, !llvm.loop !155

for.body.i114.preheader.us:                       ; preds = %for.body.i128.us, %middle.block30
  %arrayidx33.us = getelementptr inbounds double, double* %0, i64 %mul28.us
  %mul34.us = mul nsw i64 %ik.0133.us, %conv
  br i1 %min.iters.check18, label %for.body.i114.us.preheader, label %vector.body17

vector.body17:                                    ; preds = %for.body.i114.preheader.us, %vector.body17
  %index22 = phi i64 [ %index.next23, %vector.body17 ], [ 0, %for.body.i114.preheader.us ]
  %31 = getelementptr inbounds double, double* %arrayidx33.us, i64 %index22
  %32 = bitcast double* %31 to <2 x double>*
  %wide.load26 = load <2 x double>, <2 x double>* %32, align 8, !tbaa !3
  %33 = getelementptr inbounds double, double* %31, i64 2
  %34 = bitcast double* %33 to <2 x double>*
  %wide.load27 = load <2 x double>, <2 x double>* %34, align 8, !tbaa !3
  %35 = getelementptr inbounds double, double* %2, i64 %index22
  %36 = bitcast double* %35 to <2 x double>*
  %wide.load28 = load <2 x double>, <2 x double>* %36, align 8, !tbaa !3
  %37 = getelementptr inbounds double, double* %35, i64 2
  %38 = bitcast double* %37 to <2 x double>*
  %wide.load29 = load <2 x double>, <2 x double>* %38, align 8, !tbaa !3
  %39 = fmul <2 x double> %wide.load26, %wide.load28
  %40 = fmul <2 x double> %wide.load27, %wide.load29
  %41 = getelementptr inbounds double, double* %3, i64 %index22
  %42 = bitcast double* %41 to <2 x double>*
  store <2 x double> %39, <2 x double>* %42, align 8, !tbaa !3
  %43 = getelementptr inbounds double, double* %41, i64 2
  %44 = bitcast double* %43 to <2 x double>*
  store <2 x double> %40, <2 x double>* %44, align 8, !tbaa !3
  %index.next23 = add i64 %index22, 4
  %45 = icmp eq i64 %index.next23, %n.vec21
  br i1 %45, label %middle.block15, label %vector.body17, !llvm.loop !156

middle.block15:                                   ; preds = %vector.body17
  br i1 %cmp.n25, label %for.body7.i.us.preheader, label %for.body.i114.us.preheader

for.body.i114.us.preheader:                       ; preds = %for.body.i114.preheader.us, %middle.block15
  %indvars.iv69.i.us.ph = phi i64 [ 0, %for.body.i114.preheader.us ], [ %n.vec21, %middle.block15 ]
  br label %for.body.i114.us

for.body.i114.us:                                 ; preds = %for.body.i114.us.preheader, %for.body.i114.us
  %indvars.iv69.i.us = phi i64 [ %indvars.iv.next70.i.us, %for.body.i114.us ], [ %indvars.iv69.i.us.ph, %for.body.i114.us.preheader ]
  %arrayidx.i111.us = getelementptr inbounds double, double* %arrayidx33.us, i64 %indvars.iv69.i.us
  %46 = load double, double* %arrayidx.i111.us, align 8, !tbaa !3
  %arrayidx2.i112.us = getelementptr inbounds double, double* %2, i64 %indvars.iv69.i.us
  %47 = load double, double* %arrayidx2.i112.us, align 8, !tbaa !3
  %mul.i113.us = fmul double %46, %47
  %arrayidx4.i.us = getelementptr inbounds double, double* %3, i64 %indvars.iv69.i.us
  store double %mul.i113.us, double* %arrayidx4.i.us, align 8, !tbaa !3
  %indvars.iv.next70.i.us = add nuw nsw i64 %indvars.iv69.i.us, 1
  %exitcond72.not.i.us = icmp eq i64 %indvars.iv.next70.i.us, %wide.trip.count.i119
  br i1 %exitcond72.not.i.us, label %for.body7.i.us.preheader, label %for.body.i114.us, !llvm.loop !157

for.body7.i.us.preheader:                         ; preds = %for.body.i114.us, %middle.block15
  %arrayidx35.us = getelementptr inbounds double, double* %Ls, i64 %mul34.us
  br label %for.body7.i.us

for.body7.i.us:                                   ; preds = %for.body7.i.us.preheader, %for.cond5.loopexit.i.us
  %indvars.iv65.i.us = phi i64 [ %indvars.iv.next66.i.us, %for.cond5.loopexit.i.us ], [ 0, %for.body7.i.us.preheader ]
  %indvars.iv61.i.us = phi i64 [ %indvars.iv.next62.i.us, %for.cond5.loopexit.i.us ], [ 1, %for.body7.i.us.preheader ]
  %i.158.i.us = phi i32 [ %add.i115.us, %for.cond5.loopexit.i.us ], [ 0, %for.body7.i.us.preheader ]
  %48 = xor i64 %indvars.iv65.i.us, -1
  %49 = add nsw i64 %48, %wide.trip.count.i119
  %indvars.iv.next66.i.us = add nuw nsw i64 %indvars.iv65.i.us, 1
  %add.i115.us = add nuw nsw i32 %i.158.i.us, 1
  %cmp1254.i.us = icmp ult i64 %indvars.iv.next66.i.us, %wide.trip.count.i119
  br i1 %cmp1254.i.us, label %for.body13.lr.ph.i.us, label %for.cond5.loopexit.i.us

for.body13.lr.ph.i.us:                            ; preds = %for.body7.i.us
  %50 = xor i32 %i.158.i.us, -1
  %sub9.i.us = add i32 %mul8.i, %50
  %51 = trunc i64 %indvars.iv65.i.us to i32
  %mul10.i.us = mul nsw i32 %sub9.i.us, %51
  %div.i.us = sdiv i32 %mul10.i.us, 2
  %arrayidx19.i.us = getelementptr inbounds double, double* %2, i64 %indvars.iv65.i.us
  %52 = sext i32 %div.i.us to i64
  %53 = load double, double* %arrayidx19.i.us, align 8, !tbaa !3
  %min.iters.check = icmp ult i64 %49, 4
  br i1 %min.iters.check, label %for.body13.i.us.preheader, label %vector.ph

vector.ph:                                        ; preds = %for.body13.lr.ph.i.us
  %n.vec = and i64 %49, -4
  %ind.end = add i64 %indvars.iv61.i.us, %n.vec
  %ind.end8 = add i64 %n.vec, %52
  %broadcast.splatinsert = insertelement <2 x double> poison, double %53, i32 0
  %broadcast.splat = shufflevector <2 x double> %broadcast.splatinsert, <2 x double> poison, <2 x i32> zeroinitializer
  %broadcast.splatinsert13 = insertelement <2 x double> poison, double %53, i32 0
  %broadcast.splat14 = shufflevector <2 x double> %broadcast.splatinsert13, <2 x double> poison, <2 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %offset.idx = add i64 %indvars.iv61.i.us, %index
  %offset.idx9 = add i64 %index, %52
  %54 = getelementptr inbounds double, double* %3, i64 %offset.idx
  %55 = bitcast double* %54 to <2 x double>*
  %wide.load = load <2 x double>, <2 x double>* %55, align 8, !tbaa !3
  %56 = getelementptr inbounds double, double* %54, i64 2
  %57 = bitcast double* %56 to <2 x double>*
  %wide.load10 = load <2 x double>, <2 x double>* %57, align 8, !tbaa !3
  %58 = getelementptr inbounds double, double* %arrayidx35.us, i64 %offset.idx9
  %59 = bitcast double* %58 to <2 x double>*
  %wide.load11 = load <2 x double>, <2 x double>* %59, align 8, !tbaa !3
  %60 = getelementptr inbounds double, double* %58, i64 2
  %61 = bitcast double* %60 to <2 x double>*
  %wide.load12 = load <2 x double>, <2 x double>* %61, align 8, !tbaa !3
  %62 = fmul <2 x double> %broadcast.splat, %wide.load11
  %63 = fmul <2 x double> %broadcast.splat14, %wide.load12
  %64 = fadd <2 x double> %wide.load, %62
  %65 = fadd <2 x double> %wide.load10, %63
  %66 = bitcast double* %54 to <2 x double>*
  store <2 x double> %64, <2 x double>* %66, align 8, !tbaa !3
  %67 = bitcast double* %56 to <2 x double>*
  store <2 x double> %65, <2 x double>* %67, align 8, !tbaa !3
  %index.next = add i64 %index, 4
  %68 = icmp eq i64 %index.next, %n.vec
  br i1 %68, label %middle.block, label %vector.body, !llvm.loop !158

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %49, %n.vec
  br i1 %cmp.n, label %for.cond5.loopexit.i.us, label %for.body13.i.us.preheader

for.body13.i.us.preheader:                        ; preds = %for.body13.lr.ph.i.us, %middle.block
  %indvars.iv63.i.us.ph = phi i64 [ %indvars.iv61.i.us, %for.body13.lr.ph.i.us ], [ %ind.end, %middle.block ]
  %indvars.iv.i116.us.ph = phi i64 [ %52, %for.body13.lr.ph.i.us ], [ %ind.end8, %middle.block ]
  br label %for.body13.i.us

for.body13.i.us:                                  ; preds = %for.body13.i.us.preheader, %for.body13.i.us
  %indvars.iv63.i.us = phi i64 [ %indvars.iv.next64.i.us, %for.body13.i.us ], [ %indvars.iv63.i.us.ph, %for.body13.i.us.preheader ]
  %indvars.iv.i116.us = phi i64 [ %indvars.iv.next.i117.us, %for.body13.i.us ], [ %indvars.iv.i116.us.ph, %for.body13.i.us.preheader ]
  %arrayidx15.i.us = getelementptr inbounds double, double* %3, i64 %indvars.iv63.i.us
  %69 = load double, double* %arrayidx15.i.us, align 8, !tbaa !3
  %arrayidx17.i.us = getelementptr inbounds double, double* %arrayidx35.us, i64 %indvars.iv.i116.us
  %70 = load double, double* %arrayidx17.i.us, align 8, !tbaa !3
  %mul20.i.us = fmul double %53, %70
  %add21.i.us = fadd double %69, %mul20.i.us
  store double %add21.i.us, double* %arrayidx15.i.us, align 8, !tbaa !3
  %indvars.iv.next.i117.us = add nsw i64 %indvars.iv.i116.us, 1
  %indvars.iv.next64.i.us = add nuw nsw i64 %indvars.iv63.i.us, 1
  %exitcond.not.i118.us = icmp eq i64 %indvars.iv.next64.i.us, %wide.trip.count.i119
  br i1 %exitcond.not.i118.us, label %for.cond5.loopexit.i.us, label %for.body13.i.us, !llvm.loop !159

for.cond5.loopexit.i.us:                          ; preds = %for.body13.i.us, %middle.block, %for.body7.i.us
  %indvars.iv.next62.i.us = add nuw nsw i64 %indvars.iv61.i.us, 1
  %exitcond68.not.i.us = icmp eq i64 %indvars.iv.next66.i.us, %wide.trip.count.i119
  br i1 %exitcond68.not.i.us, label %cQtimesx.exit.loopexit.us, label %for.body7.i.us, !llvm.loop !45

cQtimesx.exit.loopexit.us:                        ; preds = %for.cond5.loopexit.i.us
  %.pre.us = load double, double* %3, align 8, !tbaa !3
  %arrayidx38.us = getelementptr inbounds double, double* %alphas, i64 %ik.0133.us
  %71 = load double, double* %arrayidx38.us, align 8, !tbaa !3
  %arrayidx39.us = getelementptr inbounds double, double* %1, i64 %ik.0133.us
  %72 = load double, double* %arrayidx39.us, align 8, !tbaa !3
  %add.us = fadd double %71, %72
  %mul.i102.us = fmul double %.pre.us, %.pre.us
  br i1 %cmp15.i, label %for.body.i109.us, label %sqnorm.exit.us

for.body.i109.us:                                 ; preds = %cQtimesx.exit.loopexit.us, %for.body.i109.us
  %indvars.iv.i105.us = phi i64 [ %indvars.iv.next.i107.us, %for.body.i109.us ], [ 1, %cQtimesx.exit.loopexit.us ]
  %res.017.i.us = phi double [ %add.i106.us, %for.body.i109.us ], [ %mul.i102.us, %cQtimesx.exit.loopexit.us ]
  %arrayidx2.i.us = getelementptr inbounds double, double* %3, i64 %indvars.iv.i105.us
  %73 = load double, double* %arrayidx2.i.us, align 8, !tbaa !3
  %mul5.i.us = fmul double %73, %73
  %add.i106.us = fadd double %res.017.i.us, %mul5.i.us
  %indvars.iv.next.i107.us = add nuw nsw i64 %indvars.iv.i105.us, 1
  %exitcond.not.i108.us = icmp eq i64 %indvars.iv.next.i107.us, %wide.trip.count.i119
  br i1 %exitcond.not.i108.us, label %sqnorm.exit.us, label %for.body.i109.us, !llvm.loop !10

sqnorm.exit.us:                                   ; preds = %for.body.i109.us, %cQtimesx.exit.loopexit.us
  %res.0.lcssa.i.us = phi double [ %mul.i102.us, %cQtimesx.exit.loopexit.us ], [ %add.i106.us, %for.body.i109.us ]
  %mul42.us = fmul double %res.0.lcssa.i.us, 5.000000e-01
  %sub43.us = fsub double %add.us, %mul42.us
  %arrayidx44.us = getelementptr inbounds double, double* %4, i64 %ik.0133.us
  store double %sub43.us, double* %arrayidx44.us, align 8, !tbaa !3
  %inc.us = add nuw nsw i64 %ik.0133.us, 1
  %exitcond.not.us = icmp eq i64 %inc.us, %conv4
  br i1 %exitcond.not.us, label %for.end.loopexit, label %for.body23.us, !llvm.loop !62

for.body23:                                       ; preds = %for.body23.preheader, %sqnorm.exit
  %ik.0133 = phi i64 [ %inc, %sqnorm.exit ], [ 0, %for.body23.preheader ]
  %arrayidx38 = getelementptr inbounds double, double* %alphas, i64 %ik.0133
  %74 = load double, double* %arrayidx38, align 8, !tbaa !3
  %arrayidx39 = getelementptr inbounds double, double* %1, i64 %ik.0133
  %75 = load double, double* %arrayidx39, align 8, !tbaa !3
  %add = fadd double %74, %75
  br i1 %cmp15.i, label %for.body.i109, label %sqnorm.exit

for.body.i109:                                    ; preds = %for.body23, %for.body.i109
  %indvars.iv.i105 = phi i64 [ %indvars.iv.next.i107, %for.body.i109 ], [ 1, %for.body23 ]
  %res.017.i = phi double [ %add.i106, %for.body.i109 ], [ %mul.i102, %for.body23 ]
  %arrayidx2.i = getelementptr inbounds double, double* %3, i64 %indvars.iv.i105
  %76 = load double, double* %arrayidx2.i, align 8, !tbaa !3
  %mul5.i = fmul double %76, %76
  %add.i106 = fadd double %res.017.i, %mul5.i
  %indvars.iv.next.i107 = add nuw nsw i64 %indvars.iv.i105, 1
  %exitcond.not.i108 = icmp eq i64 %indvars.iv.next.i107, %wide.trip.count.i119
  br i1 %exitcond.not.i108, label %sqnorm.exit, label %for.body.i109, !llvm.loop !10

sqnorm.exit:                                      ; preds = %for.body.i109, %for.body23
  %res.0.lcssa.i = phi double [ %mul.i102, %for.body23 ], [ %add.i106, %for.body.i109 ]
  %mul42 = fmul double %res.0.lcssa.i, 5.000000e-01
  %sub43 = fsub double %add, %mul42
  %arrayidx44 = getelementptr inbounds double, double* %4, i64 %ik.0133
  store double %sub43, double* %arrayidx44, align 8, !tbaa !3
  %inc = add nuw nsw i64 %ik.0133, 1
  %exitcond.not = icmp eq i64 %inc, %conv4
  br i1 %exitcond.not, label %for.end.loopexit, label %for.body23, !llvm.loop !62

for.end.loopexit:                                 ; preds = %sqnorm.exit, %sqnorm.exit.us
  %.lcssa = phi double [ %.pre.us, %sqnorm.exit.us ], [ %13, %sqnorm.exit ]
  %.pre146 = load double, double* %4, align 8, !tbaa !3
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %for.cond19.preheader
  %77 = phi double [ %.pre146, %for.end.loopexit ], [ %12, %for.cond19.preheader ]
  %78 = phi double [ %.lcssa, %for.end.loopexit ], [ %13, %for.cond19.preheader ]
  br i1 %cmp13.i.i, label %for.body.i.i, label %arr_max.exit.i

for.body.i.i:                                     ; preds = %for.end, %for.body.i.i
  %indvars.iv.i.i = phi i64 [ %indvars.iv.next.i.i, %for.body.i.i ], [ 1, %for.end ]
  %m.015.i.i = phi double [ %m.1.i.i, %for.body.i.i ], [ %77, %for.end ]
  %arrayidx1.i.i = getelementptr inbounds double, double* %4, i64 %indvars.iv.i.i
  %79 = load double, double* %arrayidx1.i.i, align 8, !tbaa !3
  %cmp2.i.i = fcmp olt double %m.015.i.i, %79
  %m.1.i.i = select i1 %cmp2.i.i, double %79, double %m.015.i.i
  %indvars.iv.next.i.i = add nuw nsw i64 %indvars.iv.i.i, 1
  %exitcond.not.i.i = icmp eq i64 %indvars.iv.next.i.i, %wide.trip.count.i.i
  br i1 %exitcond.not.i.i, label %arr_max.exit.i, label %for.body.i.i, !llvm.loop !7

arr_max.exit.i:                                   ; preds = %for.body.i.i, %for.end
  %m.0.lcssa.i.i = phi double [ %77, %for.end ], [ %m.1.i.i, %for.body.i.i ]
  br i1 %cmp37.i, label %for.body.preheader.i, label %log_sum_exp.exit

for.body.preheader.i:                             ; preds = %arr_max.exit.i
  %sub.i135 = fsub double %77, %m.0.lcssa.i.i
  %80 = tail call double @llvm.exp.f64(double %sub.i135) #14
  %add.i136 = fadd double %80, 0.000000e+00
  br i1 %exitcond.not.i99137, label %log_sum_exp.exit, label %for.body.for.body_crit_edge.i, !llvm.loop !22

for.body.for.body_crit_edge.i:                    ; preds = %for.body.preheader.i, %for.body.for.body_crit_edge.i
  %indvars.iv.next.i98139 = phi i64 [ %indvars.iv.next.i98, %for.body.for.body_crit_edge.i ], [ 1, %for.body.preheader.i ]
  %add.i138 = phi double [ %add.i, %for.body.for.body_crit_edge.i ], [ %add.i136, %for.body.preheader.i ]
  %arrayidx.phi.trans.insert.i = getelementptr inbounds double, double* %4, i64 %indvars.iv.next.i98139
  %.pre.i101 = load double, double* %arrayidx.phi.trans.insert.i, align 8, !tbaa !3
  %sub.i = fsub double %.pre.i101, %m.0.lcssa.i.i
  %81 = tail call double @llvm.exp.f64(double %sub.i) #14
  %add.i = fadd double %add.i138, %81
  %indvars.iv.next.i98 = add nuw nsw i64 %indvars.iv.next.i98139, 1
  %exitcond.not.i99 = icmp eq i64 %indvars.iv.next.i98, %wide.trip.count.i.i
  br i1 %exitcond.not.i99, label %log_sum_exp.exit, label %for.body.for.body_crit_edge.i, !llvm.loop !22

log_sum_exp.exit:                                 ; preds = %for.body.for.body_crit_edge.i, %for.body.preheader.i, %arr_max.exit.i
  %semx.0.lcssa.i = phi double [ 0.000000e+00, %arr_max.exit.i ], [ %add.i136, %for.body.preheader.i ], [ %add.i, %for.body.for.body_crit_edge.i ]
  %82 = tail call double @llvm.log.f64(double %semx.0.lcssa.i) #14
  %add1.i = fadd double %m.0.lcssa.i.i, %82
  %add47 = fadd double %slse.0143, %add1.i
  %exitcond145.not = icmp eq i64 %tiv.next, %conv17
  br i1 %exitcond145.not, label %for.end50, label %for.cond19.preheader, !llvm.loop !63

for.end50:                                        ; preds = %log_sum_exp.exit, %preprocess_qs.exit
  %slse.0.lcssa = phi double [ 0.000000e+00, %preprocess_qs.exit ], [ %add47, %log_sum_exp.exit ]
  store double %slse.0.lcssa, double* %err, align 8, !tbaa !3
  tail call void @free(i8* %call)
  tail call void @free(i8* %call6)
  tail call void @free(i8* %call9)
  tail call void @free(i8* %call12)
  tail call void @free(i8* %call15)
  ret void
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #10

; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn
declare void @llvm.experimental.noalias.scope.decl(metadata) #11

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare <2 x double> @llvm.exp.v2f64(<2 x double>) #3

; Function Attrs: nofree nosync nounwind readnone willreturn
declare double @llvm.vector.reduce.fadd.v2f64(double, <2 x double>) #12

attributes #0 = { norecurse nounwind readonly ssp uwtable "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nofree norecurse nounwind ssp uwtable "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readonly ssp uwtable "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #4 = { nofree nounwind ssp uwtable "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind ssp uwtable "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { inaccessiblememonly nofree nounwind willreturn allocsize(0) "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { inaccessiblemem_or_argmemonly nounwind willreturn "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { inaccessiblememonly nofree nounwind willreturn allocsize(0,1) "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #9 = { argmemonly nofree nosync nounwind willreturn }
attributes #10 = { argmemonly nofree nosync nounwind willreturn writeonly }
attributes #11 = { inaccessiblememonly nofree nosync nounwind willreturn }
attributes #12 = { nofree nosync nounwind readnone willreturn }
attributes #13 = { allocsize(0) }
attributes #14 = { nounwind }
attributes #15 = { nounwind allocsize(0) }
attributes #16 = { allocsize(0,1) }

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
!11 = !{!12}
!12 = distinct !{!12, !13}
!13 = distinct !{!13, !"LVerDomain"}
!14 = !{!15}
!15 = distinct !{!15, !13}
!16 = !{!17}
!17 = distinct !{!17, !13}
!18 = !{!12, !15}
!19 = distinct !{!19, !8, !9, !20}
!20 = !{!"llvm.loop.isvectorized", i32 1}
!21 = distinct !{!21, !8, !9, !20}
!22 = distinct !{!22, !8, !9}
!23 = !{!24}
!24 = distinct !{!24, !25}
!25 = distinct !{!25, !"LVerDomain"}
!26 = !{!27, !28}
!27 = distinct !{!27, !25}
!28 = distinct !{!28, !25}
!29 = !{!28}
!30 = !{!27}
!31 = distinct !{!31, !8, !9, !20}
!32 = distinct !{!32, !8, !9, !20}
!33 = distinct !{!33, !8, !9}
!34 = distinct !{!34, !8, !9}
!35 = !{!36}
!36 = distinct !{!36, !37}
!37 = distinct !{!37, !"LVerDomain"}
!38 = !{!39}
!39 = distinct !{!39, !37}
!40 = !{!41}
!41 = distinct !{!41, !37}
!42 = !{!36, !39}
!43 = distinct !{!43, !8, !9, !20}
!44 = distinct !{!44, !8, !9, !20}
!45 = distinct !{!45, !8, !9}
!46 = !{!47}
!47 = distinct !{!47, !48}
!48 = distinct !{!48, !"LVerDomain"}
!49 = !{!50}
!50 = distinct !{!50, !48}
!51 = !{!52, !47}
!52 = distinct !{!52, !48}
!53 = !{!52}
!54 = distinct !{!54, !8, !9, !20}
!55 = distinct !{!55, !8, !9, !20}
!56 = distinct !{!56, !8, !9, !20}
!57 = distinct !{!57, !8, !9, !20}
!58 = distinct !{!58, !8, !9, !20}
!59 = distinct !{!59, !8, !9, !20}
!60 = distinct !{!60, !8, !9, !20}
!61 = distinct !{!61, !8, !9, !20}
!62 = distinct !{!62, !8, !9}
!63 = distinct !{!63, !8, !9}
!64 = !{!65}
!65 = distinct !{!65, !66, !"diffeec_main_term: %alphas"}
!66 = distinct !{!66, !"diffeec_main_term"}
!67 = !{!68}
!68 = distinct !{!68, !66, !"diffeec_main_term: %means"}
!69 = !{!70}
!70 = distinct !{!70, !66, !"diffeec_main_term: %Qs"}
!71 = !{!72}
!72 = distinct !{!72, !66, !"diffeec_main_term: %Ls"}
!73 = !{!74}
!74 = distinct !{!74, !66, !"diffeec_main_term: %x"}
!75 = !{!65, !68, !70, !72, !74}
!76 = !{!65, !68, !72, !74}
!77 = distinct !{}
!78 = !{!65, !68, !70, !72}
!79 = !{!65, !70, !72, !74}
!80 = distinct !{!80, !8, !9, !20}
!81 = distinct !{!81, !8, !9, !20}
!82 = distinct !{!82, !8, !9, !20}
!83 = distinct !{}
!84 = distinct !{!84, !8, !9, !20}
!85 = !{!65, !68, !70, !74}
!86 = distinct !{!86, !8, !9, !20}
!87 = distinct !{}
!88 = distinct !{!88, !8, !9, !20}
!89 = distinct !{}
!90 = !{!68, !70, !72, !74}
!91 = distinct !{}
!92 = distinct !{}
!93 = distinct !{}
!94 = distinct !{}
!95 = distinct !{}
!96 = distinct !{!96, !20}
!97 = distinct !{!97, !98, !20}
!98 = !{!"llvm.loop.unroll.runtime.disable"}
!99 = distinct !{!99, !98, !20}
!100 = distinct !{}
!101 = distinct !{!101, !98, !20}
!102 = distinct !{!102, !20}
!103 = distinct !{!103, !20}
!104 = distinct !{!104, !20}
!105 = distinct !{!105, !98, !20}
!106 = distinct !{}
!107 = distinct !{!107, !98, !20}
!108 = distinct !{!108, !20}
!109 = distinct !{!109, !98, !20}
!110 = distinct !{!110, !20}
!111 = distinct !{!111, !8, !9}
!112 = distinct !{!112, !8, !9}
!113 = distinct !{!113, !8, !9}
!114 = !{!115}
!115 = distinct !{!115, !116}
!116 = distinct !{!116, !"LVerDomain"}
!117 = !{!118}
!118 = distinct !{!118, !116}
!119 = !{!120}
!120 = distinct !{!120, !116}
!121 = !{!118, !115}
!122 = distinct !{!122, !8, !9, !20}
!123 = distinct !{!123, !8, !9, !20}
!124 = distinct !{!124, !8, !9}
!125 = !{!126}
!126 = distinct !{!126, !127}
!127 = distinct !{!127, !"LVerDomain"}
!128 = !{!129}
!129 = distinct !{!129, !127}
!130 = distinct !{!130, !8, !9, !20}
!131 = distinct !{!131, !8, !9, !20}
!132 = distinct !{!132, !8, !9, !20}
!133 = distinct !{!133, !8, !9, !20}
!134 = distinct !{!134, !8, !9}
!135 = distinct !{!135, !8, !9, !20}
!136 = distinct !{!136, !8, !9, !20}
!137 = distinct !{!137, !8, !9, !20}
!138 = distinct !{!138, !8, !9, !20}
!139 = distinct !{!139, !8, !9, !20}
!140 = distinct !{!140, !8, !9, !20}
!141 = distinct !{!141, !8, !9}
!142 = distinct !{!142, !8, !9, !20}
!143 = distinct !{!143, !8, !9, !20}
!144 = distinct !{!144, !8, !9, !20}
!145 = distinct !{!145, !8, !9, !20}
!146 = distinct !{!146, !8, !9, !20}
!147 = distinct !{!147, !8, !9, !20}
!148 = distinct !{!148, !8, !9, !20}
!149 = distinct !{!149, !8, !9, !20}
!150 = distinct !{!150, !8, !9, !20}
!151 = distinct !{!151, !8, !9}
!152 = distinct !{!152, !8, !9, !20}
!153 = distinct !{!153, !8, !9}
!154 = distinct !{!154, !8, !9, !20}
!155 = distinct !{!155, !8, !9, !20}
!156 = distinct !{!156, !8, !9, !20}
!157 = distinct !{!157, !8, !9, !20}
!158 = distinct !{!158, !8, !9, !20}
!159 = distinct !{!159, !8, !9, !20}
