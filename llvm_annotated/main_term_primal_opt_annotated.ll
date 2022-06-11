;; Trying to understand the performance of Enzyme's GMM benchmark.
;; The pre-AD optimization has a dramatic effect on performance, so I wanted
;; to see if the LLVM post -O2 looks significantly different in terms of its
;; algorithm.

; ModuleID = '-'
source_filename = "-"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

@enzyme_const = external local_unnamed_addr global i32, align 4

; Function Attrs: norecurse nounwind readonly ssp uwtable
; define dso_local double @arr_max(i32 %n, double* nocapture readonly %x) local_unnamed_addr #0 {
; entry:
;   %0 = load double, double* %x, align 8, !tbaa !3
;   %cmp13 = icmp sgt i32 %n, 1
;   br i1 %cmp13, label %for.body.preheader, label %for.end

; for.body.preheader:                               ; preds = %entry
;   %wide.trip.count = zext i32 %n to i64
;   br label %for.body

; for.body:                                         ; preds = %for.body.preheader, %for.body
;   %indvars.iv = phi i64 [ 1, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
;   %m.015 = phi double [ %0, %for.body.preheader ], [ %m.1, %for.body ]
;   %arrayidx1 = getelementptr inbounds double, double* %x, i64 %indvars.iv
;   %1 = load double, double* %arrayidx1, align 8, !tbaa !3
;   %cmp2 = fcmp olt double %m.015, %1
;   %m.1 = select i1 %cmp2, double %1, double %m.015
;   %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
;   %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
;   br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !7

; for.end:                                          ; preds = %for.body, %entry
;   %m.0.lcssa = phi double [ %0, %entry ], [ %m.1, %for.body ]
;   ret double %m.0.lcssa
; }

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: norecurse nounwind readonly ssp uwtable
; define dso_local double @sqnorm(i32 %n, double* nocapture readonly %x) local_unnamed_addr #0 {
; entry:
;   %0 = load double, double* %x, align 8, !tbaa !3
;   %mul = fmul double %0, %0
;   %cmp15 = icmp sgt i32 %n, 1
;   br i1 %cmp15, label %for.body.preheader, label %for.end

; for.body.preheader:                               ; preds = %entry
;   %wide.trip.count = zext i32 %n to i64
;   br label %for.body

; for.body:                                         ; preds = %for.body.preheader, %for.body
;   %indvars.iv = phi i64 [ 1, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
;   %res.017 = phi double [ %mul, %for.body.preheader ], [ %add, %for.body ]
;   %arrayidx2 = getelementptr inbounds double, double* %x, i64 %indvars.iv
;   %1 = load double, double* %arrayidx2, align 8, !tbaa !3
;   %mul5 = fmul double %1, %1
;   %add = fadd double %res.017, %mul5
;   %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
;   %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
;   br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !10

; for.end:                                          ; preds = %for.body, %entry
;   %res.0.lcssa = phi double [ %mul, %entry ], [ %add, %for.body ]
;   ret double %res.0.lcssa
; }

; Function Attrs: nofree norecurse nounwind ssp uwtable
; define dso_local void @subtract(i32 %d, double* nocapture readonly %x, double* nocapture readonly %y, double* nocapture %out) local_unnamed_addr #2 {
; entry:
;   %cmp10 = icmp sgt i32 %d, 0
;   br i1 %cmp10, label %for.body.preheader, label %for.end

; for.body.preheader:                               ; preds = %entry
;   %wide.trip.count = zext i32 %d to i64
;   br label %for.body

; for.body:                                         ; preds = %for.body.preheader, %for.body
;   %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
;   %arrayidx = getelementptr inbounds double, double* %x, i64 %indvars.iv
;   %0 = load double, double* %arrayidx, align 8, !tbaa !3
;   %arrayidx2 = getelementptr inbounds double, double* %y, i64 %indvars.iv
;   %1 = load double, double* %arrayidx2, align 8, !tbaa !3
;   %sub = fsub double %0, %1
;   %arrayidx4 = getelementptr inbounds double, double* %out, i64 %indvars.iv
;   store double %sub, double* %arrayidx4, align 8, !tbaa !3
;   %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
;   %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
;   br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !11

; for.end:                                          ; preds = %for.body, %entry
;   ret void
; }

; Function Attrs: nounwind readonly ssp uwtable
; define dso_local double @log_sum_exp(i32 %n, double* nocapture readonly %x) local_unnamed_addr #3 {
; entry:
;   %0 = load double, double* %x, align 8, !tbaa !3
;   %cmp13.i = icmp sgt i32 %n, 1
;   br i1 %cmp13.i, label %for.body.preheader.i, label %arr_max.exit

; for.body.preheader.i:                             ; preds = %entry
;   %wide.trip.count.i = zext i32 %n to i64
;   br label %for.body.i

; for.body.i:                                       ; preds = %for.body.i, %for.body.preheader.i
;   %indvars.iv.i = phi i64 [ 1, %for.body.preheader.i ], [ %indvars.iv.next.i, %for.body.i ]
;   %m.015.i = phi double [ %0, %for.body.preheader.i ], [ %m.1.i, %for.body.i ]
;   %arrayidx1.i = getelementptr inbounds double, double* %x, i64 %indvars.iv.i
;   %1 = load double, double* %arrayidx1.i, align 8, !tbaa !3
;   %cmp2.i = fcmp olt double %m.015.i, %1
;   %m.1.i = select i1 %cmp2.i, double %1, double %m.015.i
;   %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
;   %exitcond.not.i = icmp eq i64 %indvars.iv.next.i, %wide.trip.count.i
;   br i1 %exitcond.not.i, label %arr_max.exit, label %for.body.i, !llvm.loop !7

; arr_max.exit:                                     ; preds = %for.body.i, %entry
;   %m.0.lcssa.i = phi double [ %0, %entry ], [ %m.1.i, %for.body.i ]
;   %cmp11 = icmp sgt i32 %n, 0
;   br i1 %cmp11, label %for.body.preheader, label %for.end

; for.body.preheader:                               ; preds = %arr_max.exit
;   %wide.trip.count = zext i32 %n to i64
;   %sub14 = fsub double %0, %m.0.lcssa.i
;   %2 = tail call double @llvm.exp.f64(double %sub14)
;   %add15 = fadd double %2, 0.000000e+00
;   %exitcond.not16 = icmp eq i32 %n, 1
;   br i1 %exitcond.not16, label %for.end, label %for.body.for.body_crit_edge, !llvm.loop !12

; for.body.for.body_crit_edge:                      ; preds = %for.body.preheader, %for.body.for.body_crit_edge
;   %indvars.iv.next18 = phi i64 [ %indvars.iv.next, %for.body.for.body_crit_edge ], [ 1, %for.body.preheader ]
;   %add17 = phi double [ %add, %for.body.for.body_crit_edge ], [ %add15, %for.body.preheader ]
;   %arrayidx.phi.trans.insert = getelementptr inbounds double, double* %x, i64 %indvars.iv.next18
;   %.pre = load double, double* %arrayidx.phi.trans.insert, align 8, !tbaa !3
;   %sub = fsub double %.pre, %m.0.lcssa.i
;   %3 = tail call double @llvm.exp.f64(double %sub)
;   %add = fadd double %add17, %3
;   %indvars.iv.next = add nuw nsw i64 %indvars.iv.next18, 1
;   %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
;   br i1 %exitcond.not, label %for.end, label %for.body.for.body_crit_edge, !llvm.loop !12

; for.end:                                          ; preds = %for.body.for.body_crit_edge, %for.body.preheader, %arr_max.exit
;   %semx.0.lcssa = phi double [ 0.000000e+00, %arr_max.exit ], [ %add15, %for.body.preheader ], [ %add, %for.body.for.body_crit_edge ]
;   %4 = tail call double @llvm.log.f64(double %semx.0.lcssa)
;   %add1 = fadd double %m.0.lcssa.i, %4
;   ret double %add1
; }

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare double @llvm.exp.f64(double) #4

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare double @llvm.log.f64(double) #4

; Function Attrs: nofree nounwind ssp uwtable
; define dso_local void @preprocess_qs(i32 %d, i32 %k, double* nocapture readonly %Qs, double* nocapture %sum_qs, double* nocapture %Qdiags) local_unnamed_addr #5 {
; entry:
;   %cmp37 = icmp sgt i32 %k, 0
;   br i1 %cmp37, label %for.body.lr.ph, label %for.end17

; for.body.lr.ph:                                   ; preds = %entry
;   %cmp235 = icmp sgt i32 %d, 0
;   %wide.trip.count44 = zext i32 %k to i64
;   %wide.trip.count = zext i32 %d to i64
;   br label %for.body

; for.body:                                         ; preds = %for.body.lr.ph, %for.inc15
;   %indvars.iv42 = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next43, %for.inc15 ]
;   %arrayidx = getelementptr inbounds double, double* %sum_qs, i64 %indvars.iv42
;   store double 0.000000e+00, double* %arrayidx, align 8, !tbaa !3
;   br i1 %cmp235, label %for.body3.lr.ph, label %for.inc15

; for.body3.lr.ph:                                  ; preds = %for.body
;   %0 = trunc i64 %indvars.iv42 to i32
;   %mul = mul nsw i32 %0, %d
;   %1 = sext i32 %mul to i64
;   %arrayidx546 = getelementptr inbounds double, double* %Qs, i64 %1
;   %2 = load double, double* %arrayidx546, align 8, !tbaa !3
;   %add847 = fadd double %2, 0.000000e+00
;   store double %add847, double* %arrayidx, align 8, !tbaa !3
;   %3 = tail call double @llvm.exp.f64(double %2)
;   %arrayidx1448 = getelementptr inbounds double, double* %Qdiags, i64 %1
;   store double %3, double* %arrayidx1448, align 8, !tbaa !3
;   %exitcond.not49 = icmp eq i32 %d, 1
;   br i1 %exitcond.not49, label %for.inc15, label %for.body3.for.body3_crit_edge, !llvm.loop !13

; for.body3.for.body3_crit_edge:                    ; preds = %for.body3.lr.ph, %for.body3.for.body3_crit_edge
;   %indvars.iv.next50 = phi i64 [ %indvars.iv.next, %for.body3.for.body3_crit_edge ], [ 1, %for.body3.lr.ph ]
;   %.pre = load double, double* %arrayidx, align 8, !tbaa !3
;   %4 = add nsw i64 %indvars.iv.next50, %1
;   %arrayidx5 = getelementptr inbounds double, double* %Qs, i64 %4
;   %5 = load double, double* %arrayidx5, align 8, !tbaa !3
;   %add8 = fadd double %5, %.pre
;   store double %add8, double* %arrayidx, align 8, !tbaa !3
;   %6 = tail call double @llvm.exp.f64(double %5)
;   %arrayidx14 = getelementptr inbounds double, double* %Qdiags, i64 %4
;   store double %6, double* %arrayidx14, align 8, !tbaa !3
;   %indvars.iv.next = add nuw nsw i64 %indvars.iv.next50, 1
;   %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
;   br i1 %exitcond.not, label %for.inc15, label %for.body3.for.body3_crit_edge, !llvm.loop !13

; for.inc15:                                        ; preds = %for.body3.for.body3_crit_edge, %for.body3.lr.ph, %for.body
;   %indvars.iv.next43 = add nuw nsw i64 %indvars.iv42, 1
;   %exitcond45.not = icmp eq i64 %indvars.iv.next43, %wide.trip.count44
;   br i1 %exitcond45.not, label %for.end17, label %for.body, !llvm.loop !14

; for.end17:                                        ; preds = %for.inc15, %entry
;   ret void
; }

; Function Attrs: nofree norecurse nounwind ssp uwtable
; define dso_local void @cQtimesx(i32 %d, double* nocapture readonly %Qdiag, double* nocapture readonly %ltri, double* nocapture readonly %x, double* nocapture %out) local_unnamed_addr #2 {
; entry:
;   %cmp59 = icmp sgt i32 %d, 0
;   br i1 %cmp59, label %for.body.preheader, label %for.end30

; for.body.preheader:                               ; preds = %entry
;   %wide.trip.count71 = zext i32 %d to i64
;   br label %for.body

; for.cond5.preheader:                              ; preds = %for.body
;   br i1 %cmp59, label %for.body7.lr.ph, label %for.end30

; for.body7.lr.ph:                                  ; preds = %for.cond5.preheader
;   %mul8 = shl nuw nsw i32 %d, 1
;   %0 = zext i32 %d to i64
;   %wide.trip.count67 = zext i32 %d to i64
;   br label %for.body7

; for.body:                                         ; preds = %for.body.preheader, %for.body
;   %indvars.iv69 = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next70, %for.body ]
;   %arrayidx = getelementptr inbounds double, double* %Qdiag, i64 %indvars.iv69
;   %1 = load double, double* %arrayidx, align 8, !tbaa !3
;   %arrayidx2 = getelementptr inbounds double, double* %x, i64 %indvars.iv69
;   %2 = load double, double* %arrayidx2, align 8, !tbaa !3
;   %mul = fmul double %1, %2
;   %arrayidx4 = getelementptr inbounds double, double* %out, i64 %indvars.iv69
;   store double %mul, double* %arrayidx4, align 8, !tbaa !3
;   %indvars.iv.next70 = add nuw nsw i64 %indvars.iv69, 1
;   %exitcond72.not = icmp eq i64 %indvars.iv.next70, %wide.trip.count71
;   br i1 %exitcond72.not, label %for.cond5.preheader, label %for.body, !llvm.loop !15

; for.cond5.loopexit:                               ; preds = %for.body13, %for.body7
;   %indvars.iv.next62 = add nuw nsw i64 %indvars.iv61, 1
;   %exitcond68.not = icmp eq i64 %indvars.iv.next66, %wide.trip.count67
;   br i1 %exitcond68.not, label %for.end30, label %for.body7, !llvm.loop !16

; for.body7:                                        ; preds = %for.body7.lr.ph, %for.cond5.loopexit
;   %indvars.iv65 = phi i64 [ 0, %for.body7.lr.ph ], [ %indvars.iv.next66, %for.cond5.loopexit ]
;   %indvars.iv61 = phi i64 [ 1, %for.body7.lr.ph ], [ %indvars.iv.next62, %for.cond5.loopexit ]
;   %i.158 = phi i32 [ 0, %for.body7.lr.ph ], [ %add, %for.cond5.loopexit ]
;   %indvars.iv.next66 = add nuw nsw i64 %indvars.iv65, 1
;   %add = add nuw nsw i32 %i.158, 1
;   %cmp1254 = icmp ult i64 %indvars.iv.next66, %0
;   br i1 %cmp1254, label %for.body13.lr.ph, label %for.cond5.loopexit

; for.body13.lr.ph:                                 ; preds = %for.body7
;   %3 = xor i32 %i.158, -1
;   %sub9 = add i32 %mul8, %3
;   %4 = trunc i64 %indvars.iv65 to i32
;   %mul10 = mul nsw i32 %sub9, %4
;   %div = sdiv i32 %mul10, 2
;   %arrayidx19 = getelementptr inbounds double, double* %x, i64 %indvars.iv65
;   %5 = sext i32 %div to i64
;   br label %for.body13

; for.body13:                                       ; preds = %for.body13.lr.ph, %for.body13
;   %indvars.iv63 = phi i64 [ %indvars.iv61, %for.body13.lr.ph ], [ %indvars.iv.next64, %for.body13 ]
;   %indvars.iv = phi i64 [ %5, %for.body13.lr.ph ], [ %indvars.iv.next, %for.body13 ]
;   %arrayidx15 = getelementptr inbounds double, double* %out, i64 %indvars.iv63
;   %6 = load double, double* %arrayidx15, align 8, !tbaa !3
;   %arrayidx17 = getelementptr inbounds double, double* %ltri, i64 %indvars.iv
;   %7 = load double, double* %arrayidx17, align 8, !tbaa !3
;   %8 = load double, double* %arrayidx19, align 8, !tbaa !3
;   %mul20 = fmul double %7, %8
;   %add21 = fadd double %6, %mul20
;   store double %add21, double* %arrayidx15, align 8, !tbaa !3
;   %indvars.iv.next = add nsw i64 %indvars.iv, 1
;   %indvars.iv.next64 = add nuw nsw i64 %indvars.iv63, 1
;   %exitcond.not = icmp eq i64 %indvars.iv.next64, %wide.trip.count67
;   br i1 %exitcond.not, label %for.cond5.loopexit, label %for.body13, !llvm.loop !17

; for.end30:                                        ; preds = %for.cond5.loopexit, %entry, %for.cond5.preheader
;   ret void
; }

; Function Attrs: nounwind ssp uwtable
define dso_local void @ec_main_term(i32 %d, i32 %k, i32 %n, double* noalias nocapture readonly %alphas, double* noalias nocapture readonly %means, double* noalias nocapture readonly %Qs, double* noalias nocapture readonly %Ls, double* noalias nocapture readonly %x, double* nocapture %err) #6 {
entry:
  %sub = add nsw i32 %d, -1
  %mul = mul nsw i32 %sub, %d
  %div = sdiv i32 %mul, 2
  ;; %conv is d * (d - 1) / 2 -> tri_size
  %conv = sext i32 %div to i64
  %mul1 = mul nsw i32 %k, %d
  %conv2 = sext i32 %mul1 to i64
  %mul3 = shl nsw i64 %conv2, 3
  ;; %mul3 is k * d
  %call = tail call i8* @malloc(i64 %mul3) #11
  %0 = bitcast i8* %call to double*
  %conv4 = sext i32 %k to i64
  %mul5 = shl nsw i64 %conv4, 3
  ;; %mul5 is k
  %call6 = tail call i8* @malloc(i64 %mul5) #11
  ;; %1 is sum_qs
  %1 = bitcast i8* %call6 to double*
  %conv7 = sext i32 %d to i64
  %mul8 = shl nsw i64 %conv7, 3
  ;; %mul8 is d
  %call9 = tail call i8* @malloc(i64 %mul8) #11
  ;; %2 is xcentered
  %2 = bitcast i8* %call9 to double*
  %call12 = tail call i8* @malloc(i64 %mul8) #11
  ;; %3 is Qxcentered
  %3 = bitcast i8* %call12 to double*
  %call15 = tail call i8* @malloc(i64 %mul5) #11
  ;; %4 is main_term
  %4 = bitcast i8* %call15 to double*
  %cmp37.i = icmp sgt i32 %k, 0
  br i1 %cmp37.i, label %for.body.lr.ph.i, label %preprocess_qs.exit

for.body.lr.ph.i:                                 ; preds = %entry
  %cmp235.i = icmp sgt i32 %d, 0
  %wide.trip.count44.i = zext i32 %k to i64
  %wide.trip.count.i = zext i32 %d to i64
  br label %for.body.i

for.body.i:                                       ; preds = %for.inc15.i, %for.body.lr.ph.i
  %indvars.iv42.i = phi i64 [ 0, %for.body.lr.ph.i ], [ %indvars.iv.next43.i, %for.inc15.i ]
  %arrayidx.i = getelementptr inbounds double, double* %1, i64 %indvars.iv42.i
  store double 0.000000e+00, double* %arrayidx.i, align 8, !tbaa !3
  br i1 %cmp235.i, label %for.body3.lr.ph.i, label %for.inc15.i

for.body3.lr.ph.i:                                ; preds = %for.body.i
  %5 = trunc i64 %indvars.iv42.i to i32
  %mul.i = mul nsw i32 %5, %d
  %6 = sext i32 %mul.i to i64
  br label %for.body3.i

for.body3.i:                                      ; preds = %for.body3.i, %for.body3.lr.ph.i
  %7 = phi double [ 0.000000e+00, %for.body3.lr.ph.i ], [ %add8.i, %for.body3.i ]
  %indvars.iv.i = phi i64 [ 0, %for.body3.lr.ph.i ], [ %indvars.iv.next.i, %for.body3.i ]
  %8 = add nsw i64 %indvars.iv.i, %6
  %arrayidx5.i = getelementptr inbounds double, double* %Qs, i64 %8
  %9 = load double, double* %arrayidx5.i, align 8, !tbaa !3
  %add8.i = fadd double %7, %9
  %10 = tail call double @llvm.exp.f64(double %9) #12
  %arrayidx14.i = getelementptr inbounds double, double* %0, i64 %8
  store double %10, double* %arrayidx14.i, align 8, !tbaa !3
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
  %exitcond.not.i = icmp eq i64 %indvars.iv.next.i, %wide.trip.count.i
  br i1 %exitcond.not.i, label %for.inc15.i.loopexit, label %for.body3.i, !llvm.loop !13

for.inc15.i.loopexit:                             ; preds = %for.body3.i
  store double %add8.i, double* %arrayidx.i, align 8, !tbaa !3
  br label %for.inc15.i

for.inc15.i:                                      ; preds = %for.inc15.i.loopexit, %for.body.i
  %indvars.iv.next43.i = add nuw nsw i64 %indvars.iv42.i, 1
  %exitcond45.not.i = icmp eq i64 %indvars.iv.next43.i, %wide.trip.count44.i
  br i1 %exitcond45.not.i, label %preprocess_qs.exit, label %for.body.i, !llvm.loop !14

preprocess_qs.exit:                               ; preds = %for.inc15.i, %entry
  ;; %conv17 is n, this is essentially the outer loop over n
  %conv17 = sext i32 %n to i64
  %cmp140 = icmp sgt i32 %n, 0
  br i1 %cmp140, label %for.cond19.preheader.lr.ph, label %for.end50
  ;; This is the end of the preprocess_qs function.

for.cond19.preheader.lr.ph:                       ; preds = %preprocess_qs.exit
  %cmp10.i = icmp sgt i32 %d, 0
  %wide.trip.count.i119 = zext i32 %d to i64
  %mul8.i = shl nuw nsw i32 %d, 1
  %cmp15.i = icmp sgt i32 %d, 1
  %cmp13.i.i = icmp sgt i32 %k, 1
  %wide.trip.count.i.i = zext i32 %k to i64
  %exitcond.not.i99137 = icmp eq i32 %k, 1
  br label %for.cond19.preheader

for.cond19.preheader:                             ; preds = %for.cond19.preheader.lr.ph, %log_sum_exp.exit
  ;; Outer loop branches back to here
  ;; %11 is main_term[0]
  %11 = phi double [ undef, %for.cond19.preheader.lr.ph ], [ %28, %log_sum_exp.exit ]
  %12 = phi double [ undef, %for.cond19.preheader.lr.ph ], [ %29, %log_sum_exp.exit ]
  %slse.0143 = phi double [ 0.000000e+00, %for.cond19.preheader.lr.ph ], [ %add47, %log_sum_exp.exit ]
  %ix.0141 = phi i64 [ 0, %for.cond19.preheader.lr.ph ], [ %inc49, %log_sum_exp.exit ]
  br i1 %cmp37.i, label %for.body23.lr.ph, label %for.end

for.body23.lr.ph:                                 ; preds = %for.cond19.preheader
  ;; %conv7 is d
  %mul25 = mul nsw i64 %ix.0141, %conv7
  ;; %arrayidx26 is the slice of x
  %arrayidx26 = getelementptr inbounds double, double* %x, i64 %mul25
  br label %for.body23

for.body23:                                       ; preds = %for.body23.lr.ph, %sqnorm.exit
  ;; Inner loop over k branches here
  ;; %24 is %3, Qxcentered
  %13 = phi double [ %12, %for.body23.lr.ph ], [ %24, %sqnorm.exit ]
  %ik.0133 = phi i64 [ 0, %for.body23.lr.ph ], [ %inc, %sqnorm.exit ]
  %mul28 = mul nsw i64 %ik.0133, %conv7
  ;; %arrayidx29 is the slice of means
  %arrayidx29 = getelementptr inbounds double, double* %means, i64 %mul28
  ;; cQtimesx exit is the last block in the loop?
  ;; That's different than before.
  br i1 %cmp10.i, label %for.body.i128, label %cQtimesx.exit

for.body.i128:                                    ; preds = %for.body23, %for.body.i128
  ;; This is the subtract function
  %indvars.iv.i121 = phi i64 [ %indvars.iv.next.i126, %for.body.i128 ], [ 0, %for.body23 ]
  %arrayidx.i122 = getelementptr inbounds double, double* %arrayidx26, i64 %indvars.iv.i121
  %14 = load double, double* %arrayidx.i122, align 8, !tbaa !3
  %arrayidx2.i123 = getelementptr inbounds double, double* %arrayidx29, i64 %indvars.iv.i121
  %15 = load double, double* %arrayidx2.i123, align 8, !tbaa !3
  %sub.i124 = fsub double %14, %15
  ;; %2 is xcentered
  %arrayidx4.i125 = getelementptr inbounds double, double* %2, i64 %indvars.iv.i121
  store double %sub.i124, double* %arrayidx4.i125, align 8, !tbaa !3
  %indvars.iv.next.i126 = add nuw nsw i64 %indvars.iv.i121, 1
  ;; %wide.trip.count.i119 is d
  %exitcond.not.i127 = icmp eq i64 %indvars.iv.next.i126, %wide.trip.count.i119
  br i1 %exitcond.not.i127, label %for.body.i114.preheader, label %for.body.i128, !llvm.loop !11
  ;; End of the subtract function

for.body.i114.preheader:                          ; preds = %for.body.i128
  ;; %arrayidx33 is Qdiags, %mul28 is ik * d
  %arrayidx33 = getelementptr inbounds double, double* %0, i64 %mul28
  ;; %conv is tri_size, %mul34 is ik * tri_size
  %mul34 = mul nsw i64 %ik.0133, %conv
  ;; %arrayidx35 is Ls slice
  %arrayidx35 = getelementptr inbounds double, double* %Ls, i64 %mul34
  br label %for.body.i114

for.body.i114:                                    ; preds = %for.body.i114.preheader, %for.body.i114
  ;; This is the multiplication of Qdiag and x (* cQtimesx starts here *)
  %indvars.iv69.i = phi i64 [ %indvars.iv.next70.i, %for.body.i114 ], [ 0, %for.body.i114.preheader ]
  %arrayidx.i111 = getelementptr inbounds double, double* %arrayidx33, i64 %indvars.iv69.i
  %16 = load double, double* %arrayidx.i111, align 8, !tbaa !3
  %arrayidx2.i112 = getelementptr inbounds double, double* %2, i64 %indvars.iv69.i
  %17 = load double, double* %arrayidx2.i112, align 8, !tbaa !3
  %mul.i113 = fmul double %16, %17
  %arrayidx4.i = getelementptr inbounds double, double* %3, i64 %indvars.iv69.i
  store double %mul.i113, double* %arrayidx4.i, align 8, !tbaa !3
  %indvars.iv.next70.i = add nuw nsw i64 %indvars.iv69.i, 1
  ;; %wide.trip.count.i119 is d
  %exitcond72.not.i = icmp eq i64 %indvars.iv.next70.i, %wide.trip.count.i119
  br i1 %exitcond72.not.i, label %for.body7.i, label %for.body.i114, !llvm.loop !15
  ;; This is the end of the elementwise multiplication
  ;; This next block is skipped and then it goes to the one after.

for.cond5.loopexit.i:                             ; preds = %for.body13.i, %for.body7.i
  %indvars.iv.next62.i = add nuw nsw i64 %indvars.iv61.i, 1
  %exitcond68.not.i = icmp eq i64 %indvars.iv.next66.i, %wide.trip.count.i119
  br i1 %exitcond68.not.i, label %cQtimesx.exit.loopexit, label %for.body7.i, !llvm.loop !16

for.body7.i:                                      ; preds = %for.body.i114, %for.cond5.loopexit.i
  ;; This is the beginning of the TRMV
  ;; %indvars.iv65.i is i
  %indvars.iv65.i = phi i64 [ %indvars.iv.next66.i, %for.cond5.loopexit.i ], [ 0, %for.body.i114 ]
  ;; I think %indvars.iv61.i is j
  %indvars.iv61.i = phi i64 [ %indvars.iv.next62.i, %for.cond5.loopexit.i ], [ 1, %for.body.i114 ]
  %i.158.i = phi i32 [ %add.i115, %for.cond5.loopexit.i ], [ 0, %for.body.i114 ]
  %indvars.iv.next66.i = add nuw nsw i64 %indvars.iv65.i, 1
  %add.i115 = add nuw nsw i32 %i.158.i, 1
  ;; %wide.trip.count.i119 is d
  %cmp1254.i = icmp ult i64 %indvars.iv.next66.i, %wide.trip.count.i119
  br i1 %cmp1254.i, label %for.body13.lr.ph.i, label %for.cond5.loopexit.i

for.body13.lr.ph.i:                               ; preds = %for.body7.i
  %18 = xor i32 %i.158.i, -1
  %sub9.i = add i32 %mul8.i, %18
  %19 = trunc i64 %indvars.iv65.i to i32
  %mul10.i = mul nsw i32 %sub9.i, %19
  %div.i = sdiv i32 %mul10.i, 2
  ;; %2 is xcentered
  %arrayidx19.i = getelementptr inbounds double, double* %2, i64 %indvars.iv65.i
  ;; %20 is the Lidx init per outer loop
  %20 = sext i32 %div.i to i64
  %21 = load double, double* %arrayidx19.i, align 8, !tbaa !3
  br label %for.body13.i

for.body13.i:                                     ; preds = %for.body13.i, %for.body13.lr.ph.i
  %indvars.iv63.i = phi i64 [ %indvars.iv61.i, %for.body13.lr.ph.i ], [ %indvars.iv.next64.i, %for.body13.i ]
  ;; %indvars.iv.i116 is Lidx
  %indvars.iv.i116 = phi i64 [ %20, %for.body13.lr.ph.i ], [ %indvars.iv.next.i117, %for.body13.i ]
  ;; %3 is Qxcentered
  %arrayidx15.i = getelementptr inbounds double, double* %3, i64 %indvars.iv63.i
  %22 = load double, double* %arrayidx15.i, align 8, !tbaa !3
  ;; This is Ls slice
  %arrayidx17.i = getelementptr inbounds double, double* %arrayidx35, i64 %indvars.iv.i116
  %23 = load double, double* %arrayidx17.i, align 8, !tbaa !3
  %mul20.i = fmul double %23, %21
  %add21.i = fadd double %22, %mul20.i
  store double %add21.i, double* %arrayidx15.i, align 8, !tbaa !3
  %indvars.iv.next.i117 = add nsw i64 %indvars.iv.i116, 1
  %indvars.iv.next64.i = add nuw nsw i64 %indvars.iv63.i, 1
  %exitcond.not.i118 = icmp eq i64 %indvars.iv.next64.i, %wide.trip.count.i119
  br i1 %exitcond.not.i118, label %for.cond5.loopexit.i, label %for.body13.i, !llvm.loop !17

cQtimesx.exit.loopexit:                           ; preds = %for.cond5.loopexit.i
  ;; %3 is Qxcentered, we're entering sqnorm
  %.pre = load double, double* %3, align 8, !tbaa !3
  br label %cQtimesx.exit

cQtimesx.exit:                                    ; preds = %cQtimesx.exit.loopexit, %for.body23
  %24 = phi double [ %.pre, %cQtimesx.exit.loopexit ], [ %13, %for.body23 ]
  %arrayidx38 = getelementptr inbounds double, double* %alphas, i64 %ik.0133
  ;; %25 is a[ik]
  %25 = load double, double* %arrayidx38, align 8, !tbaa !3
  %arrayidx39 = getelementptr inbounds double, double* %1, i64 %ik.0133
  ;; %26 is sum_qs[ik]
  %26 = load double, double* %arrayidx39, align 8, !tbaa !3
  %add = fadd double %25, %26
  ;; This is the first line of sqnorm, x[0] * x[0]
  %mul.i102 = fmul double %24, %24
  br i1 %cmp15.i, label %for.body.i109, label %sqnorm.exit

for.body.i109:                                    ; preds = %cQtimesx.exit, %for.body.i109
  %indvars.iv.i105 = phi i64 [ %indvars.iv.next.i107, %for.body.i109 ], [ 1, %cQtimesx.exit ]
  ;; %res.017.i is the result of sqnorm
  %res.017.i = phi double [ %add.i106, %for.body.i109 ], [ %mul.i102, %cQtimesx.exit ]
  %arrayidx2.i = getelementptr inbounds double, double* %3, i64 %indvars.iv.i105
  %27 = load double, double* %arrayidx2.i, align 8, !tbaa !3
  %mul5.i = fmul double %27, %27
  %add.i106 = fadd double %res.017.i, %mul5.i
  %indvars.iv.next.i107 = add nuw nsw i64 %indvars.iv.i105, 1
  ;; %wide.trip.count.i119 is d
  %exitcond.not.i108 = icmp eq i64 %indvars.iv.next.i107, %wide.trip.count.i119
  br i1 %exitcond.not.i108, label %sqnorm.exit, label %for.body.i109, !llvm.loop !10

sqnorm.exit:                                      ; preds = %for.body.i109, %cQtimesx.exit
  ;; This is the computation that populates main_term[k]
  %res.0.lcssa.i = phi double [ %mul.i102, %cQtimesx.exit ], [ %add.i106, %for.body.i109 ]
  %mul42 = fmul double %res.0.lcssa.i, 5.000000e-01
  %sub43 = fsub double %add, %mul42
  %arrayidx44 = getelementptr inbounds double, double* %4, i64 %ik.0133
  ;; main_term[k] is written to here
  store double %sub43, double* %arrayidx44, align 8, !tbaa !3
  %inc = add nuw nsw i64 %ik.0133, 1
  ;; %conv4 is k
  %exitcond.not = icmp eq i64 %inc, %conv4
  br i1 %exitcond.not, label %for.end.loopexit, label %for.body23, !llvm.loop !18
  ;; End of inner loop over k

for.end.loopexit:                                 ; preds = %sqnorm.exit
  ;; %4 is main_term
  %.pre146 = load double, double* %4, align 8, !tbaa !3
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %for.cond19.preheader
  ;; arr_max function is here
  ;; I believe these %28 and %29 values are SSA values that stem from how
  ;; arrmax and sum initially take the first element of their array arguments
  %28 = phi double [ %.pre146, %for.end.loopexit ], [ %11, %for.cond19.preheader ]
  %29 = phi double [ %24, %for.end.loopexit ], [ %12, %for.cond19.preheader ]
  br i1 %cmp13.i.i, label %for.body.i.i, label %arr_max.exit.i

for.body.i.i:                                     ; preds = %for.end, %for.body.i.i
  %indvars.iv.i.i = phi i64 [ %indvars.iv.next.i.i, %for.body.i.i ], [ 1, %for.end ]
  %m.015.i.i = phi double [ %m.1.i.i, %for.body.i.i ], [ %28, %for.end ]
  %arrayidx1.i.i = getelementptr inbounds double, double* %4, i64 %indvars.iv.i.i
  %30 = load double, double* %arrayidx1.i.i, align 8, !tbaa !3
  %cmp2.i.i = fcmp olt double %m.015.i.i, %30
  %m.1.i.i = select i1 %cmp2.i.i, double %30, double %m.015.i.i
  %indvars.iv.next.i.i = add nuw nsw i64 %indvars.iv.i.i, 1
  %exitcond.not.i.i = icmp eq i64 %indvars.iv.next.i.i, %wide.trip.count.i.i
  br i1 %exitcond.not.i.i, label %arr_max.exit.i, label %for.body.i.i, !llvm.loop !7

arr_max.exit.i:                                   ; preds = %for.body.i.i, %for.end
  %m.0.lcssa.i.i = phi double [ %28, %for.end ], [ %m.1.i.i, %for.body.i.i ]
  br i1 %cmp37.i, label %for.body.preheader.i, label %log_sum_exp.exit

for.body.preheader.i:                             ; preds = %arr_max.exit.i
  ;; In this BB it computes the first iteration of the logsumexp loop for some reason.
  %sub.i135 = fsub double %28, %m.0.lcssa.i.i
  %31 = tail call double @llvm.exp.f64(double %sub.i135) #12
  %add.i136 = fadd double %31, 0.000000e+00
  br i1 %exitcond.not.i99137, label %log_sum_exp.exit, label %for.body.for.body_crit_edge.i, !llvm.loop !12

for.body.for.body_crit_edge.i:                    ; preds = %for.body.preheader.i, %for.body.for.body_crit_edge.i
  %indvars.iv.next.i98139 = phi i64 [ %indvars.iv.next.i98, %for.body.for.body_crit_edge.i ], [ 1, %for.body.preheader.i ]
  %add.i138 = phi double [ %add.i, %for.body.for.body_crit_edge.i ], [ %add.i136, %for.body.preheader.i ]
  %arrayidx.phi.trans.insert.i = getelementptr inbounds double, double* %4, i64 %indvars.iv.next.i98139
  %.pre.i101 = load double, double* %arrayidx.phi.trans.insert.i, align 8, !tbaa !3
  %sub.i = fsub double %.pre.i101, %m.0.lcssa.i.i
  %32 = tail call double @llvm.exp.f64(double %sub.i) #12
  %add.i = fadd double %add.i138, %32
  %indvars.iv.next.i98 = add nuw nsw i64 %indvars.iv.next.i98139, 1
  %exitcond.not.i99 = icmp eq i64 %indvars.iv.next.i98, %wide.trip.count.i.i
  br i1 %exitcond.not.i99, label %log_sum_exp.exit, label %for.body.for.body_crit_edge.i, !llvm.loop !12

log_sum_exp.exit:                                 ; preds = %for.body.for.body_crit_edge.i, %for.body.preheader.i, %arr_max.exit.i
  %semx.0.lcssa.i = phi double [ 0.000000e+00, %arr_max.exit.i ], [ %add.i136, %for.body.preheader.i ], [ %add.i, %for.body.for.body_crit_edge.i ]
  ;; It implements the same algorithm as the unoptimized version.
  %33 = tail call double @llvm.log.f64(double %semx.0.lcssa.i) #12
  %add1.i = fadd double %m.0.lcssa.i.i, %33
  %add47 = fadd double %slse.0143, %add1.i
  %inc49 = add nuw nsw i64 %ix.0141, 1
  %exitcond145.not = icmp eq i64 %inc49, %conv17
  br i1 %exitcond145.not, label %for.end50, label %for.cond19.preheader, !llvm.loop !19

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
declare noalias noundef i8* @malloc(i64) local_unnamed_addr #7

; Function Attrs: inaccessiblemem_or_argmemonly nounwind willreturn
declare void @free(i8* nocapture noundef) local_unnamed_addr #8

; Function Attrs: nounwind ssp uwtable
declare void @__enzyme_autodiff(i8*, ...) local_unnamed_addr #9

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #1

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
attributes #11 = { allocsize(0) }
attributes #12 = { nounwind }
attributes #13 = { allocsize(0,1) }

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
!30 = distinct !{!30, !8, !9}
!31 = distinct !{!31, !8, !9}
!32 = distinct !{!32, !8, !9}
!33 = distinct !{!33, !8, !9}
!34 = distinct !{!34, !8, !9}
!35 = distinct !{!35, !8, !9}
