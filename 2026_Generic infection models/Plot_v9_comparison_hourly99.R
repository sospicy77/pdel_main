#############################################
# Figures: v8 vs v8-fit2 vs v9 against the pear fire blight model
# (hourly_99 / CF-99, 2010-2025). Run after Calibrate_v9_hourly99.R
#############################################

out_dir <- file.path("Output", "pear_99", "calibration_v9")
load(file.path(out_dir, "v9_results.RData"))

col_gold <- "#fdae6b"; col_inf <- "#a63603"
cols <- c(v8 = "#54278f", v8_fit2 = "#1b9e77", v9_D = "#80b1d3", v9_T = "#fb8072", v9 = "#d95f02")
labs <- c(v8 = "v8 (current)", v8_fit2 = "v8-fit2 (base)", v9_D = "v9-D (+dispersal)",
          v9_T = "v9-T (+temperature in I)", v9 = "v9 (+both)")
pchs <- c(v8 = 16, v8_fit2 = 17, v9_D = 15, v9_T = 18, v9 = 8)
main3 <- c("v8", "v8_fit2", "v9")
all5  <- names(versions)
to_bir <- function(s) 25 + 75 * s
mlab <- c(RMSE = "RMSE (lower better)", MAE = "MAE (lower better)", Bias = "Bias (0 best)",
          R2_NSE = "R2 (NSE)", r2 = "r2", Pearson_r = "Pearson r", Spearman_rho = "Spearman rho",
          Kendall_tau = "Kendall tau-b", AUC_infection = "AUC: infection day",
          AUC_score_ge_1_3 = "AUC: BIR >= 50", AUC_score_ge_2_3 = "AUC: BIR >= 75",
          Class_acc = "Class accuracy")

# -------------------------------
# G1. (b)-style daily infection risk by season
# -------------------------------
png(file.path(out_dir, "G1_daily_risk_by_season.png"), width = 3200, height = 3100, res = 250)
par(mfrow = c(4, 4), mar = c(2.6, 3.2, 2.2, 3.2), oma = c(3.2, 1.5, 4, 1.5))
for(y in sort(unique(pred$season))){
  x <- pred[pred$season == y, ]
  plot(x$date, x$BIR, type = "h", lwd = 5, lend = 1, col = col_gold, ylim = c(0, 112),
       xaxt = "n", xlab = "", ylab = "", las = 1, cex.axis = 0.8)
  points(x$date[x$infection], rep(108, sum(x$infection)), pch = 25, bg = col_inf, col = col_inf, cex = 0.9)
  for(v in main3) lines(x$date, to_bir(x[[versions[v]]]), col = cols[v], lwd = 1.8,
                        type = "o", pch = pchs[v], cex = 0.4)
  axis(4, at = to_bir(c(0, 0.5, 1)), labels = c("0", "0.5", "1"), las = 1, cex.axis = 0.8)
  axis.Date(1, at = seq(min(x$date), max(x$date), by = 7), format = "%m-%d", cex.axis = 0.8)
  s <- per_season[per_season$season == y, ]
  title(sprintf("%d  RMSE %.2f | %.2f | %.2f", y, s$RMSE[s$version == "v8"],
                s$RMSE[s$version == "v8_fit2"], s$RMSE[s$version == "v9"]), adj = 0, cex.main = 0.9)
}
mtext("Pear BIR (0-100)", side = 2, outer = TRUE, line = 0)
mtext("Model score (0-1):  0 = BIR 25,  1 = BIR 100", side = 4, outer = TRUE, line = 0)
mtext("(b) Daily infection risk in the pear bloom period - CF-99, 2010-2025 (fitted versions: leave-one-season-out)",
      side = 3, outer = TRUE, line = 2.2, font = 2, cex = 1.05)
par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
plot(0, 0, type = "n", bty = "n", xaxt = "n", yaxt = "n")
legend("bottom", c("Pear BIR (gold standard)", "Pear infection day", labs[main3],
                   "title: RMSE v8 | v8-fit2 | v9"),
       col = c(col_gold, col_inf, cols[main3], NA), lwd = c(6, NA, 1.8, 1.8, 1.8, NA),
       pch = c(NA, 25, pchs[main3], NA), pt.bg = col_inf, horiz = TRUE, bty = "n", cex = 0.85)
dev.off()

# -------------------------------
# G2. Pooled metrics with 95% CI (v8, v8-fit2, v9)
# -------------------------------
mets <- c("RMSE", "MAE", "Bias", "R2_NSE", "Pearson_r", "Spearman_rho", "Kendall_tau",
          "AUC_infection", "AUC_score_ge_1_3", "AUC_score_ge_2_3")
png(file.path(out_dir, "G2_pooled_metrics_CI.png"), width = 3200, height = 1500, res = 250)
par(mfrow = c(2, 5), mar = c(3, 2.5, 2.8, 1), oma = c(2.5, 2, 3, 0))
for(m in mets){
  est <- pooled[m, main3]; lo <- ci_lo[m, main3]; hi <- ci_hi[m, main3]
  yl <- range(c(lo, hi, est, if(m %in% c("Bias", "R2_NSE")) 0), na.rm = TRUE)
  yl <- yl + c(-1, 1) * 0.1 * diff(yl)
  plot(1:3, est, ylim = yl, xlim = c(0.5, 3.6), pch = pchs[main3], col = cols[main3],
       cex = 1.6, xaxt = "n", xlab = "", ylab = "", las = 1)
  arrows(1:3, lo, 1:3, hi, angle = 90, code = 3, length = 0.04, col = cols[main3], lwd = 2)
  if(m %in% c("Bias", "R2_NSE")) abline(h = 0, lty = 3)
  if(grepl("AUC", m)) abline(h = 0.5, lty = 3)
  axis(1, at = 1:3, labels = c("v8", "v8-fit2", "v9"), cex.axis = 0.85)
  text(1:3, est, sprintf("%.2f", est), pos = 4, cex = 0.8, col = cols[main3])
  title(mlab[m], cex.main = 0.95)
}
mtext("Pooled out-of-sample performance over 338 bloom days with 95% season-block bootstrap CI",
      side = 3, outer = TRUE, line = 1, font = 2)
mtext("v8 = current parameters (not fitted); v8-fit2 and v9 = leave-one-season-out predictions",
      side = 1, outer = TRUE, line = 1, cex = 0.85)
dev.off()

# -------------------------------
# G3. Improvement relative to v8 and to v8-fit2 (change with 95% CI)
# -------------------------------
dm <- c("RMSE", "MAE", "R2_NSE", "Pearson_r", "Spearman_rho", "Kendall_tau",
        "AUC_infection", "AUC_score_ge_1_3", "AUC_score_ge_2_3", "Class_acc")
sign_better <- ifelse(dm %in% c("RMSE", "MAE"), -1, 1)    # positive = improvement
png(file.path(out_dir, "G3_improvement_CI.png"), width = 3000, height = 1400, res = 250)
par(mfrow = c(1, 3), mar = c(4.5, 9, 3, 1))
for(cmp in c("v8_fit2_vs_v8", "v9_vs_v8", "v9_vs_v8_fit2")){
  ch <- pooled_table[match(dm, pooled_table$metric), paste0(cmp, ".change")] * sign_better
  lo <- pooled_table[match(dm, pooled_table$metric), paste0(cmp, ".CI_low")] * sign_better
  hi <- pooled_table[match(dm, pooled_table$metric), paste0(cmp, ".CI_high")] * sign_better
  lo2 <- pmin(lo, hi); hi2 <- pmax(lo, hi)
  sig <- lo2 > 0 | hi2 < 0
  yy <- rev(seq_along(dm))
  plot(ch, yy, xlim = range(c(lo2, hi2, 0)), yaxt = "n", ylab = "", pch = 16,
       col = ifelse(sig & ch > 0, "#1a9850", ifelse(sig & ch < 0, "#d73027", "grey50")),
       xlab = "Improvement (positive = better; RMSE/MAE sign flipped)", cex = 1.3)
  segments(lo2, yy, hi2, yy, lwd = 2,
           col = ifelse(sig & ch > 0, "#1a9850", ifelse(sig & ch < 0, "#d73027", "grey50")))
  abline(v = 0, lty = 2)
  axis(2, at = yy, labels = mlab[dm], las = 1, cex.axis = 0.85)
  title(c(v8_fit2_vs_v8 = "v8-fit2 vs v8", v9_vs_v8 = "v9 vs v8",
          v9_vs_v8_fit2 = "v9 vs v8-fit2 (effect of add-on modules)")[cmp], cex.main = 1)
}
legend("bottomright", c("significant improvement", "significant decline", "not significant"),
       col = c("#1a9850", "#d73027", "grey50"), pch = 16, bty = "n", cex = 0.8)
dev.off()

# -------------------------------
# G4. Prediction distribution by gold-standard class
# -------------------------------
png(file.path(out_dir, "G4_prediction_by_BIR_class.png"), width = 3000, height = 1250, res = 250)
par(mfrow = c(1, 3), mar = c(4.5, 4.5, 3.2, 1))
for(v in main3){
  p <- pred[[versions[v]]]
  boxplot(p ~ pred$BIR, col = adjustcolor(cols[v], 0.35), border = cols[v], ylim = c(0, 1),
          xlab = "Pear BIR (gold standard)", ylab = "Predicted score (0-1)", las = 1, outline = FALSE)
  stripchart(p ~ pred$BIR, vertical = TRUE, method = "jitter", jitter = 0.2, add = TRUE,
             pch = 16, cex = 0.4, col = adjustcolor(cols[v], 0.6))
  segments(1:4 - 0.4, c(0, 1/3, 2/3, 1), 1:4 + 0.4, c(0, 1/3, 2/3, 1), col = col_inf, lwd = 3)
  title(sprintf("%s\nRMSE %.3f   R2 %.2f   r %.2f   rho %.2f", labs[v], pooled["RMSE", v],
                pooled["R2_NSE", v], pooled["Pearson_r", v], pooled["Spearman_rho", v]), cex.main = 0.95)
}
legend("topleft", "target (BIR - 25) / 75", col = col_inf, lwd = 3, bty = "n", cex = 0.9)
dev.off()

# -------------------------------
# G5. Per-season metrics
# -------------------------------
png(file.path(out_dir, "G5_metrics_by_season.png"), width = 3200, height = 1300, res = 250)
par(mfrow = c(1, 4), mar = c(4.5, 4, 3.2, 1))
for(m in c("RMSE", "R2_NSE", "Pearson_r", "Spearman_rho")){
  w <- reshape(per_season[per_season$version %in% main3, c("season", "version", m)],
               idvar = "season", timevar = "version", direction = "wide")
  yy <- w$season; vals <- as.matrix(w[, paste0(m, ".", main3)])
  xl <- range(vals, na.rm = TRUE); if(m == "R2_NSE") xl[1] <- max(xl[1], -3)
  plot(NA, xlim = xl, ylim = range(yy), xlab = mlab[m], ylab = if(m == "RMSE") "Season" else "",
       las = 1, yaxt = "n")
  axis(2, at = yy, labels = yy, las = 1, cex.axis = 0.75)
  segments(apply(vals, 1, min, na.rm = TRUE), yy, apply(vals, 1, max, na.rm = TRUE), yy, col = "grey80")
  for(k in seq_along(main3)) points(pmax(vals[, k], xl[1]), yy, pch = pchs[main3[k]], col = cols[main3[k]])
  if(m != "RMSE") abline(v = 0, lty = 3)
  b8  <- if(m == "RMSE") sum(vals[, 3] < vals[, 1]) else sum(vals[, 3] > vals[, 1], na.rm = TRUE)
  bf2 <- if(m == "RMSE") sum(vals[, 3] < vals[, 2]) else sum(vals[, 3] > vals[, 2], na.rm = TRUE)
  title(sprintf("%s\nv9 better: vs v8 %d/16, vs fit2 %d/16", mlab[m], b8, bf2), cex.main = 0.9)
}
legend("bottomleft", labs[main3], col = cols[main3], pch = pchs[main3], bty = "n", cex = 0.8)
dev.off()

# -------------------------------
# G6. ROC curves
# -------------------------------
roc_xy <- function(score, label){
  th <- c(Inf, sort(unique(score), decreasing = TRUE))
  data.frame(fpr = sapply(th, function(t) mean(score[!label] >= t)),
             tpr = sapply(th, function(t) mean(score[label] >= t)))
}
png(file.path(out_dir, "G6_ROC.png"), width = 3300, height = 1250, res = 250)
par(mfrow = c(1, 3), mar = c(4.5, 4.5, 3, 1), pty = "s")
tg <- list("Infection day (BIR = 100)" = list(l = pred$infection, m = "AUC_infection"),
           "BIR >= 75" = list(l = pred$BIR >= 75, m = "AUC_score_ge_2_3"),
           "BIR >= 50" = list(l = pred$BIR >= 50, m = "AUC_score_ge_1_3"))
for(tn in names(tg)){
  plot(0:1, 0:1, type = "n", xlab = "False positive rate", ylab = "True positive rate", las = 1)
  abline(0, 1, lty = 3, col = "grey60")
  for(v in main3){
    r <- roc_xy(pred[[versions[v]]], tg[[tn]]$l)
    lines(r$fpr, r$tpr, col = cols[v], lwd = 2.2, type = "s")
  }
  legend("bottomright", sprintf("%s  AUC %.3f", labs[main3], pooled[tg[[tn]]$m, main3]),
         col = cols[main3], lwd = 2.2, bty = "n", cex = 0.8)
  title(sprintf("ROC: %s (n+ = %d)", tn, sum(tg[[tn]]$l)), cex.main = 1)
}
dev.off()

# -------------------------------
# G7. Module ablation: base, +D, +T, +both
# -------------------------------
am <- c("RMSE", "R2_NSE", "Pearson_r", "Spearman_rho", "AUC_infection", "AUC_score_ge_1_3",
        "AUC_score_ge_2_3", "Class_acc")
abl <- c("v8_fit2", "v9_D", "v9_T", "v9")
png(file.path(out_dir, "G7_module_ablation.png"), width = 3200, height = 1500, res = 250)
par(mfrow = c(2, 4), mar = c(3.5, 3, 3, 1), oma = c(2.5, 1, 3, 0))
for(m in am){
  est <- pooled[m, abl]; lo <- ci_lo[m, abl]; hi <- ci_hi[m, abl]
  yl <- range(c(lo, hi), na.rm = TRUE); yl <- yl + c(-1, 1) * 0.1 * diff(yl)
  bp <- barplot(est, col = adjustcolor(cols[abl], 0.7), border = cols[abl], ylim = yl, xpd = FALSE,
                names.arg = c("base", "+D", "+T", "+D+T"), las = 1, cex.names = 0.9)
  arrows(bp, lo, bp, hi, angle = 90, code = 3, length = 0.04, lwd = 1.5)
  abline(h = pooled[m, "v8"], col = cols["v8"], lty = 2, lwd = 2)
  text(bp, est, sprintf("%.3f", est), pos = 3, cex = 0.75, offset = 0.8)
  title(mlab[m], cex.main = 0.95)
}
mtext("Contribution of the add-on modules (leave-one-season-out, 95% CI); dashed line = v8 (current)",
      side = 3, outer = TRUE, line = 1, font = 2)
mtext("base = v8-fit2;  +D = rain splash + insect dispersal;  +T = temperature response f_T in I;  +D+T = v9",
      side = 1, outer = TRUE, line = 1, cex = 0.85)
dev.off()

# -------------------------------
# G8. v9 component diagnostics
# -------------------------------
png(file.path(out_dir, "G8_v9_components.png"), width = 3200, height = 1250, res = 250)
par(mfrow = c(1, 4), mar = c(4.5, 4.5, 3, 1))
wet <- pred$rain > 0
plot(pred$rain[wet], pred$D_v9[wet], pch = 16, cex = 0.7, col = ifelse(pred$W[wet], col_inf, "grey50"),
     xlab = "Daily rain (mm, log scale)", ylab = "v9 D", las = 1, log = "x",
     xlim = c(0.05, max(pred$rain) + 1), ylim = range(pred$D_v9))
points(rep(0.05, sum(pred$rain == 0)), pred$D_v9[pred$rain == 0], pch = 16, cex = 0.7, col = "grey30")
text(0.05, min(pred$D_v9), "no rain", pos = 4, cex = 0.75)
title("D: rain splash + insect floor", cex.main = 1)
legend("bottomright", c("pear W = TRUE", "pear W = FALSE"), col = c(col_inf, "grey50"), pch = 16, bty = "n", cex = 0.8)
plot(pred$temp, pred$fT_v9, pch = 16, cex = 0.7, col = ifelse(pred$T, col_inf, "grey50"),
     xlab = expression("Daily mean temperature ("*degree*C*")"), ylab = expression("v9 "*f[T](T)*" in I"), las = 1)
abline(v = 15.6, lty = 3)
title("I add-on: temperature response", cex.main = 1)
legend("bottomright", c("pear T = TRUE", "pear T = FALSE"), col = c(col_inf, "grey50"), pch = 16, bty = "n", cex = 0.8)
grp <- interaction(ifelse(pred$W, "W+", "W-"), ifelse(pred$T, "T+", "T-"), sep = " ")
boxplot(pred$I_v9 ~ grp, col = adjustcolor(cols["v9"], 0.35), border = cols["v9"],
        ylab = "v9 I", xlab = "pear flags", las = 1, ylim = c(0, 1))
title("I by pear wetness / temperature flags", cex.main = 1)
plot(pred$EIP, pred$N_v9, pch = 16, cex = 0.7, col = ifelse(pred$H, col_inf, "grey50"),
     xlab = "Pear EIP", ylab = "v9 N", las = 1)
abline(v = 100, lty = 3)
title(sprintf("N vs pear EIP (r = %.2f)", cor(pred$EIP, pred$N_v9)), cex.main = 1)
dev.off()

# -------------------------------
# G9. Parameters across versions and LOSO stability
# -------------------------------
png(file.path(out_dir, "G9_parameters.png"), width = 2800, height = 1500, res = 250)
par(mar = c(4.5, 8, 3, 1))
nm <- spec_all$name
norm <- function(v, k) (v - spec_all$lower[k]) / (spec_all$upper[k] - spec_all$lower[k])
plot(NA, xlim = c(0, 1), ylim = c(0.5, length(nm) + 0.5), yaxt = "n", ylab = "",
     xlab = "Position within search bounds (0 = lower, 1 = upper)")
axis(2, at = seq_along(nm), labels = sprintf("%s\n[%g, %g]", nm, spec_all$lower, spec_all$upper),
     las = 1, cex.axis = 0.75)
for(k in seq_along(nm)){
  lo <- fit_rows[fit_rows$variant == "v9" & fit_rows$design == "LOSO", nm[k]]
  points(norm(lo, k), k - runif(length(lo), 0.1, 0.3), pch = 17, col = adjustcolor(cols["v9"], 0.6), cex = 0.8)
  if(!is.na(param_table$v8[k])) points(norm(param_table$v8[k], k), k, pch = 16, col = cols["v8"], cex = 1.6)
  if(!is.na(param_table$v8_fit2[k])) points(norm(param_table$v8_fit2[k], k), k + 0.15, pch = 17, col = cols["v8_fit2"], cex = 1.4)
  points(norm(param_table$v9[k], k), k, pch = 4, col = "black", cex = 1.8, lwd = 2.5)
}
abline(v = c(0, 1), lty = 3, col = "grey50")
legend("right", c("v8 (current)", "v8-fit2", "v9 (all seasons)", "v9 LOSO fits (16)"),
       pch = c(16, 17, 4, 17), col = c(cols["v8"], cols["v8_fit2"], "black", adjustcolor(cols["v9"], 0.6)),
       pt.lwd = c(1, 1, 2.5, 1), bty = "n", cex = 0.8, bg = "white")
title("Calibrated parameters by version (v9 LOSO estimates show stability)", cex.main = 1)
dev.off()
