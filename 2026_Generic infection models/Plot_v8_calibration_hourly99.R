#############################################
# Figures for the v8 re-calibration with hourly_99 (CF-99), 2010-2025
# Run after Calibrate_v8_hourly99.R
#############################################

out_dir <- file.path("Output", "pear_99", "calibration")
load(file.path(out_dir, "calibration_results.RData"))

col_gold <- "#fdae6b"; col_inf <- "#a63603"
cols <- c(v8 = "#54278f", v8_fit = "#969696", v8_fit2_loso = "#1b9e77")
labs <- c(v8 = "v8 (current)", v8_fit = "v8-fit (ID1001)", v8_fit2_loso = "v8-fit2 (LOSO)")
pchs <- c(v8 = 16, v8_fit = 15, v8_fit2_loso = 17)
show <- names(cols)
to_bir <- function(score) 25 + 75 * score       # model score -> BIR scale

# -------------------------------
# F1. (b)-style daily infection risk by season
# -------------------------------
png(file.path(out_dir, "F1_daily_risk_by_season.png"), width = 3200, height = 3100, res = 250)
par(mfrow = c(4, 4), mar = c(2.6, 3.2, 2.2, 3.2), oma = c(3.2, 1.5, 4, 1.5))
for(y in sort(unique(pred$season))){
  x <- pred[pred$season == y, ]
  plot(x$date, x$BIR, type = "h", lwd = 5, lend = 1, col = col_gold, ylim = c(0, 112),
       xaxt = "n", xlab = "", ylab = "", las = 1, cex.axis = 0.8)
  points(x$date[x$infection], rep(108, sum(x$infection)), pch = 25, bg = col_inf, col = col_inf, cex = 0.9)
  for(v in show){
    lines(x$date, to_bir(x[[v]]), col = cols[v], lwd = if(v == "v8_fit") 1.2 else 1.8,
          lty = if(v == "v8_fit") 2 else 1, type = "o", pch = pchs[v], cex = 0.4)
  }
  axis(4, at = to_bir(c(0, 0.5, 1)), labels = c("0", "0.5", "1"), las = 1, cex.axis = 0.8)
  axis.Date(1, at = seq(min(x$date), max(x$date), by = 7), format = "%m-%d", cex.axis = 0.8)
  s <- per_season[per_season$season == y, ]
  title(sprintf("%d   RMSE  v8 %.2f | fit2 %.2f", y, s$RMSE[s$version == "v8"],
                s$RMSE[s$version == "v8_fit2_loso"]), adj = 0, cex.main = 0.9)
}
mtext("Pear BIR (0-100)", side = 2, outer = TRUE, line = 0)
mtext("Model score (0-1):  0 = BIR 25,  1 = BIR 100", side = 4, outer = TRUE, line = 0)
mtext("(b) Daily infection risk in the pear bloom period - CF-99, 2010-2025 (v8-fit2: leave-one-season-out prediction)",
      side = 3, outer = TRUE, line = 2.2, font = 2, cex = 1.05)
par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
plot(0, 0, type = "n", bty = "n", xaxt = "n", yaxt = "n")
legend("bottom", c("Pear BIR (gold standard)", "Pear infection day", labs[show]),
       col = c(col_gold, col_inf, cols[show]), lwd = c(6, NA, 1.8, 1.2, 1.8),
       lty = c(1, NA, 1, 2, 1), pch = c(NA, 25, pchs[show]), pt.bg = col_inf,
       horiz = TRUE, bty = "n", cex = 0.9)
dev.off()

# -------------------------------
# F2. Pooled metrics with 95% season-block bootstrap CI
# -------------------------------
mets <- c("RMSE", "MAE", "Bias", "R2_NSE", "Pearson_r", "Spearman_rho", "Kendall_tau",
          "AUC_infection", "AUC_score_ge_1_3", "AUC_score_ge_2_3")
mlab <- c(RMSE = "RMSE (lower better)", MAE = "MAE (lower better)", Bias = "Bias (0 best)",
          R2_NSE = "R2 (NSE, 1 - SSE/SST)", Pearson_r = "Pearson r", Spearman_rho = "Spearman rho",
          Kendall_tau = "Kendall tau-b", AUC_infection = "AUC: infection day",
          AUC_score_ge_1_3 = "AUC: BIR >= 50", AUC_score_ge_2_3 = "AUC: BIR >= 75")
png(file.path(out_dir, "F2_pooled_metrics_CI.png"), width = 3200, height = 1500, res = 250)
par(mfrow = c(2, 5), mar = c(3, 2.5, 2.8, 1), oma = c(2.5, 2, 3, 0))
pooled_v <- c(v8 = "v8", v8_fit = "v8_fit", v8_fit2_loso = "v8_fit2_LOSO")
for(m in mets){
  est <- sapply(show, function(v) pooled_table[pooled_table$metric == m, pooled_v[v]])
  lo <- ci_lo[m, show]; hi <- ci_hi[m, show]
  yl <- range(c(lo, hi, est, if(m %in% c("Bias", "R2_NSE")) 0), na.rm = TRUE)
  yl <- yl + c(-1, 1) * 0.08 * diff(yl)
  plot(1:3, est, ylim = yl, xlim = c(0.5, 3.5), pch = pchs[show], col = cols[show],
       cex = 1.6, xaxt = "n", xlab = "", ylab = "", las = 1)
  arrows(1:3, lo, 1:3, hi, angle = 90, code = 3, length = 0.04, col = cols[show], lwd = 2)
  if(m %in% c("Bias", "R2_NSE")) abline(h = 0, lty = 3)
  if(grepl("AUC", m)) abline(h = 0.5, lty = 3)
  axis(1, at = 1:3, labels = c("v8", "fit", "fit2"), cex.axis = 0.9)
  text(1:3, est, sprintf("%.2f", est), pos = 4, cex = 0.8, col = cols[show])
  title(mlab[m], cex.main = 0.95)
}
mtext("Pooled performance over 338 bloom days (dots) with 95% season-block bootstrap CI (bars)",
      side = 3, outer = TRUE, line = 1, font = 2)
mtext("v8 = current, fit = v8-fit (ID1001), fit2 = v8-fit2 (hourly_99, leave-one-season-out)",
      side = 1, outer = TRUE, line = 1, cex = 0.85)
dev.off()

# -------------------------------
# F3. Prediction distribution by gold-standard class (observed vs predicted)
# -------------------------------
png(file.path(out_dir, "F3_prediction_by_BIR_class.png"), width = 3000, height = 1250, res = 250)
par(mfrow = c(1, 3), mar = c(4.5, 4.5, 3, 1))
for(v in show){
  boxplot(pred[[v]] ~ pred$BIR, col = adjustcolor(cols[v], 0.35), border = cols[v],
          ylim = c(0, 1), xlab = "Pear BIR (gold standard)", ylab = "Predicted score (0-1)",
          las = 1, outline = FALSE)
  stripchart(pred[[v]] ~ pred$BIR, vertical = TRUE, method = "jitter", jitter = 0.2,
             add = TRUE, pch = 16, cex = 0.4, col = adjustcolor(cols[v], 0.6))
  segments(1:4 - 0.4, c(0, 1/3, 2/3, 1), 1:4 + 0.4, c(0, 1/3, 2/3, 1), col = col_inf, lwd = 3)
  pt <- pooled_table[, pooled_v[v]]; names(pt) <- pooled_table$metric
  title(sprintf("%s\nRMSE %.3f   R2 %.2f   rho %.2f", labs[v], pt["RMSE"], pt["R2_NSE"],
                pt["Spearman_rho"]), cex.main = 1)
}
legend("topleft", "target (BIR - 25) / 75", col = col_inf, lwd = 3, bty = "n", cex = 0.9)
dev.off()

# -------------------------------
# F4. Per-season metrics
# -------------------------------
png(file.path(out_dir, "F4_metrics_by_season.png"), width = 3200, height = 1300, res = 250)
par(mfrow = c(1, 4), mar = c(4.5, 4, 3, 1))
for(m in c("RMSE", "R2_NSE", "Pearson_r", "Spearman_rho")){
  w <- reshape(per_season[per_season$version %in% show, c("season", "version", m)],
               idvar = "season", timevar = "version", direction = "wide")
  yy <- w$season
  vals <- as.matrix(w[, paste0(m, ".", show)])
  xl <- range(vals, na.rm = TRUE); if(m == "R2_NSE") xl[1] <- max(xl[1], -3)
  plot(NA, xlim = xl, ylim = range(yy), xlab = mlab[m], ylab = if(m == "RMSE") "Season" else "",
       las = 1, yaxt = "n")
  axis(2, at = yy, labels = yy, las = 1, cex.axis = 0.75)
  segments(vals[, 1], yy, vals[, 3], yy, col = "grey75")
  for(k in seq_along(show)) points(pmax(vals[, k], xl[1]), yy, pch = pchs[show[k]], col = cols[show[k]])
  if(m != "RMSE") abline(v = 0, lty = 3)
  better <- if(m == "RMSE") sum(vals[, 3] < vals[, 1]) else sum(vals[, 3] > vals[, 1], na.rm = TRUE)
  title(sprintf("%s\nfit2 better than v8: %d / %d seasons", mlab[m], better, nrow(vals)), cex.main = 0.95)
}
legend("bottomleft", labs[show], col = cols[show], pch = pchs[show], bty = "n", cex = 0.85)
dev.off()

# -------------------------------
# F5. ROC curves: infection day and BIR >= 75
# -------------------------------
roc_xy <- function(score, label){
  th <- c(Inf, sort(unique(score), decreasing = TRUE))
  data.frame(fpr = sapply(th, function(t) mean(score[!label] >= t)),
             tpr = sapply(th, function(t) mean(score[label] >= t)))
}
png(file.path(out_dir, "F5_ROC.png"), width = 2600, height = 1300, res = 250)
par(mfrow = c(1, 2), mar = c(4.5, 4.5, 3, 1), pty = "s")
targets <- list("Infection day (BIR = 100)" = list(lab = pred$infection, m = "AUC_infection"),
                "BIR >= 75" = list(lab = pred$BIR >= 75, m = "AUC_score_ge_2_3"))
for(tn in names(targets)){
  plot(0:1, 0:1, type = "n", xlab = "False positive rate", ylab = "True positive rate", las = 1)
  abline(0, 1, lty = 3, col = "grey60")
  for(v in show){
    r <- roc_xy(pred[[v]], targets[[tn]]$lab)
    lines(r$fpr, r$tpr, col = cols[v], lwd = 2.2, type = "s", lty = if(v == "v8_fit") 2 else 1)
  }
  au <- sapply(show, function(v) pooled_table[pooled_table$metric == targets[[tn]]$m, pooled_v[v]])
  legend("bottomright", sprintf("%s  AUC %.3f", labs[show], au), col = cols[show], lwd = 2.2,
         lty = c(1, 2, 1), bty = "n", cex = 0.85)
  title(sprintf("ROC: %s (n+ = %d)", tn, sum(targets[[tn]]$lab)), cex.main = 1)
}
dev.off()

# -------------------------------
# F6. Calibration design: single season vs leave-one-season-out, parameter stability
# -------------------------------
png(file.path(out_dir, "F6_calibration_design.png"), width = 3200, height = 1400, res = 250)
layout(matrix(1:4, nrow = 1), widths = c(1, 1, 1, 2.2))
par(mar = c(4.5, 4.2, 3.2, 1))
for(m in c("RMSE", "Spearman_rho", "AUC_infection")){
  v <- single[[m]]
  loso <- pooled_table[pooled_table$metric == m, "v8_fit2_LOSO"]
  base <- pooled_table[pooled_table$metric == m, "v8"]
  yl <- range(c(v, loso, base), na.rm = TRUE)
  boxplot(v, col = "#c7e9c0", border = "#238b45", ylim = yl, las = 1, ylab = mlab[m],
          main = "", outline = FALSE)
  stripchart(v, vertical = TRUE, method = "jitter", add = TRUE, pch = 16, col = "#238b45", cex = 0.7)
  abline(h = loso, col = cols["v8_fit2_loso"], lwd = 2.5)
  abline(h = base, col = cols["v8"], lwd = 2, lty = 2)
  title(sprintf("%s\nsingle-season fits (n = 16)", m), cex.main = 0.9)
  if(m == "RMSE") legend("topright", c("LOSO (fit2)", "v8 current"), col = cols[c("v8_fit2_loso", "v8")],
                         lwd = c(2.5, 2), lty = c(1, 2), bty = "n", cex = 0.8)
}
par(mar = c(4.5, 7.5, 3.2, 1))
pr <- param_rows
norm <- function(v, k) (v - fit_spec$lower[k]) / (fit_spec$upper[k] - fit_spec$lower[k])
plot(NA, xlim = c(0, 1), ylim = c(0.5, nrow(fit_spec) + 0.5), yaxt = "n",
     xlab = "Position within search bounds (0 = lower, 1 = upper)", ylab = "")
axis(2, at = seq_len(nrow(fit_spec)),
     labels = sprintf("%s\n[%g, %g]", fit_spec$name, fit_spec$lower, fit_spec$upper),
     las = 1, cex.axis = 0.75)
for(k in seq_len(nrow(fit_spec))){
  nm <- fit_spec$name[k]
  s1 <- pr[pr$design == "SINGLE", nm]; lo <- pr[pr$design == "LOSO", nm]
  points(norm(s1, k), k + runif(length(s1), 0.08, 0.3), pch = 1, col = "#74c476", cex = 0.9)
  points(norm(lo, k), k - runif(length(lo), 0.08, 0.3), pch = 17, col = cols["v8_fit2_loso"], cex = 0.8)
  points(norm(param_table$v8[k], k), k, pch = 16, col = cols["v8"], cex = 1.6)
  points(norm(param_table$v8_fit2[k], k), k, pch = 4, col = "black", cex = 1.8, lwd = 2.5)
}
abline(v = c(0, 1), lty = 3, col = "grey50")
legend("bottomright", c("v8 current", "v8-fit2 (all seasons)", "LOSO fits (16)", "single-season fits (16)"),
       pch = c(16, 4, 17, 1), col = c(cols["v8"], "black", cols["v8_fit2_loso"], "#74c476"),
       pt.lwd = c(1, 2.5, 1, 1), bty = "n", cex = 0.8, bg = "white")
title("Parameter estimates: stability across calibration designs", cex.main = 0.95)
dev.off()

# -------------------------------
# F7. Component diagnostics for v8-fit2: N vs pear EIP, I vs pear W
# -------------------------------
png(file.path(out_dir, "F7_components.png"), width = 3000, height = 1250, res = 250)
par(mfrow = c(1, 3), mar = c(4.5, 4.5, 3, 1))
plot(pred$EIP, pred$N_fit2, pch = 16, cex = 0.6, col = ifelse(pred$H, col_inf, "grey50"),
     xlab = "Pear EIP", ylab = "v8-fit2 N", las = 1)
abline(v = 100, lty = 3)
title(sprintf("N vs EIP   (r = %.2f)", cor(pred$EIP, pred$N_fit2)), cex.main = 1)
legend("bottomright", c("pear H = TRUE", "pear H = FALSE"), col = c(col_inf, "grey50"), pch = 16, bty = "n", cex = 0.85)
plot(pred$temp, pred$N_fit2, pch = 16, cex = 0.6, col = ifelse(pred$T, col_inf, "grey50"),
     xlab = expression("Daily mean temperature ("*degree*C*")"), ylab = "v8-fit2 N", las = 1)
abline(v = 15.6, lty = 3)
title(sprintf("N vs temperature   (r = %.2f)", cor(pred$temp, pred$N_fit2)), cex.main = 1)
legend("bottomright", c("pear T = TRUE", "pear T = FALSE"), col = c(col_inf, "grey50"), pch = 16, bty = "n", cex = 0.85)
boxplot(list("v8: W=F" = pred$I_v8[!pred$W], "v8: W=T" = pred$I_v8[pred$W],
             "fit2: W=F" = pred$I_fit2[!pred$W], "fit2: W=T" = pred$I_fit2[pred$W]),
        col = adjustcolor(c(cols["v8"], cols["v8"], cols["v8_fit2_loso"], cols["v8_fit2_loso"]), 0.4),
        ylab = "v8 I (wetness)", las = 1, ylim = c(0, 1))
title("I by pear wetness flag W", cex.main = 1)
dev.off()
