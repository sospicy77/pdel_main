#############################################
# Final comparison against the pear fire blight model (gold standard),
# hourly_99 / CF-99, 2010-2025 bloom periods (338 days).
#   v8          : current v8 parameters (not fitted)
#   v8-fit2     : v8 base modules, fitted (Calibrate_v8_hourly99.R)
#   v9 (prev)   : base + D/T add-ons, 8 parameters (Calibrate_v9_hourly99.R)
#   v9-param    : plan A + extra parameters rh_thresh_M, k_I (search stage 1)
#   v9 (final)  : + degree-hour growth and previous-day rain wetting, gp = 4
# All fitted versions: leave-one-season-out (out-of-sample) predictions.
#############################################

pear_dir <- file.path("Output", "pear_99")
out_dir  <- file.path(pear_dir, "final_v9")
n_boot   <- 2000
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

c8 <- new.env(); load(file.path(pear_dir, "calibration", "calibration_results.RData"), envir = c8)
c9 <- new.env(); load(file.path(pear_dir, "calibration_v9", "v9_results.RData"), envir = c9)
sv <- new.env(); load(file.path(pear_dir, "search_v9", "search_results.RData"), envir = sv)

pred <- c8$pred[, c("season", "date", "temp", "rh", "rain", "BIR", "score", "infection",
                    "H", "W", "T", "EIP")]
stopifnot(all(pred$date == c9$pred$date), all(pred$date == sv$frame$date[sv$bl]))
pred$v8       <- c8$pred$v8
pred$v8_fit2  <- c8$pred$v8_fit2_loso
pred$v9_prev  <- c9$pred$v9_loso
pred$v9_param <- sv$stage1$loso_pred
pred$v9_final <- sv$final$loso_pred
versions <- c("v8", "v8_fit2", "v9_prev", "v9_param", "v9_final")
labs <- c(v8 = "v8 (current)", v8_fit2 = "v8-fit2", v9_prev = "v9 (previous)",
          v9_param = "v9 + parameters", v9_final = "v9 (final)")
cols <- c(v8 = "#54278f", v8_fit2 = "#1b9e77", v9_prev = "#fc8d62", v9_param = "#8da0cb",
          v9_final = "#e7298a")
pchs <- c(v8 = 16, v8_fit2 = 17, v9_prev = 15, v9_param = 18, v9_final = 8)

# Final-model components (all-season fit)
src <- new.env()
txt <- readLines("Search_v9_hourly99.R")
eval(parse(text = txt[grep("^temp_response <- function", txt):(grep("^# 3. CALIBRATION", txt) - 2)]),
     envir = src)
src$frame <- sv$frame
comp <- src$model_risk(modifyList(sv$defaults, as.list(sv$final$all$par)), sv$final$cfg)
pred$N_final <- comp$N[sv$bl]; pred$D_final <- comp$D[sv$bl]
pred$W_final <- comp$W[sv$bl]; pred$I_final <- comp$I[sv$bl]
pred$rain_prev <- sv$frame$rain_prev[sv$bl]; pred$dh183 <- sv$frame$dh183[sv$bl]
write.csv(pred, file.path(out_dir, "final_predictions_bloom_days.csv"), row.names = FALSE)

# -------------------------------
# Metrics and bootstrap
# -------------------------------
auc <- function(s, l){ p <- s[l]; n <- s[!l]
  if(!length(p) || !length(n)) return(NA); mean(outer(p, n, ">") + 0.5 * outer(p, n, "==")) }
sc <- function(a, b, m) if(length(a) < 3 || sd(a) == 0 || sd(b) == 0) NA else cor(a, b, method = m)
metrics <- function(p, y, inf){
  nearest <- c(0, 1/3, 2/3, 1)[apply(abs(outer(p, c(0, 1/3, 2/3, 1), "-")), 1, which.min)]
  c(RMSE = sqrt(mean((p - y)^2)), MAE = mean(abs(p - y)), Bias = mean(p - y),
    R2_NSE = 1 - sum((y - p)^2) / sum((y - mean(y))^2), Pearson_r = sc(p, y, "pearson"),
    Spearman_rho = sc(p, y, "spearman"), Kendall_tau = sc(p, y, "kendall"),
    AUC_infection = auc(p, inf), AUC_score_ge_1_3 = auc(p, y >= 1/3 - 1e-9),
    AUC_score_ge_2_3 = auc(p, y >= 2/3 - 1e-9), Class_acc = mean(abs(nearest - y) < 1e-9))
}
pooled <- sapply(versions, function(v) metrics(pred[[v]], pred$score, pred$infection))
set.seed(99)
rows <- split(seq_len(nrow(pred)), pred$season)
boot <- replicate(n_boot, {
  i <- unlist(rows[sample(names(rows), replace = TRUE)], use.names = FALSE)
  sapply(versions, function(v) metrics(pred[[v]][i], pred$score[i], pred$infection[i]))
}, simplify = "array")
ci_lo <- apply(boot, c(1, 2), quantile, 0.025, na.rm = TRUE)
ci_hi <- apply(boot, c(1, 2), quantile, 0.975, na.rm = TRUE)
cmp_pairs <- list(final_vs_v8 = c("v9_final", "v8"), final_vs_v8fit2 = c("v9_final", "v8_fit2"),
                  final_vs_v9prev = c("v9_final", "v9_prev"), param_vs_v9prev = c("v9_param", "v9_prev"),
                  final_vs_param = c("v9_final", "v9_param"))
diff_tab <- do.call(rbind, lapply(names(cmp_pairs), function(nm){
  a <- cmp_pairs[[nm]][1]; b <- cmp_pairs[[nm]][2]; d <- boot[, a, ] - boot[, b, ]
  data.frame(comparison = nm, metric = rownames(pooled), change = pooled[, a] - pooled[, b],
             CI_low = apply(d, 1, quantile, 0.025, na.rm = TRUE),
             CI_high = apply(d, 1, quantile, 0.975, na.rm = TRUE), row.names = NULL)
}))
write.csv(data.frame(metric = rownames(pooled), round(pooled, 4)),
          file.path(out_dir, "final_metrics_pooled.csv"), row.names = FALSE)
write.csv(diff_tab, file.path(out_dir, "final_metric_changes_CI.csv"), row.names = FALSE)

per_season <- do.call(rbind, lapply(split(pred, pred$season), function(x)
  do.call(rbind, lapply(versions, function(v)
    data.frame(season = x$season[1], version = v, t(metrics(x[[v]], x$score, x$infection)))))))
write.csv(per_season, file.path(out_dir, "final_metrics_by_season.csv"), row.names = FALSE)

fp <- do.call(rbind, lapply(sv$final$folds, `[[`, "par"))
param_tab <- data.frame(parameter = names(sv$final$all$par), value = round(sv$final$all$par, 4),
                        LOSO_min = round(apply(fp, 2, min), 4), LOSO_max = round(apply(fp, 2, max), 4),
                        lower = sv$bounds[names(sv$final$all$par), 1],
                        upper = sv$bounds[names(sv$final$all$par), 2], row.names = NULL)
write.csv(param_tab, file.path(out_dir, "final_v9_parameters.csv"), row.names = FALSE)

sink(file.path(out_dir, "final_report.txt"), split = TRUE)
cat("Final v9:", sv$final$label, "\n\nParameters (all seasons; LOSO range)\n")
print(param_tab, row.names = FALSE)
cat("\nPooled out-of-sample metrics (338 bloom days)\n")
print(round(pooled, 3))
cat("\nChanges with 95% season-block bootstrap CI\n")
print(cbind(diff_tab[, 1:2], round(diff_tab[, 3:5], 3)), row.names = FALSE)
cat("\nMean prediction by BIR class\n")
print(round(sapply(versions, function(v) tapply(pred[[v]], pred$BIR, mean)), 3))
sink()

# -------------------------------
# Figures
# -------------------------------
mlab <- c(RMSE = "RMSE (lower better)", MAE = "MAE (lower better)", Bias = "Bias (0 best)",
          R2_NSE = "R2 (NSE)", Pearson_r = "Pearson r", Spearman_rho = "Spearman rho",
          Kendall_tau = "Kendall tau-b", AUC_infection = "AUC: infection day",
          AUC_score_ge_1_3 = "AUC: BIR >= 50", AUC_score_ge_2_3 = "AUC: BIR >= 75",
          Class_acc = "Class accuracy")
to_bir <- function(s) 25 + 75 * s
show3 <- c("v8", "v9_prev", "v9_final")
col_gold <- "#fdae6b"; col_inf <- "#a63603"

# H1 (b)-style daily risk by season
png(file.path(out_dir, "H1_daily_risk_by_season.png"), width = 3200, height = 3100, res = 250)
par(mfrow = c(4, 4), mar = c(2.6, 3.2, 2.2, 3.2), oma = c(3.2, 1.5, 4, 1.5))
for(y in sort(unique(pred$season))){
  x <- pred[pred$season == y, ]
  plot(x$date, x$BIR, type = "h", lwd = 5, lend = 1, col = col_gold, ylim = c(0, 112),
       xaxt = "n", xlab = "", ylab = "", las = 1, cex.axis = 0.8)
  points(x$date[x$infection], rep(108, sum(x$infection)), pch = 25, bg = col_inf, col = col_inf, cex = 0.9)
  for(v in show3) lines(x$date, to_bir(x[[v]]), col = cols[v], lwd = if(v == "v9_final") 2.2 else 1.5,
                        type = "o", pch = pchs[v], cex = 0.4)
  axis(4, at = to_bir(c(0, 0.5, 1)), labels = c("0", "0.5", "1"), las = 1, cex.axis = 0.8)
  axis.Date(1, at = seq(min(x$date), max(x$date), by = 7), format = "%m-%d", cex.axis = 0.8)
  s <- per_season[per_season$season == y, ]
  title(sprintf("%d  RMSE %.2f | %.2f | %.2f", y, s$RMSE[s$version == "v8"],
                s$RMSE[s$version == "v9_prev"], s$RMSE[s$version == "v9_final"]), adj = 0, cex.main = 0.9)
}
mtext("Pear BIR (0-100)", side = 2, outer = TRUE, line = 0)
mtext("Model score (0-1):  0 = BIR 25,  1 = BIR 100", side = 4, outer = TRUE, line = 0)
mtext("(b) Daily infection risk - CF-99 2010-2025 (fitted versions: leave-one-season-out)",
      side = 3, outer = TRUE, line = 2.2, font = 2, cex = 1.05)
par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
plot(0, 0, type = "n", bty = "n", xaxt = "n", yaxt = "n")
legend("bottom", c("Pear BIR (gold standard)", "Pear infection day", labs[show3], "title: RMSE v8 | v9 prev | v9 final"),
       col = c(col_gold, col_inf, cols[show3], NA), lwd = c(6, NA, 1.5, 1.5, 2.2, NA),
       pch = c(NA, 25, pchs[show3], NA), pt.bg = col_inf, horiz = TRUE, bty = "n", cex = 0.85)
dev.off()

# H2 pooled metrics with CI, all versions
mets <- c("RMSE", "MAE", "Bias", "R2_NSE", "Pearson_r", "Spearman_rho", "Kendall_tau",
          "AUC_infection", "AUC_score_ge_1_3", "AUC_score_ge_2_3")
png(file.path(out_dir, "H2_pooled_metrics_CI.png"), width = 3400, height = 1550, res = 250)
par(mfrow = c(2, 5), mar = c(3.6, 2.5, 2.8, 1), oma = c(2.5, 2, 3, 0))
for(m in mets){
  est <- pooled[m, versions]; lo <- ci_lo[m, versions]; hi <- ci_hi[m, versions]
  yl <- range(c(lo, hi, if(m %in% c("Bias", "R2_NSE")) 0), na.rm = TRUE); yl <- yl + c(-1, 1) * 0.08 * diff(yl)
  plot(seq_along(versions), est, ylim = yl, xlim = c(0.5, 5.7), pch = pchs[versions], col = cols[versions],
       cex = 1.5, xaxt = "n", xlab = "", ylab = "", las = 1)
  arrows(seq_along(versions), lo, seq_along(versions), hi, angle = 90, code = 3, length = 0.03,
         col = cols[versions], lwd = 2)
  if(m %in% c("Bias", "R2_NSE")) abline(h = 0, lty = 3)
  if(grepl("AUC", m)) abline(h = 0.5, lty = 3)
  axis(1, at = seq_along(versions), labels = c("v8", "fit2", "v9\nprev", "v9\n+par", "v9\nfinal"),
       cex.axis = 0.75, padj = 0.5)
  text(seq_along(versions), est, sprintf("%.2f", est), pos = 4, cex = 0.7, col = cols[versions])
  title(mlab[m], cex.main = 0.95)
}
mtext("Pooled out-of-sample performance (338 bloom days) with 95% season-block bootstrap CI",
      side = 3, outer = TRUE, line = 1, font = 2)
dev.off()

# H3 improvement path of the search (LOSO)
path_labels <- c("v9 plan A (7 parameters)", "v9 plan A (7 parameters) + rh_thresh_M",
                 "v9 plan A (7 parameters) + rh_thresh_M + k_I",
                 "v9 plan A (7 parameters) + rh_thresh_M + k_I + [dh_growth]",
                 "v9 plan A (7 parameters) + rh_thresh_M + k_I + [dh_growth] + [prev_rain]",
                 sv$final$label)
path_short <- c("plan A\n(7 par)", "+ RH_th,M", "+ k_I", "+ DH growth\n(M3)", "+ prev-day rain\n(M1)", "gp = 4")
pm <- sapply(path_labels, function(l) sv$results[[l]]$metrics)
png(file.path(out_dir, "H3_search_path.png"), width = 3200, height = 1300, res = 250)
par(mfrow = c(1, 4), mar = c(6, 4.5, 3, 1))
for(m in c("RMSE", "R2_NSE", "Spearman_rho", "AUC_infection")){
  v <- pm[m, ]
  plot(seq_along(v), v, type = "o", pch = 16, lwd = 2, col = cols["v9_final"], xaxt = "n", xlab = "",
       ylab = mlab[m], las = 1, ylim = range(c(v, pooled[m, c("v8", "v8_fit2", "v9_prev")])))
  axis(1, at = seq_along(v), labels = path_short, las = 2, cex.axis = 0.75)
  abline(h = pooled[m, "v8"], col = cols["v8"], lty = 2, lwd = 1.5)
  abline(h = pooled[m, "v8_fit2"], col = cols["v8_fit2"], lty = 2, lwd = 1.5)
  abline(h = pooled[m, "v9_prev"], col = cols["v9_prev"], lty = 2, lwd = 1.5)
  rect(1.5, par("usr")[3], 3.5, par("usr")[4], col = adjustcolor("grey80", 0.3), border = NA)
  rect(3.5, par("usr")[3], 5.5, par("usr")[4], col = adjustcolor("#e7298a", 0.08), border = NA)
  title(mlab[m], cex.main = 1)
}
legend("bottomright", c("v8", "v8-fit2", "v9 (previous)", "stage 1: parameters", "stage 2: structure"),
       col = c(cols[c("v8", "v8_fit2", "v9_prev")], "grey70", adjustcolor("#e7298a", 0.3)),
       lty = c(2, 2, 2, NA, NA), pch = c(NA, NA, NA, 15, 15), pt.cex = 2, bty = "n", cex = 0.75)
dev.off()

# H4 prediction by BIR class
png(file.path(out_dir, "H4_prediction_by_BIR_class.png"), width = 3000, height = 1250, res = 250)
par(mfrow = c(1, 3), mar = c(4.5, 4.5, 3.2, 1))
for(v in show3){
  boxplot(pred[[v]] ~ pred$BIR, col = adjustcolor(cols[v], 0.35), border = cols[v], ylim = c(0, 1),
          xlab = "Pear BIR (gold standard)", ylab = "Predicted score (0-1)", las = 1, outline = FALSE)
  stripchart(pred[[v]] ~ pred$BIR, vertical = TRUE, method = "jitter", jitter = 0.2, add = TRUE,
             pch = 16, cex = 0.4, col = adjustcolor(cols[v], 0.6))
  segments(1:4 - 0.4, c(0, 1/3, 2/3, 1), 1:4 + 0.4, c(0, 1/3, 2/3, 1), col = col_inf, lwd = 3)
  title(sprintf("%s\nRMSE %.3f   R2 %.2f   r %.2f   rho %.2f", labs[v], pooled["RMSE", v],
                pooled["R2_NSE", v], pooled["Pearson_r", v], pooled["Spearman_rho", v]), cex.main = 0.95)
}
legend("topleft", "target (BIR - 25) / 75", col = col_inf, lwd = 3, bty = "n", cex = 0.9)
dev.off()

# H5 per-season metrics
png(file.path(out_dir, "H5_metrics_by_season.png"), width = 3200, height = 1300, res = 250)
par(mfrow = c(1, 4), mar = c(4.5, 4, 3.2, 1))
for(m in c("RMSE", "R2_NSE", "Pearson_r", "Spearman_rho")){
  w <- reshape(per_season[per_season$version %in% show3, c("season", "version", m)],
               idvar = "season", timevar = "version", direction = "wide")
  yy <- w$season; vals <- as.matrix(w[, paste0(m, ".", show3)])
  xl <- range(vals, na.rm = TRUE); if(m == "R2_NSE") xl[1] <- max(xl[1], -3)
  plot(NA, xlim = xl, ylim = range(yy), xlab = mlab[m], ylab = if(m == "RMSE") "Season" else "",
       las = 1, yaxt = "n")
  axis(2, at = yy, labels = yy, las = 1, cex.axis = 0.75)
  segments(apply(vals, 1, min, na.rm = TRUE), yy, apply(vals, 1, max, na.rm = TRUE), yy, col = "grey80")
  for(k in seq_along(show3)) points(pmax(vals[, k], xl[1]), yy, pch = pchs[show3[k]], col = cols[show3[k]], cex = 1.1)
  if(m != "RMSE") abline(v = 0, lty = 3)
  b1 <- if(m == "RMSE") sum(vals[, 3] < vals[, 2]) else sum(vals[, 3] > vals[, 2], na.rm = TRUE)
  b0 <- if(m == "RMSE") sum(vals[, 3] < vals[, 1]) else sum(vals[, 3] > vals[, 1], na.rm = TRUE)
  title(sprintf("%s\nfinal better: vs v8 %d/16, vs v9 prev %d/16", mlab[m], b0, b1), cex.main = 0.85)
}
legend("bottomleft", labs[show3], col = cols[show3], pch = pchs[show3], bty = "n", cex = 0.8)
dev.off()

# H6 ROC
roc_xy <- function(score, label){
  th <- c(Inf, sort(unique(score), decreasing = TRUE))
  data.frame(fpr = sapply(th, function(t) mean(score[!label] >= t)),
             tpr = sapply(th, function(t) mean(score[label] >= t)))
}
png(file.path(out_dir, "H6_ROC.png"), width = 3300, height = 1250, res = 250)
par(mfrow = c(1, 3), mar = c(4.5, 4.5, 3, 1), pty = "s")
tg <- list("Infection day (BIR = 100)" = list(l = pred$infection, m = "AUC_infection"),
           "BIR >= 75" = list(l = pred$BIR >= 75, m = "AUC_score_ge_2_3"),
           "BIR >= 50" = list(l = pred$BIR >= 50, m = "AUC_score_ge_1_3"))
for(tn in names(tg)){
  plot(0:1, 0:1, type = "n", xlab = "False positive rate", ylab = "True positive rate", las = 1)
  abline(0, 1, lty = 3, col = "grey60")
  for(v in show3){ r <- roc_xy(pred[[v]], tg[[tn]]$l); lines(r$fpr, r$tpr, col = cols[v], lwd = 2.2, type = "s") }
  legend("bottomright", sprintf("%s  AUC %.3f", labs[show3], pooled[tg[[tn]]$m, show3]),
         col = cols[show3], lwd = 2.2, bty = "n", cex = 0.8)
  title(sprintf("ROC: %s (n+ = %d)", tn, sum(tg[[tn]]$l)), cex.main = 1)
}
dev.off()

# H7 all configurations tried in the search
st <- read.csv(file.path(pear_dir, "search_v9", "search_summary.csv"))
stage <- ifelse(grepl("\\| gp=", st$config), "stage 3: growth period",
         ifelse(grepl("\\[", st$config), "stage 2: structure",
         ifelse(st$config == "v9 plan A (7 parameters)", "plan A", "stage 1: parameters")))
scol <- c("plan A" = "black", "stage 1: parameters" = "#8da0cb", "stage 2: structure" = "#e7298a",
          "stage 3: growth period" = "#66a61e")
png(file.path(out_dir, "H7_search_configurations.png"), width = 2800, height = 1300, res = 250)
par(mfrow = c(1, 2), mar = c(4.5, 4.5, 3, 1))
plot(st$n_par + runif(nrow(st), -0.15, 0.15), st$RMSE, pch = 16, col = scol[stage], cex = 1.2,
     xlab = "Number of calibrated parameters", ylab = "LOSO RMSE", las = 1)
abline(h = pooled["RMSE", c("v8_fit2", "v9_prev")], lty = 2, col = cols[c("v8_fit2", "v9_prev")])
points(length(sv$final$cfg$free), sv$final$metrics["RMSE"], pch = 1, cex = 2.5, lwd = 2)
legend("topright", c(names(scol), "v8-fit2", "v9 (previous)", "selected"),
       col = c(scol, cols[c("v8_fit2", "v9_prev")], "black"), pch = c(16, 16, 16, 16, NA, NA, 1),
       lty = c(NA, NA, NA, NA, 2, 2, NA), bty = "n", cex = 0.75)
title("All configurations evaluated (LOSO)", cex.main = 1)
plot(st$RMSE, st$Spearman_rho, pch = 16, col = scol[stage], cex = 1.2, xlab = "LOSO RMSE",
     ylab = "LOSO Spearman rho", las = 1)
points(sv$final$metrics["RMSE"], sv$final$metrics["Spearman_rho"], pch = 1, cex = 2.5, lwd = 2)
title("Accuracy vs rank agreement", cex.main = 1)
dev.off()

# H8 final v9 components vs pear conditions
png(file.path(out_dir, "H8_final_components.png"), width = 3200, height = 1250, res = 250)
par(mfrow = c(1, 4), mar = c(4.5, 4.5, 3, 1))
plot(pred$EIP, pred$N_final, pch = 16, cex = 0.7, col = ifelse(pred$H, col_inf, "grey50"),
     xlab = "Pear EIP", ylab = "v9 final N (degree-hour growth)", las = 1)
abline(v = 100, lty = 3)
title(sprintf("N vs pear EIP (r = %.2f)", cor(pred$EIP, pred$N_final)), cex.main = 1)
legend("bottomright", c("pear H = TRUE", "pear H = FALSE"), col = c(col_inf, "grey50"), pch = 16, bty = "n", cex = 0.8)
grpW <- ifelse(pred$W & pred$rain < 0.25 & pred$rh < 80, "W+ (prev-day\nrain only)",
               ifelse(pred$W, "W+ (rain/RH)", "W-"))
grpW <- factor(grpW, levels = c("W-", "W+ (rain/RH)", "W+ (prev-day\nrain only)"))
W_prev <- c9$pred$I_v9 / pmax(c9$pred$fT_v9, 1e-9)
boxplot(list("v9 prev\nW-" = W_prev[grpW == "W-"], "final\nW-" = pred$W_final[grpW == "W-"],
             "v9 prev\nW+ rain/RH" = W_prev[grpW == "W+ (rain/RH)"], "final\nW+ rain/RH" = pred$W_final[grpW == "W+ (rain/RH)"],
             "v9 prev\nW+ prev-day" = W_prev[grpW == "W+ (prev-day\nrain only)"],
             "final\nW+ prev-day" = pred$W_final[grpW == "W+ (prev-day\nrain only)"]),
        col = rep(adjustcolor(cols[c("v9_prev", "v9_final")], 0.4), 3), las = 2, cex.axis = 0.7,
        ylab = "Wetness W", ylim = c(0, 1))
title("Wetness by pear W (incl. prev-day rain rule)", cex.main = 0.95)
rp <- pred$rain_prev > 0
plot(pred$rain_prev[rp], pred$W_final[rp], pch = 16, cex = 0.7, col = ifelse(pred$W[rp], col_inf, "grey50"),
     xlab = "Previous-day rain (mm, log scale; days with rain only)", ylab = "v9 final W", las = 1, log = "x",
     xlim = c(0.1, max(pred$rain_prev)), ylim = c(0, 1))
abline(v = sv$final$all$par["rain_prev_thresh"], lty = 2)
title(sprintf("Prev-day rain wetting (threshold %.2f mm)", sv$final$all$par["rain_prev_thresh"]), cex.main = 0.95)
boxplot(pred$I_final ~ pred$BIR, col = adjustcolor(cols["v9_final"], 0.35), border = cols["v9_final"],
        xlab = "Pear BIR", ylab = "v9 final I", las = 1, ylim = c(0, 1))
title("Infection potential I by BIR class", cex.main = 1)
dev.off()
