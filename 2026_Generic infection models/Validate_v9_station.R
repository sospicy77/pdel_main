#############################################
# Independent validation of v8 / v8-fit2 / v9 (previous) / v9 (final)
# against the pear fire blight model at another station.
# Parameters are those calibrated at CF-99 (hourly_99); nothing is refitted.
# Models are run with the model code files (run_model).
#
# Usage (from the "2026_Generic infection models" folder):
#   Rscript Validate_v9_station.R 105
# after
#   python3 pear_fire_blight_fixed.py obs/hourly_105 2010 ... 2025 Output/pear_105
#   python3 make_hourly_features.py obs/hourly_105 2010 ... 2025 Output/pear_105/hourly_features.csv
#############################################

args <- commandArgs(trailingOnly = TRUE)
station  <- if(length(args)) args[1] else "105"
pear_dir <- file.path("Output", paste0("pear_", station))
out_dir  <- file.path(pear_dir, "validation_v9")
cal_dir  <- file.path("Output", "pear_99")
seasons  <- 2010:2025
n_boot   <- 2000
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

load_model <- function(file){
  env <- new.env()
  for(e in parse(file, encoding = "UTF-8")){
    if(is.call(e) && identical(e[[1]], as.name("<-")) &&
       (as.character(e[[2]]) %in% c("global_params", "disease_params") ||
        (is.call(e[[3]]) && identical(e[[3]][[1]], as.name("function"))))){
      eval(e, envir = env)
    }
  }
  env
}
m8 <- load_model("bacterialmodel_v8.R")
m9 <- load_model("bacterialmodel_v9.R")

# -------------------------------
# Parameter sets (CF-99 calibration, full precision)
# -------------------------------
p_v8 <- m8$disease_params$fire_blight

fit2 <- read.csv(file.path(cal_dir, "calibration", "calibration_all_fits.csv"))
fit2 <- fit2[fit2$design == "ALL", ]
p_fit2 <- p_v8
for(n in c("K_d", "N0", "mu_max", "T_opt", "rh_thresh_I", "rain_thresh")) p_fit2[[n]] <- fit2[[n]]

prev <- read.csv(file.path(cal_dir, "calibration_v9", "v9_all_fits.csv"))
prev <- prev[prev$variant == "v9" & prev$design == "ALL", ]
p_prev <- modifyList(m9$disease_params$fire_blight,
  list(growth_period = NULL, rh_thresh_M = 50, k_M = 0.3, k_I = 1,
       use_dh_growth = FALSE, use_prev_rain_wetness = FALSE))
p_prev$growth_period <- NULL
for(n in c("N0", "mu_max", "T_opt", "rh_thresh_I", "rain_thresh", "rain_coeff", "insect_coeff"))
  p_prev[[n]] <- prev[[n]]

p_final <- m9$disease_params$fire_blight

runs <- list(v8 = list(m = m8, p = p_v8), v8_fit2 = list(m = m8, p = p_fit2),
             v9_prev = list(m = m9, p = p_prev), v9_final = list(m = m9, p = p_final))
versions <- names(runs)
labs <- c(v8 = "v8 (current)", v8_fit2 = "v8-fit2", v9_prev = "v9 (previous)", v9_final = "v9 (final)")
cols <- c(v8 = "#54278f", v8_fit2 = "#1b9e77", v9_prev = "#fc8d62", v9_final = "#e7298a")
pchs <- c(v8 = 16, v8_fit2 = 17, v9_prev = 15, v9_final = 8)

season_input <- function(dir, y){
  x <- read.csv(file.path(dir, sprintf("pear_daily_%d.csv", y))); x$tm <- as.Date(x$tm)
  f <- read.csv(file.path(dir, "hourly_features.csv")); f$date <- as.Date(f$date)
  f <- f[f$season == y, ]
  list(x = x, wx = data.frame(date = x$tm, temp = x$ta, rh = x$hm, rain = x$rn,
                              vulnerable_period = as.numeric(x$B == "True"),
                              degree_hours = f$dh183[match(x$tm, f$date)]))
}
predict_station <- function(dir){
  do.call(rbind, lapply(seasons, function(y){
    s <- season_input(dir, y); b <- s$x$B == "True"
    out <- data.frame(season = y, date = s$x$tm[b], temp = s$x$ta[b], rh = s$x$hm[b],
                      rain = s$x$rn[b], BIR = s$x$BIR[b], infection = s$x$infection[b] == "True",
                      H = s$x$H[b] == "True", W = s$x$W[b] == "True", T = s$x$T[b] == "True")
    for(v in versions){
      g <- runs[[v]]$m$global_params$growth_period
      out[[v]] <- runs[[v]]$m$run_model(s$wx, runs[[v]]$p, g)$Risk[b]
    }
    out
  }))
}

# -------------------------------
# Sanity check: same code reproduces the CF-99 in-sample predictions
# -------------------------------
cf <- predict_station(cal_dir)
c8 <- new.env(); load(file.path(cal_dir, "calibration", "calibration_results.RData"), envir = c8)
c9 <- new.env(); load(file.path(cal_dir, "calibration_v9", "v9_results.RData"), envir = c9)
fin <- read.csv(file.path(cal_dir, "final_v9", "final_predictions_bloom_days.csv"))
sv <- new.env(); load(file.path(cal_dir, "search_v9", "search_results.RData"), envir = sv)
src <- new.env(); txt <- readLines("Search_v9_hourly99.R")
eval(parse(text = txt[grep("^temp_response <- function", txt):(grep("^# 3. CALIBRATION", txt) - 2)]), envir = src)
src$frame <- sv$frame
fin_all <- src$model_risk(modifyList(sv$defaults, as.list(sv$final$all$par)), sv$final$cfg)$Risk[sv$bl]
check <- c(v8 = max(abs(cf$v8 - c8$pred$v8)),
           v8_fit2 = max(abs(cf$v8_fit2 - c8$pred$v8_fit2_all)),
           v9_prev = max(abs(cf$v9_prev - c9$pred$v9_all)),
           v9_final = max(abs(cf$v9_final - fin_all)))

# -------------------------------
# Validation station
# -------------------------------
pred <- predict_station(pear_dir)
pred$score <- (pred$BIR - 25) / 75
cf$score <- (cf$BIR - 25) / 75
write.csv(pred, file.path(out_dir, "validation_predictions_bloom_days.csv"), row.names = FALSE)

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
pooled_cf99_insample <- sapply(versions, function(v) metrics(cf[[v]], cf$score, cf$infection))
cf_loso <- read.csv(file.path(cal_dir, "final_v9", "final_metrics_pooled.csv"))
rownames(cf_loso) <- cf_loso$metric

set.seed(105)
rows <- split(seq_len(nrow(pred)), pred$season)
boot <- replicate(n_boot, {
  i <- unlist(rows[sample(names(rows), replace = TRUE)], use.names = FALSE)
  sapply(versions, function(v) metrics(pred[[v]][i], pred$score[i], pred$infection[i]))
}, simplify = "array")
ci_lo <- apply(boot, c(1, 2), quantile, 0.025, na.rm = TRUE)
ci_hi <- apply(boot, c(1, 2), quantile, 0.975, na.rm = TRUE)
pairs <- list(final_vs_v8 = c("v9_final", "v8"), final_vs_v8fit2 = c("v9_final", "v8_fit2"),
              final_vs_v9prev = c("v9_final", "v9_prev"))
diff_tab <- do.call(rbind, lapply(names(pairs), function(nm){
  a <- pairs[[nm]][1]; b <- pairs[[nm]][2]; d <- boot[, a, ] - boot[, b, ]
  data.frame(comparison = nm, metric = rownames(pooled), change = pooled[, a] - pooled[, b],
             CI_low = apply(d, 1, quantile, 0.025, na.rm = TRUE),
             CI_high = apply(d, 1, quantile, 0.975, na.rm = TRUE), row.names = NULL)
}))
per_season <- do.call(rbind, lapply(split(pred, pred$season), function(x)
  do.call(rbind, lapply(versions, function(v)
    data.frame(season = x$season[1], version = v, t(metrics(x[[v]], x$score, x$infection)))))))
write.csv(data.frame(metric = rownames(pooled), round(pooled, 4)),
          file.path(out_dir, "validation_metrics_pooled.csv"), row.names = FALSE)
write.csv(diff_tab, file.path(out_dir, "validation_metric_changes_CI.csv"), row.names = FALSE)
write.csv(per_season, file.path(out_dir, "validation_metrics_by_season.csv"), row.names = FALSE)

sink(file.path(out_dir, "validation_report.txt"), split = TRUE)
cat("Validation station:", station, "| bloom days:", nrow(pred), "| infection days:", sum(pred$infection),
    "in", length(unique(pred$season[pred$infection])), "seasons\n")
print(table(BIR = pred$BIR))
cat("\nCode check on CF-99 (max |run_model - calibration in-sample prediction|):\n")
print(signif(check, 3))
cat("\nPooled metrics at station", station, "(parameters from CF-99, not refitted)\n")
print(round(pooled, 3))
cat("\nReference: CF-99 LOSO (out-of-sample) for v8 / v8-fit2 / v9 prev / v9 final\n")
print(round(as.matrix(cf_loso[rownames(pooled), c("v8", "v8_fit2", "v9_prev", "v9_final")]), 3))
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
st_name <- paste0("station ", station, " (hourly_", station, ")")

# V1 (b)-style daily risk by season
png(file.path(out_dir, "V1_daily_risk_by_season.png"), width = 3200, height = 3100, res = 250)
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
mtext(paste0("(b) Daily infection risk - independent validation at ", st_name,
             ", 2010-2025 (parameters from CF-99)"), side = 3, outer = TRUE, line = 2.2, font = 2, cex = 1.05)
par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
plot(0, 0, type = "n", bty = "n", xaxt = "n", yaxt = "n")
legend("bottom", c("Pear BIR (gold standard)", "Pear infection day", labs[show3], "title: RMSE v8 | v9 prev | v9 final"),
       col = c(col_gold, col_inf, cols[show3], NA), lwd = c(6, NA, 1.5, 1.5, 2.2, NA),
       pch = c(NA, 25, pchs[show3], NA), pt.bg = col_inf, horiz = TRUE, bty = "n", cex = 0.85)
dev.off()

# V2 pooled metrics with CI; open symbols = CF-99 LOSO
mets <- c("RMSE", "MAE", "Bias", "R2_NSE", "Pearson_r", "Spearman_rho", "Kendall_tau",
          "AUC_infection", "AUC_score_ge_1_3", "AUC_score_ge_2_3")
png(file.path(out_dir, "V2_pooled_metrics_CI.png"), width = 3400, height = 1550, res = 250)
par(mfrow = c(2, 5), mar = c(3.4, 2.5, 2.8, 1), oma = c(3, 2, 3, 0))
xx <- seq_along(versions)
for(m in mets){
  est <- pooled[m, versions]; lo <- ci_lo[m, versions]; hi <- ci_hi[m, versions]
  ref <- unlist(cf_loso[m, versions])
  yl <- range(c(lo, hi, ref, if(m %in% c("Bias", "R2_NSE")) 0), na.rm = TRUE); yl <- yl + c(-1, 1) * 0.08 * diff(yl)
  plot(xx, est, ylim = yl, xlim = c(0.5, 4.7), pch = pchs[versions], col = cols[versions], cex = 1.5,
       xaxt = "n", xlab = "", ylab = "", las = 1)
  arrows(xx, lo, xx, hi, angle = 90, code = 3, length = 0.03, col = cols[versions], lwd = 2)
  points(xx + 0.25, ref, pch = 1, col = cols[versions], cex = 1.2, lwd = 1.5)
  if(m %in% c("Bias", "R2_NSE")) abline(h = 0, lty = 3)
  if(grepl("AUC", m)) abline(h = 0.5, lty = 3)
  axis(1, at = xx, labels = c("v8", "fit2", "v9\nprev", "v9\nfinal"), cex.axis = 0.75, padj = 0.5)
  text(xx, est, sprintf("%.2f", est), pos = 2, cex = 0.7, col = cols[versions])
  title(mlab[m], cex.main = 0.95)
}
mtext(paste0("Independent validation at ", st_name, ": pooled performance (", nrow(pred),
             " bloom days) with 95% season-block bootstrap CI"), side = 3, outer = TRUE, line = 1, font = 2)
mtext("filled symbols + bars = this station (parameters from CF-99); open circles = CF-99 leave-one-season-out",
      side = 1, outer = TRUE, line = 1.5, cex = 0.85)
dev.off()

# V3 prediction by BIR class
png(file.path(out_dir, "V3_prediction_by_BIR_class.png"), width = 3000, height = 1250, res = 250)
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

# V4 per-season metrics
png(file.path(out_dir, "V4_metrics_by_season.png"), width = 3200, height = 1300, res = 250)
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

# V5 ROC
roc_xy <- function(score, label){
  th <- c(Inf, sort(unique(score), decreasing = TRUE))
  data.frame(fpr = sapply(th, function(t) mean(score[!label] >= t)),
             tpr = sapply(th, function(t) mean(score[label] >= t)))
}
png(file.path(out_dir, "V5_ROC.png"), width = 3300, height = 1250, res = 250)
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

# V6 transferability of the final v9: CF-99 LOSO vs this station
tm <- c("RMSE", "R2_NSE", "Pearson_r", "Spearman_rho", "AUC_infection", "AUC_score_ge_1_3",
        "AUC_score_ge_2_3", "Class_acc")
png(file.path(out_dir, "V6_transfer_final_v9.png"), width = 3000, height = 1300, res = 250)
par(mar = c(6, 4.5, 3, 1))
mat <- rbind("CF-99 (LOSO)" = cf_loso[tm, "v9_final"], "this station" = pooled[tm, "v9_final"])
mat0 <- rbind("CF-99 (LOSO)" = cf_loso[tm, "v9_prev"], "this station" = pooled[tm, "v9_prev"])
bp <- barplot(mat, beside = TRUE, col = c("#fbb4d4", cols["v9_final"]), ylim = c(0, 1.35), yaxt = "n",
              names.arg = gsub("_", " ", tm), las = 2, cex.names = 0.8, ylab = "Value")
points(bp[1, ], mat0[1, ], pch = 1, cex = 1.3, lwd = 1.5, col = cols["v9_prev"])
points(bp[2, ], mat0[2, ], pch = 15, cex = 1.1, col = cols["v9_prev"])
axis(2, at = seq(0, 1, 0.2), las = 1)
text(bp, mat, sprintf("%.2f", mat), pos = 3, cex = 0.7)
legend("top", ncol = 2, c("v9 final: CF-99 (LOSO)", paste0("v9 final: ", st_name),
                     "v9 previous: CF-99 (LOSO)", paste0("v9 previous: ", st_name)),
       fill = c("#fbb4d4", cols["v9_final"], NA, NA), border = c("black", "black", NA, NA),
       pch = c(NA, NA, 1, 15), col = c(NA, NA, cols["v9_prev"], cols["v9_prev"]), bty = "n", cex = 0.8)
title(paste0("Transferability of v9: calibration station CF-99 vs ", st_name), cex.main = 1)
dev.off()
