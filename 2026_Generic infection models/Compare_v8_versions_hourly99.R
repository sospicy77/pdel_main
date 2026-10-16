#############################################
# Independent evaluation of two bacterialmodel_v8 versions (fire_blight)
#   v8 (current parameters)  vs  v8-fit (fitted to the pear model at ID1001, 2019)
# against the pear fire blight model (gold standard)
# Weather: obs/hourly_99 (CF-99), seasons 2010-2025, pear bloom periods
#############################################

# Run from the "2026_Generic infection models" folder after:
#   python3 pear_fire_blight_fixed.py obs/hourly_99 2010 ... 2025 Output/pear_99

model_file <- "bacterialmodel_v8.R"
fit_file   <- file.path("Output", "pear_ID1001_2019", "v8_fit_parameters.csv")
pear_dir   <- file.path("Output", "pear_99")
out_dir    <- pear_dir
disease    <- "fire_blight"
seasons    <- 2010:2025
n_boot     <- 2000

# -------------------------------
# 1. LOAD v8 AND THE TWO PARAMETER SETS
# -------------------------------

model_exprs <- parse(model_file, encoding = "UTF-8")
for(e in model_exprs){
  if(is.call(e) && identical(e[[1]], as.name("<-")) &&
     (as.character(e[[2]]) %in% c("global_params", "disease_params") ||
      (is.call(e[[3]]) && identical(e[[3]][[1]], as.name("function"))))){
    eval(e, envir = globalenv())
  }
}
gp <- global_params$growth_period
p_old <- disease_params[[disease]]
fit <- read.csv(fit_file)
p_new <- p_old
p_new[fit$parameter] <- as.list(fit$v8_fit)

# -------------------------------
# 2. RUN BOTH VERSIONS FOR EACH SEASON (bloom days kept)
# -------------------------------

all_days <- do.call(rbind, lapply(seasons, function(y){
  f <- file.path(pear_dir, sprintf("pear_daily_%d.csv", y))
  if(!file.exists(f)) return(NULL)
  x <- read.csv(f)
  x$tm <- as.Date(x$tm)
  wx <- data.frame(date = x$tm, temp = x$ta, rh = x$hm, rain = x$rn)
  r_old <- run_model(wx, p_old, gp)$Risk
  r_new <- run_model(wx, p_new, gp)$Risk
  b <- x$B == "True"
  data.frame(season = y, date = x$tm[b], ta = x$ta[b], hm = x$hm[b], rn = x$rn[b],
             H = x$H[b] == "True", W = x$W[b] == "True", T = x$T[b] == "True",
             BIR = x$BIR[b], level = x$level[b], infection = x$infection[b] == "True",
             Risk_v8 = r_old[b], Risk_v8fit = r_new[b])
}))
write.csv(all_days, file.path(out_dir, "v8_versions_daily_bloom.csv"), row.names = FALSE)

# -------------------------------
# 3. METRICS
# -------------------------------

auc <- function(score, label){
  pos <- score[label]; neg <- score[!label]
  if(length(pos) == 0 || length(neg) == 0) return(NA)
  mean(outer(pos, neg, ">") + 0.5 * outer(pos, neg, "=="))
}
safe_cor <- function(a, b, m = "pearson")
  if(length(a) < 3 || sd(a) == 0 || sd(b) == 0) NA else cor(a, b, method = m)

metrics <- function(pred, obs, infection){
  nearest <- c(0.25, 0.5, 0.75, 1)[apply(abs(outer(pred, c(0.25, 0.5, 0.75, 1), "-")), 1, which.min)]
  c(RMSE = sqrt(mean((pred - obs)^2)),
    MAE = mean(abs(pred - obs)),
    Bias = mean(pred - obs),
    R2 = 1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2),
    Pearson = safe_cor(pred, obs),
    Spearman = safe_cor(pred, obs, "spearman"),
    AUC_BIR50 = auc(pred, obs >= 0.5),
    AUC_BIR75 = auc(pred, obs >= 0.75),
    AUC_inf = auc(pred, infection),
    Class_acc = mean(nearest == obs))
}

obs <- all_days$BIR / 100
m_old <- metrics(all_days$Risk_v8, obs, all_days$infection)
m_new <- metrics(all_days$Risk_v8fit, obs, all_days$infection)
pooled <- data.frame(metric = names(m_old), v8 = m_old, v8_fit = m_new,
                     change = m_new - m_old, row.names = NULL)

# Season-block bootstrap for the change (v8-fit minus v8)
set.seed(99)
boot <- t(replicate(n_boot, {
  s <- sample(unique(all_days$season), replace = TRUE)
  x <- do.call(rbind, lapply(s, function(k) all_days[all_days$season == k, ]))
  metrics(x$Risk_v8fit, x$BIR / 100, x$infection) -
    metrics(x$Risk_v8, x$BIR / 100, x$infection)
}))
ci <- apply(boot, 2, quantile, c(0.025, 0.975), na.rm = TRUE)
pooled$change_CI_low  <- ci[1, ]
pooled$change_CI_high <- ci[2, ]
better_if_lower <- pooled$metric %in% c("RMSE", "MAE")
pooled$better <- ifelse(pooled$metric == "Bias",
                        ifelse(abs(pooled$v8_fit) < abs(pooled$v8), "v8-fit", "v8"),
                        ifelse(better_if_lower == (pooled$change < 0), "v8-fit", "v8"))
pooled$significant <- ifelse(pooled$metric == "Bias", NA,
                             pooled$change_CI_low > 0 | pooled$change_CI_high < 0)
write.csv(pooled, file.path(out_dir, "v8_versions_metrics_pooled.csv"), row.names = FALSE)

per_season <- do.call(rbind, lapply(split(all_days, all_days$season), function(x){
  o <- x$BIR / 100
  a <- metrics(x$Risk_v8, o, x$infection); b <- metrics(x$Risk_v8fit, o, x$infection)
  data.frame(season = x$season[1], days = nrow(x), infection_days = sum(x$infection),
             RMSE_v8 = a["RMSE"], RMSE_v8fit = b["RMSE"],
             Spearman_v8 = a["Spearman"], Spearman_v8fit = b["Spearman"],
             AUC_BIR50_v8 = a["AUC_BIR50"], AUC_BIR50_v8fit = b["AUC_BIR50"],
             row.names = NULL)
}))
write.csv(per_season, file.path(out_dir, "v8_versions_metrics_by_season.csv"), row.names = FALSE)

sink(file.path(out_dir, "v8_versions_report.txt"), split = TRUE)
cat("Station CF-99 (obs/hourly_99), seasons", min(all_days$season), "-", max(all_days$season),
    "| bloom days:", nrow(all_days), "| infection days:", sum(all_days$infection),
    "in", length(unique(all_days$season[all_days$infection])), "seasons\n")
cat("Gold-standard BIR distribution:\n"); print(table(BIR = all_days$BIR))
cat("\nParameters (v8-fit fitted at ID1001 2019; not refitted here)\n")
print(fit[, c("parameter", "v8_current", "v8_fit")], row.names = FALSE)
cat("\nPooled metrics (Risk vs BIR/100), change = v8-fit - v8,",
    "95% CI from", n_boot, "season-block bootstrap resamples\n")
print(cbind(pooled[, 1, drop = FALSE], round(pooled[, 2:6], 3), pooled[, 7:8]), row.names = FALSE)
cat("\nMean Risk by BIR class\n")
print(round(rbind(v8 = tapply(all_days$Risk_v8, all_days$BIR, mean),
                  v8_fit = tapply(all_days$Risk_v8fit, all_days$BIR, mean)), 3))
cat("\nPer-season metrics\n")
print(cbind(per_season[, 1:3], round(per_season[, -(1:3)], 3)), row.names = FALSE)
cat("\nSeasons where v8-fit has lower RMSE:", sum(per_season$RMSE_v8fit < per_season$RMSE_v8),
    "of", nrow(per_season), "\n")
cat("Seasons where v8-fit has higher Spearman:",
    sum(per_season$Spearman_v8fit > per_season$Spearman_v8, na.rm = TRUE), "of",
    sum(!is.na(per_season$Spearman_v8fit) & !is.na(per_season$Spearman_v8)), "\n")
sink()

# -------------------------------
# 4. PLOTS: (b)-style daily infection risk, one panel per season
# -------------------------------

col_bir <- "#fdae6b"; col_old <- "#54278f"; col_new <- "#1b9e77"

png(file.path(out_dir, "v8_versions_bloom_by_season.png"), width = 3200, height = 3000, res = 250)
par(mfrow = c(4, 4), mar = c(2.6, 3.2, 2.2, 3.2), oma = c(3, 1.5, 4, 1.5))
for(y in sort(unique(all_days$season))){
  x <- all_days[all_days$season == y, ]
  plot(x$date, x$BIR, type = "h", lwd = 5, lend = 1, col = col_bir,
       ylim = c(0, 115), xaxt = "n", xlab = "", ylab = "", las = 1, cex.axis = 0.8)
  points(x$date[x$infection], rep(108, sum(x$infection)), pch = 25,
         bg = "#a63603", col = "#a63603", cex = 0.9)
  lines(x$date, x$Risk_v8 * 100, col = col_old, lwd = 1.8, type = "o", pch = 16, cex = 0.45)
  lines(x$date, x$Risk_v8fit * 100, col = col_new, lwd = 1.8, type = "o", pch = 17, cex = 0.45)
  axis(4, at = seq(0, 100, 50), labels = c("0", "0.5", "1"), las = 1, cex.axis = 0.8)
  axis.Date(1, at = seq(min(x$date), max(x$date), by = 7), format = "%m-%d", cex.axis = 0.8)
  r <- per_season[per_season$season == y, ]
  title(sprintf("%d   RMSE %.2f / %.2f", y, r$RMSE_v8, r$RMSE_v8fit),
        adj = 0, cex.main = 0.95)
}
mtext("Pear BIR (0-100)", side = 2, outer = TRUE, line = 0)
mtext("v8 Risk (0-1)", side = 4, outer = TRUE, line = 0)
mtext("(b) Daily infection risk in the pear bloom period - CF-99 (hourly_99), 2010-2025",
      side = 3, outer = TRUE, line = 2.2, font = 2, cex = 1.1)
par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
plot(0, 0, type = "n", bty = "n", xaxt = "n", yaxt = "n")
legend("bottom", c("Pear BIR (gold standard)", "Pear infection day", "v8 (current)", "v8-fit",
                   "panel title: RMSE v8 / v8-fit"),
       col = c(col_bir, "#a63603", col_old, col_new, NA), lwd = c(6, NA, 2, 2, NA),
       pch = c(NA, 25, 16, 17, NA), pt.bg = "#a63603", horiz = TRUE, bty = "n", cex = 0.9)
dev.off()

png(file.path(out_dir, "v8_versions_summary.png"), width = 2800, height = 1300, res = 250)
par(mfrow = c(1, 3), mar = c(4.5, 4.5, 3, 1))
yy <- per_season$season
plot(per_season$RMSE_v8, yy, pch = 16, col = col_old, xlim = c(0, max(per_season[, 4:5]) * 1.05),
     xlab = "RMSE (lower is better)", ylab = "Season", las = 1)
points(per_season$RMSE_v8fit, yy, pch = 17, col = col_new)
segments(per_season$RMSE_v8, yy, per_season$RMSE_v8fit, yy, col = "grey70")
title("(a) RMSE by season", adj = 0, cex.main = 1)
plot(per_season$Spearman_v8, yy, pch = 16, col = col_old, xlim = c(-0.6, 1),
     xlab = "Spearman rho (higher is better)", ylab = "", las = 1)
points(per_season$Spearman_v8fit, yy, pch = 17, col = col_new)
segments(per_season$Spearman_v8, yy, per_season$Spearman_v8fit, yy, col = "grey70")
abline(v = 0, lty = 3)
title("(b) Rank agreement by season", adj = 0, cex.main = 1)
legend("bottomleft", c("v8 (current)", "v8-fit"), col = c(col_old, col_new),
       pch = c(16, 17), bty = "n")
bx <- rbind(v8 = tapply(all_days$Risk_v8, all_days$BIR, mean),
            v8fit = tapply(all_days$Risk_v8fit, all_days$BIR, mean))
mids <- barplot(bx, beside = TRUE, col = c(col_old, col_new), ylim = c(0, 1),
                names.arg = colnames(bx), xlab = "Pear BIR class", ylab = "Mean v8 Risk", las = 1)
segments(mids[1, ] - 0.5, as.numeric(colnames(bx)) / 100,
         mids[2, ] + 0.5, as.numeric(colnames(bx)) / 100, col = "#a63603", lwd = 3)
title("(c) Mean Risk by BIR class (dash = BIR/100)", adj = 0, cex.main = 1)
dev.off()
