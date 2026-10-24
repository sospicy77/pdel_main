#############################################
# Fit bacterialmodel_v8 (fire_blight) to the pear fire blight model
# (Maryblyt-type, gold standard) at ID1001, 2019, and compare
#   v8 (current parameters)  vs  v8-fit (re-parameterized)
#############################################

# Run from the "2026_Generic infection models" folder after:
#   python3 pear_fire_blight_fixed.py obs/ID1001.csv 2019 Output/pear_ID1001_2019
#   python3 pear_fire_blight_fixed.py obs/ID1001.csv 2019 Output/pear_ID1001_2019 \
#           --force-bloom 2019-03-01 2019-06-30
#
# Data use
#   Calibration : forced-bloom what-if, 03-01..06-30, excluding the actual bloom days
#   Test        : actual pear bloom period (B1 <= day < PF), not used in fitting
#   Target      : pear BIR / 100 (0.25, 0.50, 0.75, 1.00)

model_file   <- "bacterialmodel_v8.R"
out_dir      <- file.path("Output", "pear_ID1001_2019")
pear_file    <- file.path(out_dir, "pear_daily_2019.csv")
whatif_file  <- file.path(out_dir, "pear_daily_2019_forcebloom.csv")
disease      <- "fire_blight"
n_starts     <- 40

# -------------------------------
# 1. LOAD v8 FUNCTIONS / PARAMETERS
# -------------------------------

model_exprs <- parse(model_file, encoding = "UTF-8")
for(e in model_exprs){
  if(is.call(e) && identical(e[[1]], as.name("<-")) &&
     (as.character(e[[2]]) %in% c("global_params", "disease_params") ||
      (is.call(e[[3]]) && identical(e[[3]][[1]], as.name("function"))))){
    eval(e, envir = globalenv())
  }
}
p_old <- disease_params[[disease]]
gp    <- global_params$growth_period

# -------------------------------
# 2. DATA
# -------------------------------

read_pear <- function(f){
  x <- read.csv(f)
  x$tm <- as.Date(x$tm)
  x$B <- x$B == "True"
  x$infection <- x$infection == "True"
  x
}
pear   <- read_pear(pear_file)
whatif <- read_pear(whatif_file)

# Same daily weather for both models (aggregated from hourly ID1001)
wx <- data.frame(date = pear$tm, temp = pear$ta, rh = pear$hm, rain = pear$rn)

test_idx  <- which(pear$B)                              # actual bloom
calib_idx <- which(whatif$B & !(whatif$tm %in% pear$tm[test_idx]))
obs_test  <- pear$BIR[test_idx] / 100
obs_calib <- whatif$BIR[calib_idx] / 100

# -------------------------------
# 3. PARAMETERS TO FIT (bounded)
# -------------------------------

fit_spec <- data.frame(
  name  = c("K_d", "N0",  "mu_max", "T_opt",            "rh_thresh_I", "rain_thresh"),
  lower = c(0.01,  0.01,  0.00,     p_old$T_min + 0.5,  40,            0.0),
  upper = c(1.00,  0.99,  2.00,     p_old$T_max - 0.5,  100,           10.0)
)

# unbounded z <-> bounded parameter
to_par <- function(z) fit_spec$lower + (fit_spec$upper - fit_spec$lower) * plogis(z)
to_z   <- function(v) qlogis(pmin(pmax((v - fit_spec$lower) /
                                       (fit_spec$upper - fit_spec$lower), 1e-6), 1 - 1e-6))

with_values <- function(p, v){
  p[fit_spec$name] <- as.list(v)
  p
}

predict_risk <- function(p) run_model(wx, p, gp)$Risk

sse_calib <- function(z){
  r <- predict_risk(with_values(p_old, to_par(z)))[calib_idx]
  sum((r - obs_calib)^2)
}

# -------------------------------
# 4. MULTI-START OPTIMIZATION
# -------------------------------

set.seed(2019)
start_old <- to_z(unlist(p_old[fit_spec$name]))
starts <- c(list(start_old),
            replicate(n_starts - 1, rnorm(nrow(fit_spec), 0, 1.5), simplify = FALSE))

fits <- lapply(starts, function(z0){
  f1 <- optim(z0, sse_calib, method = "Nelder-Mead", control = list(maxit = 3000))
  f2 <- optim(f1$par, sse_calib, method = "Nelder-Mead", control = list(maxit = 3000))
  f2
})
best  <- fits[[which.min(sapply(fits, `[[`, "value"))]]
v_new <- to_par(best$par)
p_new <- with_values(p_old, v_new)

param_table <- data.frame(
  parameter = fit_spec$name,
  v8_current = unlist(p_old[fit_spec$name]),
  v8_fit = round(v_new, 4),
  lower = fit_spec$lower, upper = fit_spec$upper,
  row.names = NULL
)
write.csv(param_table, file.path(out_dir, "v8_fit_parameters.csv"), row.names = FALSE)

# -------------------------------
# 5. EVALUATION
# -------------------------------

risk_old <- predict_risk(p_old)
risk_new <- predict_risk(p_new)

auc <- function(score, label){
  pos <- score[label]; neg <- score[!label]
  if(length(pos) == 0 || length(neg) == 0) return(NA)
  mean(outer(pos, neg, ">") + 0.5 * outer(pos, neg, "=="))
}
safe_cor <- function(a, b, m = "pearson") if(sd(a) == 0 || sd(b) == 0) NA else cor(a, b, method = m)

metrics <- function(pred, obs, infection){
  nearest <- c(0.25, 0.5, 0.75, 1)[apply(abs(outer(pred, c(0.25, 0.5, 0.75, 1), "-")), 1, which.min)]
  c(
    RMSE      = sqrt(mean((pred - obs)^2)),
    MAE       = mean(abs(pred - obs)),
    Bias      = mean(pred - obs),
    R2        = 1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2),
    Pearson   = safe_cor(pred, obs),
    Spearman  = safe_cor(pred, obs, "spearman"),
    AUC_BIR50 = auc(pred, obs >= 0.5),
    AUC_BIR75 = auc(pred, obs >= 0.75),
    AUC_inf   = auc(pred, infection),
    Class_acc = mean(nearest == obs)
  )
}

sets <- list(
  Test_bloom  = list(i = test_idx,  obs = obs_test,  inf = pear$infection[test_idx]),
  Calibration = list(i = calib_idx, obs = obs_calib, inf = whatif$infection[calib_idx])
)
metric_table <- do.call(rbind, lapply(names(sets), function(s){
  x <- sets[[s]]
  rbind(
    data.frame(set = s, model = "v8 (current)", t(metrics(risk_old[x$i], x$obs, x$inf))),
    data.frame(set = s, model = "v8-fit",       t(metrics(risk_new[x$i], x$obs, x$inf)))
  )
}))
write.csv(metric_table, file.path(out_dir, "v8_fit_metrics.csv"), row.names = FALSE)

daily_test <- data.frame(date = pear$tm[test_idx], BIR = pear$BIR[test_idx],
                         level = pear$level[test_idx],
                         Risk_v8 = risk_old[test_idx], Risk_v8fit = risk_new[test_idx])
write.csv(daily_test, file.path(out_dir, "v8_fit_daily_test_bloom.csv"), row.names = FALSE)

sink(file.path(out_dir, "v8_fit_report.txt"), split = TRUE)
cat("Calibration days:", length(calib_idx),
    "| Test days (actual bloom):", length(test_idx),
    format(min(pear$tm[test_idx])), "~", format(max(pear$tm[test_idx])), "\n")
cat("Best SSE (calibration): v8 current =", round(sse_calib(start_old), 3),
    "-> v8-fit =", round(best$value, 3),
    "| starts reaching within 1% of best:",
    sum(sapply(fits, `[[`, "value") <= best$value * 1.01), "of", n_starts, "\n\n")
cat("Parameters\n"); print(param_table, row.names = FALSE)
cat("\nMetrics (Risk vs BIR/100)\n")
print(cbind(metric_table[, 1:2], round(metric_table[, -(1:2)], 3)), row.names = FALSE)
cat("\nDaily values, test period\n")
print(cbind(daily_test[, 1:3], round(daily_test[, 4:5], 3)), row.names = FALSE)
sink()

# -------------------------------
# 6. PLOTS: (b)-style daily infection risk
# -------------------------------

risk_panel <- function(dates, bir, level, r_old, r_new, title, bar_lwd, show_text){
  xr <- range(dates) + c(-0.5, 0.5)
  excluded <- range(dates[is.na(bir)])
  par(mar = c(3.5, 4.8, 2.6, 4.8))
  plot(dates, bir, type = "h", lwd = bar_lwd, col = "#fdae6b", lend = 1,
       xlim = xr, ylim = c(0, 110), xaxt = "n", xlab = "", ylab = "Pear BIR (0-100)")
  if(show_text) text(dates, bir + 5, level, cex = 0.8, col = "#a63603")
  if(all(is.finite(excluded))){
    rect(excluded[1] - 0.5, -5, excluded[2] + 0.5, 105, col = "#f0f0f0", border = NA)
    text(mean(excluded), 92, "actual bloom\n(test, excluded)", cex = 0.7, col = "grey40")
  }
  lines(dates, r_old * 100, type = "o", pch = 16, cex = 0.7, col = "#54278f", lwd = 2)
  lines(dates, r_new * 100, type = "o", pch = 17, cex = 0.7, col = "#1b9e77", lwd = 2)
  axis(4, at = seq(0, 100, 25), labels = seq(0, 1, 0.25))
  mtext("v8 Risk (0-1)", side = 4, line = 2.8, cex = 0.75)
  ticks <- if(diff(range(dates)) > 40) seq(min(dates), max(dates), by = "month") else
    seq(min(dates), max(dates), by = 2)
  axis.Date(1, at = ticks, format = if(diff(range(dates)) > 40) "%b" else "%m-%d")
  legend("topleft", c("Pear BIR (gold standard)", "v8 (current)", "v8-fit"),
         col = c("#fdae6b", "#54278f", "#1b9e77"), lwd = c(8, 2, 2),
         pch = c(NA, 16, 17), bty = "n", horiz = TRUE, cex = 0.85)
  title(title, adj = 0, cex.main = 1)
}

fmt <- function(set, model, m) {
  r <- metric_table[metric_table$set == set & metric_table$model == model, m]
  formatC(r, format = "f", digits = 3)
}

png(file.path(out_dir, "v8_fit_compare_bloom.png"), width = 2400, height = 1150, res = 250)
risk_panel(pear$tm[test_idx], pear$BIR[test_idx], pear$level[test_idx],
           risk_old[test_idx], risk_new[test_idx],
           "(b) Daily infection risk - test: actual bloom 2019 (not used in fitting)", 9, TRUE)
mtext(sprintf("RMSE: v8 %s -> v8-fit %s    |    Spearman: v8 %s -> v8-fit %s",
              fmt("Test_bloom", "v8 (current)", "RMSE"), fmt("Test_bloom", "v8-fit", "RMSE"),
              fmt("Test_bloom", "v8 (current)", "Spearman"), fmt("Test_bloom", "v8-fit", "Spearman")),
      side = 1, line = 2.4, cex = 0.8)
dev.off()

png(file.path(out_dir, "v8_fit_compare_calibration.png"), width = 2400, height = 1150, res = 250)
# Full what-if period with the excluded (test) days left blank
wi <- which(whatif$B)
blank <- function(v) replace(v, !(wi %in% calib_idx), NA)
risk_panel(whatif$tm[wi], blank(whatif$BIR[wi]), whatif$level[wi],
           blank(risk_old[wi]), blank(risk_new[wi]),
           "(b) Daily infection risk - calibration: what-if bloom 03-01 ~ 06-30 (actual bloom excluded)",
           3, FALSE)
mtext(sprintf("RMSE: v8 %s -> v8-fit %s    |    Spearman: v8 %s -> v8-fit %s    |    AUC infection: v8 %s -> v8-fit %s",
              fmt("Calibration", "v8 (current)", "RMSE"), fmt("Calibration", "v8-fit", "RMSE"),
              fmt("Calibration", "v8 (current)", "Spearman"), fmt("Calibration", "v8-fit", "Spearman"),
              fmt("Calibration", "v8 (current)", "AUC_inf"), fmt("Calibration", "v8-fit", "AUC_inf")),
      side = 1, line = 2.4, cex = 0.8)
dev.off()
