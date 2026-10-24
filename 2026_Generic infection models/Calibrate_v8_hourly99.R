#############################################
# Re-calibrate bacterialmodel_v8 (fire_blight) against the pear fire blight
# model (gold standard) with hourly_99 (CF-99), seasons 2010-2025
#
# Versions compared
#   v8       : current parameters
#   v8-fit   : fitted earlier at ID1001 2019 (target BIR/100)
#   v8-fit2  : fitted here with hourly_99 (target (BIR - 25) / 75)
#
# Designs
#   LOSO     : leave-one-season-out cross-validation (main evaluation)
#   ALL      : all 16 seasons -> final v8-fit2 parameters
#   SINGLE   : one season for fitting, the other 15 for testing (x 16)
#   ALL_B100 : all seasons with target BIR/100 (diagnostic for the target choice)
#############################################

# Run from the "2026_Generic infection models" folder after:
#   python3 pear_fire_blight_fixed.py obs/hourly_99 2010 ... 2025 Output/pear_99

library(parallel)

model_file <- "bacterialmodel_v8.R"
fit1_file  <- file.path("Output", "pear_ID1001_2019", "v8_fit_parameters.csv")
pear_dir   <- file.path("Output", "pear_99")
out_dir    <- file.path("Output", "pear_99", "calibration")
disease    <- "fire_blight"
seasons    <- 2010:2025
n_starts   <- 12
n_boot     <- 2000
n_cores    <- max(1, min(4, detectCores()))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# -------------------------------
# 1. v8 FUNCTIONS AND PARAMETER SETS
# -------------------------------

model_exprs <- parse(model_file, encoding = "UTF-8")
for(e in model_exprs){
  if(is.call(e) && identical(e[[1]], as.name("<-")) &&
     (as.character(e[[2]]) %in% c("global_params", "disease_params") ||
      (is.call(e[[3]]) && identical(e[[3]][[1]], as.name("function"))))){
    eval(e, envir = globalenv())
  }
}
gp    <- global_params$growth_period
p_old <- disease_params[[disease]]
fit1  <- read.csv(fit1_file)
p_fit1 <- p_old; p_fit1[fit1$parameter] <- as.list(fit1$v8_fit)

# Parameters to calibrate and their search bounds
fit_spec <- data.frame(
  name  = c("K_d", "N0",  "mu_max", "T_opt",           "rh_thresh_I", "rain_thresh"),
  module = c("D",  "N",   "N",      "N (f_T)",         "I (wetness)", "I (wetness)"),
  lower = c(0.01,  0.01,  0.00,     p_old$T_min + 0.5, 40,            0.0),
  upper = c(1.00,  0.99,  2.00,     p_old$T_max - 0.5, 100,           10.0)
)

# -------------------------------
# 2. DATA: bloom days of each season + growth_period lead days
# -------------------------------

frame <- do.call(rbind, lapply(seasons, function(y){
  x <- read.csv(file.path(pear_dir, sprintf("pear_daily_%d.csv", y)))
  b <- which(x$B == "True")
  keep <- (min(b) - gp):max(b)
  data.frame(season = y, date = as.Date(x$tm[keep]), temp = x$ta[keep],
             rh = x$hm[keep], rain = x$rn[keep], bloom = x$B[keep] == "True",
             BIR = x$BIR[keep], infection = x$infection[keep] == "True",
             H = x$H[keep] == "True", W = x$W[keep] == "True", T = x$T[keep] == "True",
             EIP = x$EIP[keep])
}))
bl <- which(frame$bloom)
gold <- frame[bl, ]
gold$score <- (gold$BIR - 25) / 75          # 0, 1/3, 2/3, 1 = share of H, W, T met

# Fast, vectorized v8 Risk (same equations as run_model)
lag_vec <- function(v, k) if(k == 0) v else c(rep(NA, k), v[seq_len(length(v) - k)])
fast_risk <- function(p, f = frame){
  r <- p$mu_max * temp_response(f$temp, p$T_min, p$T_opt, p$T_max) *
       rh_response(f$rh, p$rh_thresh_M)
  N <- rep(p$N0, nrow(f))
  for(k in gp:0){
    N <- N + lag_vec(r, k) * N * (1 - N)
    N <- pmax(0, pmin(N, 1))
  }
  I <- wetness_response(f$rh, f$rain, p$rh_thresh_I, p$rain_thresh)
  list(N = N, D = rep(p$K_d, nrow(f)), I = I, Risk = N * p$K_d * I)
}

# Check: fast version == run_model on full seasons
check_equal <- function(p){
  max(sapply(seasons, function(y){
    x <- read.csv(file.path(pear_dir, sprintf("pear_daily_%d.csv", y)))
    full <- run_model(data.frame(date = as.Date(x$tm), temp = x$ta, rh = x$hm, rain = x$rn), p, gp)$Risk
    fr <- frame[frame$season == y, ]
    fast <- fast_risk(p, fr)$Risk
    idx <- match(fr$date[fr$bloom], as.Date(x$tm))
    max(abs(fast[fr$bloom] - full[idx]))
  }))
}
set.seed(1)
rand_p <- function(){
  p <- p_old
  p[fit_spec$name] <- as.list(fit_spec$lower + runif(nrow(fit_spec)) * (fit_spec$upper - fit_spec$lower))
  p
}
max_diff <- max(check_equal(p_old), check_equal(p_fit1), replicate(3, check_equal(rand_p())))
if(max_diff > 1e-12) stop("fast_risk differs from run_model: ", max_diff)

# -------------------------------
# 3. CALIBRATION
# -------------------------------

to_par <- function(z) fit_spec$lower + (fit_spec$upper - fit_spec$lower) * plogis(z)
to_z   <- function(v) qlogis(pmin(pmax((v - fit_spec$lower) /
                                       (fit_spec$upper - fit_spec$lower), 1e-6), 1 - 1e-6))
with_values <- function(p, v){ p[fit_spec$name] <- as.list(v); p }

calibrate <- function(train_seasons, target = c("score", "bir100"), seed = 1){
  target <- match.arg(target)
  rows <- bl[frame$season[bl] %in% train_seasons]
  sub_idx <- which(frame$season %in% train_seasons)
  f <- frame[sub_idx, ]
  pos <- match(rows, sub_idx)
  y <- if(target == "score") (frame$BIR[rows] - 25) / 75 else frame$BIR[rows] / 100
  sse <- function(z) sum((fast_risk(with_values(p_old, to_par(z)), f)$Risk[pos] - y)^2)
  set.seed(seed)
  starts <- c(list(to_z(unlist(p_old[fit_spec$name]))),
              replicate(n_starts - 1, rnorm(nrow(fit_spec), 0, 1.5), simplify = FALSE))
  fits <- lapply(starts, function(z0){
    a <- optim(z0, sse, method = "Nelder-Mead", control = list(maxit = 2000))
    optim(a$par, sse, method = "Nelder-Mead", control = list(maxit = 2000))
  })
  vals <- sapply(fits, `[[`, "value")
  best <- fits[[which.min(vals)]]
  list(par = setNames(to_par(best$par), fit_spec$name), sse = best$value,
       n_train = length(rows), converged = sum(vals <= min(vals) * 1.01))
}

t0 <- Sys.time()
jobs <- c(list(list(design = "ALL", train = seasons, test = seasons, target = "score")),
          list(list(design = "ALL_B100", train = seasons, test = seasons, target = "bir100")),
          lapply(seasons, function(s) list(design = "LOSO", train = setdiff(seasons, s),
                                           test = s, target = "score")),
          lapply(seasons, function(s) list(design = "SINGLE", train = s,
                                           test = setdiff(seasons, s), target = "score")))
results <- mclapply(seq_along(jobs), function(j){
  jb <- jobs[[j]]
  c(jb, calibrate(jb$train, jb$target, seed = j))
}, mc.cores = n_cores)
run_minutes <- as.numeric(difftime(Sys.time(), t0, units = "mins"))

param_rows <- do.call(rbind, lapply(results, function(r)
  data.frame(design = r$design,
             test = if(r$design == "LOSO") r$test else if(r$design == "SINGLE") NA else NA,
             train = if(r$design == "SINGLE") r$train else NA,
             t(r$par), sse = r$sse, n_train = r$n_train, converged_starts = r$converged)))
write.csv(param_rows, file.path(out_dir, "calibration_all_fits.csv"), row.names = FALSE)

res_all  <- results[[1]]
res_b100 <- results[[2]]
p_fit2   <- with_values(p_old, res_all$par)
p_b100   <- with_values(p_old, res_b100$par)

at_bound <- function(v) ifelse(abs(v - fit_spec$lower) < 1e-3 * (fit_spec$upper - fit_spec$lower), "lower bound",
                        ifelse(abs(v - fit_spec$upper) < 1e-3 * (fit_spec$upper - fit_spec$lower), "upper bound", ""))
loso_par <- do.call(rbind, lapply(results[sapply(results, `[[`, "design") == "LOSO"], `[[`, "par"))
param_table <- data.frame(
  parameter = fit_spec$name, module = fit_spec$module,
  v8 = unlist(p_old[fit_spec$name]),
  v8_fit_ID1001 = unlist(p_fit1[fit_spec$name]),
  v8_fit2 = round(res_all$par, 4),
  v8_fit2_bound = at_bound(res_all$par),
  LOSO_min = round(apply(loso_par, 2, min), 4),
  LOSO_max = round(apply(loso_par, 2, max), 4),
  target_BIR100_fit = round(res_b100$par, 4),
  BIR100_bound = at_bound(res_b100$par),
  lower = fit_spec$lower, upper = fit_spec$upper, row.names = NULL)
write.csv(param_table, file.path(out_dir, "v8_fit2_parameters.csv"), row.names = FALSE)

# -------------------------------
# 4. PREDICTIONS
# -------------------------------

pred <- gold[, c("season", "date", "temp", "rh", "rain", "BIR", "score",
                 "infection", "H", "W", "T", "EIP")]
pred$v8       <- fast_risk(p_old)$Risk[bl]
pred$v8_fit   <- fast_risk(p_fit1)$Risk[bl]
pred$v8_fit2_all <- fast_risk(p_fit2)$Risk[bl]
pred$v8_fit2_loso <- NA
for(r in results[sapply(results, `[[`, "design") == "LOSO"]){
  idx <- which(pred$season == r$test)
  pred$v8_fit2_loso[idx] <- fast_risk(with_values(p_old, r$par))$Risk[bl][idx]
}
# Components of v8-fit2 (all seasons) for diagnostics
comp <- fast_risk(p_fit2)
pred$N_fit2 <- comp$N[bl]; pred$I_fit2 <- comp$I[bl]
comp0 <- fast_risk(p_old)
pred$N_v8 <- comp0$N[bl]; pred$I_v8 <- comp0$I[bl]
write.csv(pred, file.path(out_dir, "predictions_bloom_days.csv"), row.names = FALSE)

# -------------------------------
# 5. METRICS
# -------------------------------

auc <- function(score, label){
  pos <- score[label]; neg <- score[!label]
  if(length(pos) == 0 || length(neg) == 0) return(NA)
  mean(outer(pos, neg, ">") + 0.5 * outer(pos, neg, "=="))
}
sc <- function(a, b, m) if(length(a) < 3 || sd(a) == 0 || sd(b) == 0) NA else cor(a, b, method = m)

metrics <- function(p, y, inf){
  nearest <- c(0, 1/3, 2/3, 1)[apply(abs(outer(p, c(0, 1/3, 2/3, 1), "-")), 1, which.min)]
  r <- sc(p, y, "pearson")
  c(RMSE = sqrt(mean((p - y)^2)),
    MAE = mean(abs(p - y)),
    Bias = mean(p - y),
    R2_NSE = 1 - sum((y - p)^2) / sum((y - mean(y))^2),
    r2 = r^2,
    Pearson_r = r,
    Spearman_rho = sc(p, y, "spearman"),
    Kendall_tau = sc(p, y, "kendall"),
    AUC_infection = auc(p, inf),
    AUC_score_ge_1_3 = auc(p, y >= 1/3 - 1e-9),
    AUC_score_ge_2_3 = auc(p, y >= 2/3 - 1e-9),
    Class_acc = mean(abs(nearest - y) < 1e-9))
}
versions <- c(v8 = "v8", v8_fit = "v8_fit", v8_fit2_loso = "v8_fit2_loso",
              v8_fit2_all = "v8_fit2_all")
pooled <- sapply(versions, function(v) metrics(pred[[v]], pred$score, pred$infection))

# Season-block bootstrap: 95% CI for each version and for (fit2 LOSO - v8)
set.seed(99)
seas_rows <- split(seq_len(nrow(pred)), pred$season)
boot <- replicate(n_boot, {
  s <- sample(names(seas_rows), replace = TRUE)
  i <- unlist(seas_rows[s], use.names = FALSE)
  sapply(versions, function(v) metrics(pred[[v]][i], pred$score[i], pred$infection[i]))
}, simplify = "array")
ci_lo <- apply(boot, c(1, 2), quantile, 0.025, na.rm = TRUE)
ci_hi <- apply(boot, c(1, 2), quantile, 0.975, na.rm = TRUE)
d_boot <- boot[, "v8_fit2_loso", ] - boot[, "v8", ]
pooled_table <- data.frame(metric = rownames(pooled),
                           v8 = pooled[, "v8"], v8_fit = pooled[, "v8_fit"],
                           v8_fit2_LOSO = pooled[, "v8_fit2_loso"],
                           v8_fit2_all_insample = pooled[, "v8_fit2_all"],
                           change_fit2_vs_v8 = pooled[, "v8_fit2_loso"] - pooled[, "v8"],
                           change_CI_low = apply(d_boot, 1, quantile, 0.025, na.rm = TRUE),
                           change_CI_high = apply(d_boot, 1, quantile, 0.975, na.rm = TRUE),
                           row.names = NULL)
write.csv(pooled_table, file.path(out_dir, "metrics_pooled.csv"), row.names = FALSE)

per_season <- do.call(rbind, lapply(split(pred, pred$season), function(x){
  do.call(rbind, lapply(names(versions), function(v)
    data.frame(season = x$season[1], version = v, days = nrow(x),
               infection_days = sum(x$infection),
               t(metrics(x[[v]], x$score, x$infection)))))
}))
write.csv(per_season, file.path(out_dir, "metrics_by_season.csv"), row.names = FALSE)

# Single-season calibration: performance on the other 15 seasons
single <- do.call(rbind, lapply(results[sapply(results, `[[`, "design") == "SINGLE"], function(r){
  idx <- pred$season %in% r$test
  pr <- fast_risk(with_values(p_old, r$par))$Risk[bl][idx]
  data.frame(train_season = r$train, n_train = r$n_train,
             t(metrics(pr, pred$score[idx], pred$infection[idx])))
}))
write.csv(single, file.path(out_dir, "single_season_calibration.csv"), row.names = FALSE)

# Target diagnostic: fit with BIR/100 vs (BIR - 25) / 75, evaluated in-sample
target_diag <- rbind(
  data.frame(target = "(BIR-25)/75", t(metrics(fast_risk(p_fit2)$Risk[bl], gold$score, gold$infection)),
             mean_I = mean(fast_risk(p_fit2)$I[bl])),
  data.frame(target = "BIR/100", t(metrics(fast_risk(p_b100)$Risk[bl], gold$score, gold$infection)),
             mean_I = mean(fast_risk(p_b100)$I[bl])))
write.csv(target_diag, file.path(out_dir, "target_choice_diagnostic.csv"), row.names = FALSE)

# -------------------------------
# 6. REPORT
# -------------------------------

sink(file.path(out_dir, "calibration_report.txt"), split = TRUE)
cat("CF-99 (hourly_99) seasons", min(seasons), "-", max(seasons), "| bloom days:", nrow(gold),
    "| infection days:", sum(gold$infection), "\n")
cat("Gold-standard target: score = (BIR - 25) / 75\n"); print(table(BIR = gold$BIR))
cat("fast_risk vs run_model max |diff|:", signif(max_diff, 3), "\n")
cat("Fits:", length(results), "| starts per fit:", n_starts, "| run time:",
    round(run_minutes, 1), "min\n")
cat("ALL fit: converged starts", res_all$converged, "of", n_starts, "\n\n")
cat("Parameters\n"); print(param_table, row.names = FALSE)
cat("\nPooled metrics (all 338 bloom days; v8_fit2_LOSO = out-of-sample)\n")
print(cbind(pooled_table[, 1, drop = FALSE], round(pooled_table[, -1], 3)), row.names = FALSE)
cat("\nMean prediction by BIR class\n")
print(round(sapply(versions, function(v) tapply(pred[[v]], pred$BIR, mean)), 3))
cat("\nSingle-season calibration -> other 15 seasons (summary over 16 choices)\n")
print(round(sapply(single[, c("RMSE", "R2_NSE", "Pearson_r", "Spearman_rho", "AUC_infection")],
                   function(v) c(min = min(v, na.rm = TRUE), median = median(v, na.rm = TRUE),
                                 max = max(v, na.rm = TRUE))), 3))
cat("\nTarget choice diagnostic (in-sample, all seasons)\n")
print(cbind(target_diag[, 1, drop = FALSE], round(target_diag[, -1], 3)), row.names = FALSE)
sink()

save(pred, per_season, pooled_table, single, param_table, param_rows, target_diag,
     fit_spec, ci_lo, ci_hi, versions, file = file.path(out_dir, "calibration_results.RData"))
