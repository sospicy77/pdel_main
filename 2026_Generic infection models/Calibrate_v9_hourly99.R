#############################################
# Calibrate bacterialmodel_v9 (fire_blight, base + add-on modules) against the
# pear fire blight model (gold standard) with hourly_99 (CF-99), 2010-2025,
# and compare v8, v8-fit2 and v9.
#
# Variants (module ablation, all calibrated the same way)
#   v8-fit2 : base modules only (results of Calibrate_v8_hourly99.R)
#   v9-D    : base + dispersal add-on (rain splash, insect)
#   v9-T    : base + infection add-on (temperature response f_T in I)
#   v9      : base + both add-ons
# Not calibratable here: wind and rain x wind dispersal (no wind in hourly_99),
# vulnerable period V (= 1 on every bloom day used for evaluation).
#
# Design: leave-one-season-out (LOSO) for evaluation, all seasons for final
# parameters. Target: score = (BIR - 25) / 75.
#############################################

# Run from the "2026_Generic infection models" folder after
# Calibrate_v8_hourly99.R.

library(parallel)

model_file <- "bacterialmodel_v9.R"
v8_file    <- "bacterialmodel_v8.R"
pear_dir   <- file.path("Output", "pear_99")
fit2_rdata <- file.path(pear_dir, "calibration", "calibration_results.RData")
out_dir    <- file.path(pear_dir, "calibration_v9")
disease    <- "fire_blight"
seasons    <- 2010:2025
n_starts   <- 12
n_boot     <- 2000
n_cores    <- max(1, min(4, detectCores()))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# -------------------------------
# 1. v9 FUNCTIONS, PARAMETERS, EARLIER RESULTS
# -------------------------------

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
m9 <- load_model(model_file)
m8 <- load_model(v8_file)
gp <- m9$global_params$growth_period
p_v8 <- m8$disease_params[[disease]]

prev <- new.env(); load(fit2_rdata, envir = prev)
fit2_par <- setNames(prev$param_table$v8_fit2, prev$param_table$parameter)

# v9 template: v8 base values + add-on settings; switches set per variant
p_template <- modifyList(m9$disease_params[[disease]], p_v8)
p_template$use_wind <- FALSE; p_template$use_interaction <- FALSE
p_template$use_vulnerable_period <- FALSE      # V = 1 on all evaluated bloom days

spec_all <- data.frame(
  name   = c("K_d", "N0", "mu_max", "T_opt", "rh_thresh_I", "rain_thresh",
             "rain_coeff", "insect_coeff"),
  module = c("D base", "N", "N", "N (+ I f_T in v9-T, v9)", "I base", "I base",
             "D add-on", "D add-on"),
  lower  = c(0.01, 0.01, 0.00, p_v8$T_min + 0.5, 40, 0.0, 0.0, 0.0),
  upper  = c(1.00, 0.99, 2.00, p_v8$T_max - 0.5, 100, 10.0, 1.0, 1.0)
)
variants <- list(
  v9_D = list(rain = TRUE,  insect = TRUE,  temp_I = FALSE),
  v9_T = list(rain = FALSE, insect = FALSE, temp_I = TRUE),
  v9   = list(rain = TRUE,  insect = TRUE,  temp_I = TRUE)
)
spec_for <- function(v){
  s <- spec_all
  if(!variants[[v]]$rain)   s <- s[s$name != "rain_coeff", ]
  if(!variants[[v]]$insect) s <- s[s$name != "insect_coeff", ]
  s
}
param_for <- function(v, values){
  p <- p_template
  p$use_rain <- variants[[v]]$rain; p$use_insect <- variants[[v]]$insect
  p$use_temp_I <- variants[[v]]$temp_I
  p[names(values)] <- as.list(values)
  p
}

# -------------------------------
# 2. DATA (bloom days + growth_period lead days)
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
gold$score <- (gold$BIR - 25) / 75

# Vectorized v9 (same equations as run_model; checked below)
lag_vec <- function(v, k) if(k == 0) v else c(rep(NA, k), v[seq_len(length(v) - k)])
fast_v9 <- function(p, f = frame){
  r <- p$mu_max * m9$temp_response(f$temp, p$T_min, p$T_opt, p$T_max) *
       m9$rh_response(f$rh, p$rh_thresh_M)
  N <- rep(p$N0, nrow(f))
  for(k in gp:0){
    N <- N + lag_vec(r, k) * N * (1 - N)
    N <- pmax(0, pmin(N, 1))
  }
  D <- p$K_d * m9$calc_dispersal_addon(f, p)
  I <- m9$wetness_response(f$rh, f$rain, p$rh_thresh_I, p$rain_thresh)
  if(p$use_temp_I) I <- I * m9$temp_response(f$temp, p$T_min, p$T_opt, p$T_max)
  list(N = N, D = D, I = I, Risk = N * D * I)
}

check_equal <- function(p){
  max(sapply(seasons, function(y){
    x <- read.csv(file.path(pear_dir, sprintf("pear_daily_%d.csv", y)))
    wx <- data.frame(date = as.Date(x$tm), temp = x$ta, rh = x$hm, rain = x$rn)
    full <- m9$run_model(wx, p, gp)$Risk
    fr <- frame[frame$season == y, ]
    idx <- match(fr$date[fr$bloom], wx$date)
    max(abs(fast_v9(p, fr)$Risk[fr$bloom] - full[idx]))
  }))
}
set.seed(3)
max_diff <- max(sapply(names(variants), function(v){
  s <- spec_for(v)
  check_equal(param_for(v, setNames(s$lower + runif(nrow(s)) * (s$upper - s$lower), s$name)))
}))
if(max_diff > 1e-12) stop("fast_v9 differs from run_model: ", max_diff)

# -------------------------------
# 3. CALIBRATION
# -------------------------------

calibrate <- function(v, train_seasons, seed){
  s <- spec_for(v)
  to_par <- function(z) setNames(s$lower + (s$upper - s$lower) * plogis(z), s$name)
  to_z <- function(val) qlogis(pmin(pmax((val - s$lower) / (s$upper - s$lower), 1e-6), 1 - 1e-6))
  sub <- which(frame$season %in% train_seasons)
  f <- frame[sub, ]
  pos <- which(f$bloom)
  y <- (f$BIR[pos] - 25) / 75
  sse <- function(z) sum((fast_v9(param_for(v, to_par(z)), f)$Risk[pos] - y)^2)
  # starts: v8 values, v8-fit2 values, random
  base_start <- unlist(p_template[s$name]); fit2_start <- base_start
  fit2_start[names(fit2_par)[names(fit2_par) %in% s$name]] <- fit2_par[names(fit2_par) %in% s$name]
  set.seed(seed)
  starts <- c(list(to_z(base_start), to_z(fit2_start)),
              replicate(n_starts - 2, rnorm(nrow(s), 0, 1.5), simplify = FALSE))
  fits <- lapply(starts, function(z0){
    a <- optim(z0, sse, method = "Nelder-Mead", control = list(maxit = 3000))
    optim(a$par, sse, method = "Nelder-Mead", control = list(maxit = 3000))
  })
  vals <- sapply(fits, `[[`, "value")
  list(par = to_par(fits[[which.min(vals)]]$par), sse = min(vals),
       converged = sum(vals <= min(vals) * 1.01))
}

t0 <- Sys.time()
jobs <- unlist(lapply(names(variants), function(v)
  c(list(list(variant = v, design = "ALL", test = NA, train = seasons)),
    lapply(seasons, function(s) list(variant = v, design = "LOSO", test = s,
                                     train = setdiff(seasons, s))))), recursive = FALSE)
results <- mclapply(seq_along(jobs), function(j)
  c(jobs[[j]], calibrate(jobs[[j]]$variant, jobs[[j]]$train, seed = 100 + j)),
  mc.cores = n_cores)
run_minutes <- as.numeric(difftime(Sys.time(), t0, units = "mins"))

fit_rows <- do.call(rbind, lapply(results, function(r){
  v <- setNames(rep(NA_real_, nrow(spec_all)), spec_all$name); v[names(r$par)] <- r$par
  data.frame(variant = r$variant, design = r$design, test = r$test, t(v),
             sse = r$sse, converged_starts = r$converged)
}))
write.csv(fit_rows, file.path(out_dir, "v9_all_fits.csv"), row.names = FALSE)

final <- lapply(setNames(names(variants), names(variants)), function(v)
  results[[which(sapply(results, function(r) r$variant == v && r$design == "ALL"))]])

at_bound <- function(name, val){
  k <- match(name, spec_all$name); w <- spec_all$upper[k] - spec_all$lower[k]
  ifelse(is.na(val), "", ifelse(abs(val - spec_all$lower[k]) < 1e-3 * w, "lower",
                         ifelse(abs(val - spec_all$upper[k]) < 1e-3 * w, "upper", "")))
}
param_table <- data.frame(parameter = spec_all$name, module = spec_all$module,
                          v8 = sapply(spec_all$name, function(n) if(is.null(p_v8[[n]])) NA else p_v8[[n]]),
                          v8_fit2 = fit2_par[spec_all$name], row.names = NULL)
for(v in names(variants)){
  val <- final[[v]]$par[spec_all$name]
  param_table[[v]] <- round(val, 4)
  param_table[[paste0(v, "_bound")]] <- at_bound(spec_all$name, val)
  lo <- fit_rows[fit_rows$variant == v & fit_rows$design == "LOSO", spec_all$name]
  param_table[[paste0(v, "_LOSO_range")]] <- ifelse(is.na(val), "",
    sprintf("%.3f-%.3f", apply(lo, 2, min), apply(lo, 2, max)))
}
param_table$lower <- spec_all$lower; param_table$upper <- spec_all$upper
write.csv(param_table, file.path(out_dir, "v9_parameters.csv"), row.names = FALSE)

# -------------------------------
# 4. PREDICTIONS
# -------------------------------

pred <- prev$pred[, c("season", "date", "temp", "rh", "rain", "BIR", "score", "infection",
                      "H", "W", "T", "EIP", "v8", "v8_fit2_loso", "v8_fit2_all")]
stopifnot(all(pred$date == gold$date))
for(v in names(variants)){
  pred[[paste0(v, "_all")]] <- fast_v9(param_for(v, final[[v]]$par))$Risk[bl]
  pred[[paste0(v, "_loso")]] <- NA
  for(r in results[sapply(results, function(r) r$variant == v && r$design == "LOSO")]){
    idx <- which(pred$season == r$test)
    pred[[paste0(v, "_loso")]][idx] <- fast_v9(param_for(v, r$par))$Risk[bl][idx]
  }
}
cmp <- fast_v9(param_for("v9", final$v9$par))
pred$N_v9 <- cmp$N[bl]; pred$D_v9 <- cmp$D[bl]; pred$I_v9 <- cmp$I[bl]
pred$fT_v9 <- m9$temp_response(pred$temp, p_template$T_min, final$v9$par["T_opt"], p_template$T_max)
write.csv(pred, file.path(out_dir, "v9_predictions_bloom_days.csv"), row.names = FALSE)

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
  c(RMSE = sqrt(mean((p - y)^2)), MAE = mean(abs(p - y)), Bias = mean(p - y),
    R2_NSE = 1 - sum((y - p)^2) / sum((y - mean(y))^2), r2 = r^2, Pearson_r = r,
    Spearman_rho = sc(p, y, "spearman"), Kendall_tau = sc(p, y, "kendall"),
    AUC_infection = auc(p, inf), AUC_score_ge_1_3 = auc(p, y >= 1/3 - 1e-9),
    AUC_score_ge_2_3 = auc(p, y >= 2/3 - 1e-9),
    Class_acc = mean(abs(nearest - y) < 1e-9))
}

versions <- c(v8 = "v8", v8_fit2 = "v8_fit2_loso", v9_D = "v9_D_loso",
              v9_T = "v9_T_loso", v9 = "v9_loso")
pooled <- sapply(versions, function(v) metrics(pred[[v]], pred$score, pred$infection))
insample <- sapply(c(v8_fit2 = "v8_fit2_all", v9_D = "v9_D_all", v9_T = "v9_T_all", v9 = "v9_all"),
                   function(v) metrics(pred[[v]], pred$score, pred$infection))

set.seed(99)
seas_rows <- split(seq_len(nrow(pred)), pred$season)
boot <- replicate(n_boot, {
  i <- unlist(seas_rows[sample(names(seas_rows), replace = TRUE)], use.names = FALSE)
  sapply(versions, function(v) metrics(pred[[v]][i], pred$score[i], pred$infection[i]))
}, simplify = "array")
ci_lo <- apply(boot, c(1, 2), quantile, 0.025, na.rm = TRUE)
ci_hi <- apply(boot, c(1, 2), quantile, 0.975, na.rm = TRUE)
diff_ci <- function(a, b){
  d <- boot[, a, ] - boot[, b, ]
  cbind(change = pooled[, a] - pooled[, b],
        CI_low = apply(d, 1, quantile, 0.025, na.rm = TRUE),
        CI_high = apply(d, 1, quantile, 0.975, na.rm = TRUE))
}
pooled_table <- data.frame(metric = rownames(pooled), round(pooled, 4),
                           v9_vs_v8 = round(diff_ci("v9", "v8"), 4),
                           v9_vs_v8_fit2 = round(diff_ci("v9", "v8_fit2"), 4),
                           v8_fit2_vs_v8 = round(diff_ci("v8_fit2", "v8"), 4),
                           row.names = NULL)
write.csv(pooled_table, file.path(out_dir, "v9_metrics_pooled.csv"), row.names = FALSE)

per_season <- do.call(rbind, lapply(split(pred, pred$season), function(x)
  do.call(rbind, lapply(names(versions), function(v)
    data.frame(season = x$season[1], version = v, days = nrow(x),
               infection_days = sum(x$infection),
               t(metrics(x[[versions[v]]], x$score, x$infection)))))))
write.csv(per_season, file.path(out_dir, "v9_metrics_by_season.csv"), row.names = FALSE)

# -------------------------------
# 6. REPORT
# -------------------------------

sink(file.path(out_dir, "v9_report.txt"), split = TRUE)
cat("CF-99 (hourly_99)", min(seasons), "-", max(seasons), "| bloom days:", nrow(gold),
    "| infection days:", sum(gold$infection), "\n")
cat("fast_v9 vs run_model max |diff|:", signif(max_diff, 3), "| fits:", length(results),
    "| run time:", round(run_minutes, 1), "min\n")
cat("Converged starts (ALL fits):",
    paste(sapply(names(final), function(v) sprintf("%s %d/%d", v, final[[v]]$converged, n_starts)),
          collapse = ", "), "\n\n")
cat("Parameters\n"); print(param_table, row.names = FALSE)
cat("\nPooled out-of-sample metrics (LOSO; v8 not fitted)\n")
print(cbind(metric = rownames(pooled), as.data.frame(round(pooled, 3))), row.names = FALSE)
cat("\nIn-sample (all seasons) metrics\n")
print(cbind(metric = rownames(insample), as.data.frame(round(insample, 3))), row.names = FALSE)
cat("\nChange with 95% season-block bootstrap CI\n")
for(cmp_name in c("v9_vs_v8", "v9_vs_v8_fit2", "v8_fit2_vs_v8")){
  cat("--", cmp_name, "--\n")
  print(cbind(metric = pooled_table$metric,
              pooled_table[, grep(paste0("^", cmp_name, "\\."), names(pooled_table))]),
        row.names = FALSE)
}
cat("\nMean prediction by BIR class (LOSO)\n")
print(round(sapply(versions, function(v) tapply(pred[[v]], pred$BIR, mean)), 3))
sink()

save(pred, per_season, pooled_table, pooled, insample, ci_lo, ci_hi, versions, param_table,
     fit_rows, spec_all, final, file = file.path(out_dir, "v9_results.RData"))
