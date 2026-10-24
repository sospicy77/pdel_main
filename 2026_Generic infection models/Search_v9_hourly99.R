#############################################
# v9 (fire_blight) parameter and structure search against the pear fire
# blight model (gold standard), hourly_99 / CF-99, 2010-2025.
#
# Stage 1: forward selection of extra calibratable parameters
# Stage 2: forward selection of literature-based structural add-ons
# Stage 3: growth_period grid for the selected configuration
# Selection criterion: leave-one-season-out (LOSO) RMSE of
#   score = (BIR - 25) / 75 on the 338 bloom days.
#
# Structural add-on candidates (fire blight infection biology):
#   M1 prev_rain : wetting also on the day after rain >= R_prev
#                  (Maryblyt: rain >= 2.5 mm on the previous day; Steiner 1990)
#   M2 dew       : wetting from dew / leaf wetness, estimated as hours with
#                  RH >= 90 % (LWD proxy; Sentelhas et al. 2008)
#   M3 dh_growth : epiphytic growth driven by degree-hours > 18.3 °C instead
#                  of daily mean temperature (Maryblyt EIP, Cougarblight)
#   M4 w_floor   : residual wetness probability (unobserved wetting such as
#                  light dew, nectar dilution) - semi-empirical
#############################################

# Run from the "2026_Generic infection models" folder after
# make_hourly_features.py and the pear model runs for hourly_99.

library(parallel)

pear_dir <- file.path("Output", "pear_99")
feat_file <- file.path(pear_dir, "hourly_features.csv")
out_dir  <- file.path(pear_dir, "search_v9")
seasons  <- 2010:2025
lead     <- 8
n_cores  <- max(1, min(4, detectCores()))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# -------------------------------
# 1. DATA
# -------------------------------
feat <- read.csv(feat_file); feat$date <- as.Date(feat$date)
frame <- do.call(rbind, lapply(seasons, function(y){
  x <- read.csv(file.path(pear_dir, sprintf("pear_daily_%d.csv", y)))
  x$tm <- as.Date(x$tm)
  f <- feat[feat$season == y, ]
  x$lwd_h <- f$lwd_h[match(x$tm, f$date)]
  x$dh183 <- f$dh183[match(x$tm, f$date)]
  b <- which(x$B == "True")
  keep <- (min(b) - lead):max(b)
  data.frame(season = y, date = x$tm[keep], temp = x$ta[keep], rh = x$hm[keep],
             rain = x$rn[keep], rain_prev = c(NA, x$rn)[keep], lwd_h = x$lwd_h[keep],
             dh183 = x$dh183[keep], bloom = x$B[keep] == "True", BIR = x$BIR[keep],
             infection = x$infection[keep] == "True", W = x$W[keep] == "True",
             H = x$H[keep] == "True", T = x$T[keep] == "True")
}))
bl <- which(frame$bloom)
y_all <- (frame$BIR - 25) / 75
stopifnot(!anyNA(frame[bl, c("temp", "rh", "rain", "rain_prev", "lwd_h", "dh183")]))

# -------------------------------
# 2. MODEL (vectorized v9 with candidate add-ons)
# -------------------------------
temp_response <- function(T, T_min, T_opt, T_max){
  out <- rep(0, length(T)); ok <- T > T_min & T < T_max
  e <- (T_max - T_opt) / (T_opt - T_min)
  out[ok] <- ((T[ok] - T_min) * (T_max - T[ok])^e) / ((T_opt - T_min) * (T_max - T_opt)^e)
  pmin(out, 1)
}
lag_vec <- function(v, k) if(k == 0) v else c(rep(NA, k), v[seq_len(length(v) - k)])

defaults <- list(N0 = 0.15, mu_max = 0.30, T_min = 4, T_opt = 28, T_max = 36,
                 rh_thresh_M = 50, k_M = 0.3, rh_thresh_I = 90, rain_thresh = 0.25,
                 k_I = 1, rain_coeff = 0.055, insect_coeff = 0.10, T_opt_I = 28,
                 rain_prev_thresh = 2.5, lwd_thresh = 6, DH_scale = 50, w_floor = 0)
bounds <- rbind(
  N0 = c(0.01, 0.99), mu_max = c(0, 2), T_min = c(-5, 10), T_opt = c(10.5, 34.5),
  T_max = c(35, 45), rh_thresh_M = c(0, 100), k_M = c(0.01, 2), rh_thresh_I = c(20, 100),
  rain_thresh = c(0, 10), k_I = c(0.05, 5), rain_coeff = c(0, 5), insect_coeff = c(0, 1),
  T_opt_I = c(10.5, 34.5), rain_prev_thresh = c(0.1, 20), lwd_thresh = c(0, 24),
  DH_scale = c(1, 300), w_floor = c(0, 1))

# cfg: list(free = parameter names, modules = character(), sep_TI = logical, gp = integer)
model_risk <- function(p, cfg, f = frame){
  gp <- cfg$gp
  if("dh_growth" %in% cfg$modules){
    g <- 1 - exp(-f$dh183 / p$DH_scale)
  } else {
    g <- temp_response(f$temp, p$T_min, p$T_opt, p$T_max)
  }
  r <- p$mu_max * g * plogis(p$k_M * (f$rh - p$rh_thresh_M))
  N <- rep(p$N0, nrow(f))
  for(k in gp:0){ N <- N + lag_vec(r, k) * N * (1 - N); N <- pmax(0, pmin(N, 1)) }
  D <- 1 - exp(-p$rain_coeff * f$rain) * (1 - p$insect_coeff)
  W <- ifelse(f$rain >= p$rain_thresh, 1, plogis(p$k_I * (f$rh - p$rh_thresh_I)))
  if("prev_rain" %in% cfg$modules) W <- pmax(W, as.numeric(f$rain_prev >= p$rain_prev_thresh))
  if("dew" %in% cfg$modules) W <- pmax(W, plogis(f$lwd_h - p$lwd_thresh))
  if("w_floor" %in% cfg$modules) W <- p$w_floor + (1 - p$w_floor) * W
  T_opt_I <- if(isTRUE(cfg$sep_TI)) p$T_opt_I else p$T_opt
  I <- W * temp_response(f$temp, p$T_min, T_opt_I, p$T_max)
  list(N = N, D = D, I = I, W = W, Risk = N * D * I)
}

# -------------------------------
# 3. CALIBRATION
# -------------------------------
make_p <- function(values){ p <- defaults; p[names(values)] <- as.list(values); p }
fit_cfg <- function(cfg, train, start, n_random = 3, seed = 1, maxit = 1500){
  free <- cfg$free; lo <- bounds[free, 1]; hi <- bounds[free, 2]
  to_par <- function(z) setNames(lo + (hi - lo) * plogis(z), free)
  to_z <- function(v) qlogis(pmin(pmax((v - lo) / (hi - lo), 1e-6), 1 - 1e-6))
  sub <- frame$season %in% train
  f <- frame[sub, ]; pos <- which(f$bloom); y <- (f$BIR[pos] - 25) / 75
  sse <- function(z) sum((model_risk(make_p(to_par(z)), cfg, f)$Risk[pos] - y)^2)
  set.seed(seed)
  starts <- c(list(to_z(unlist(start[free]))),
              replicate(n_random, rnorm(length(free), 0, 1.5), simplify = FALSE))
  fits <- lapply(starts, function(z0){
    a <- optim(z0, sse, method = "Nelder-Mead", control = list(maxit = maxit))
    optim(a$par, sse, method = "Nelder-Mead", control = list(maxit = maxit))
  })
  best <- fits[[which.min(sapply(fits, `[[`, "value"))]]
  list(par = to_par(best$par), sse = best$value)
}

auc <- function(s, l){ p <- s[l]; n <- s[!l]
  if(!length(p) || !length(n)) return(NA); mean(outer(p, n, ">") + 0.5 * outer(p, n, "==")) }
metrics <- function(p, y, inf) c(
  RMSE = sqrt(mean((p - y)^2)), R2_NSE = 1 - sum((y - p)^2) / sum((y - mean(y))^2),
  Pearson_r = cor(p, y), Spearman_rho = cor(p, y, method = "spearman"),
  AUC_infection = auc(p, inf), AUC_score_ge_1_3 = auc(p, y >= 1/3 - 1e-9),
  AUC_score_ge_2_3 = auc(p, y >= 2/3 - 1e-9))

# Evaluate one configuration: ALL fit (warm-started) + LOSO
evaluate_cfg <- function(cfg, start, label){
  all_fit <- fit_cfg(cfg, seasons, start, n_random = 7, seed = 11, maxit = 2500)
  start2 <- start; start2[names(all_fit$par)] <- as.list(all_fit$par)
  folds <- mclapply(seasons, function(s)
    fit_cfg(cfg, setdiff(seasons, s), start2, n_random = 2, seed = s),
    mc.cores = n_cores)
  pred <- rep(NA_real_, nrow(frame))
  for(k in seq_along(seasons)){
    idx <- which(frame$season == seasons[k])
    pred[idx] <- model_risk(make_p(folds[[k]]$par), cfg)$Risk[idx]
  }
  m <- metrics(pred[bl], y_all[bl], frame$infection[bl])
  cat(sprintf("[%s] %-55s k=%2d  LOSO RMSE %.4f  R2 %.3f  r %.3f  rho %.3f  AUCinf %.3f\n",
              format(Sys.time(), "%H:%M:%S"), label, length(cfg$free), m["RMSE"], m["R2_NSE"],
              m["Pearson_r"], m["Spearman_rho"], m["AUC_infection"]))
  list(cfg = cfg, label = label, all = all_fit, folds = folds, loso_pred = pred[bl], metrics = m,
       start = start2)
}

log_file <- file.path(out_dir, "search_log.txt")
sink(log_file, split = TRUE)
t0 <- Sys.time()

# Previous v9 (8 parameters incl. K_d) -> plan A: 7 parameters, same model
base_free <- c("N0", "mu_max", "T_opt", "rh_thresh_I", "rain_thresh", "rain_coeff", "insect_coeff")
prev <- read.csv(file.path(pear_dir, "calibration_v9", "v9_all_fits.csv"))
prev_all <- prev[prev$variant == "v9" & prev$design == "ALL", ]
start0 <- defaults; start0[base_free] <- as.list(unlist(prev_all[1, base_free]))
start0$rain_coeff <- min(start0$rain_coeff, 4.99)

results <- list()
cfg <- list(free = base_free, modules = character(), sep_TI = FALSE, gp = 3)
cur <- evaluate_cfg(cfg, start0, "v9 plan A (7 parameters)")
results[[cur$label]] <- cur

# -------------------------------
# Stage 1: extra parameters (forward selection)
# -------------------------------
cat("\n== Stage 1: extra parameters ==\n")
cand <- c("T_min", "T_max", "rh_thresh_M", "k_M", "k_I", "T_opt_I")
min_gain <- 0.002
repeat{
  if(!length(cand)) break
  trial <- lapply(cand, function(cn){
    c2 <- cur$cfg; c2$free <- c(c2$free, cn); if(cn == "T_opt_I") c2$sep_TI <- TRUE
    st <- cur$start; if(cn == "T_opt_I") st$T_opt_I <- st$T_opt
    evaluate_cfg(c2, st, paste0(cur$label, " + ", cn))
  })
  for(t in trial) results[[t$label]] <- t
  best <- trial[[which.min(sapply(trial, function(t) t$metrics["RMSE"]))]]
  if(cur$metrics["RMSE"] - best$metrics["RMSE"] < min_gain) break
  cur <- best; cand <- setdiff(cand, tail(cur$cfg$free, 1))
}
stage1 <- cur
cat("Stage 1 selected:", stage1$label, "\n")

# -------------------------------
# Stage 2: structural add-ons (forward selection)
# -------------------------------
cat("\n== Stage 2: structural add-ons ==\n")
mod_par <- list(prev_rain = "rain_prev_thresh", dew = "lwd_thresh",
                dh_growth = "DH_scale", w_floor = "w_floor")
mods <- names(mod_par)
cur <- stage1
repeat{
  if(!length(mods)) break
  trial <- lapply(mods, function(mn){
    c2 <- cur$cfg; c2$modules <- c(c2$modules, mn); c2$free <- c(c2$free, mod_par[[mn]])
    evaluate_cfg(c2, cur$start, paste0(cur$label, " + [", mn, "]"))
  })
  for(t in trial) results[[t$label]] <- t
  best <- trial[[which.min(sapply(trial, function(t) t$metrics["RMSE"]))]]
  if(cur$metrics["RMSE"] - best$metrics["RMSE"] < min_gain) break
  cur <- best; mods <- setdiff(mods, tail(cur$cfg$modules, 1))
}
stage2 <- cur
cat("Stage 2 selected:", stage2$label, "\n")

# -------------------------------
# Stage 3: growth_period grid
# -------------------------------
cat("\n== Stage 3: growth_period ==\n")
gp_trials <- lapply(setdiff(1:7, stage2$cfg$gp), function(g){
  c2 <- stage2$cfg; c2$gp <- g
  evaluate_cfg(c2, stage2$start, paste0(stage2$label, " | gp=", g))
})
for(t in gp_trials) results[[t$label]] <- t
all_gp <- c(list(stage2), gp_trials)
final <- all_gp[[which.min(sapply(all_gp, function(t) t$metrics["RMSE"]))]]
cat("Final:", final$label, " gp =", final$cfg$gp, "\n")
cat("Run time:", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), "min\n")
sink()

summary_tab <- do.call(rbind, lapply(results, function(r)
  data.frame(config = r$label, n_par = length(r$cfg$free), gp = r$cfg$gp,
             modules = paste(r$cfg$modules, collapse = "+"), t(round(r$metrics, 4)))))
write.csv(summary_tab, file.path(out_dir, "search_summary.csv"), row.names = FALSE)
save(results, stage1, stage2, final, frame, bl, y_all, defaults, bounds,
     file = file.path(out_dir, "search_results.RData"))
