#############################################
# BGMP basic-module algorithm selection test, all candidates calibrated
# (hourly_99, CF-99)
#
# Every candidate algorithm of each component is calibrated to the
# corresponding output of the example models (pear fire blight, pepper
# bacterial spot) with the same objective (sum of squared errors), the
# same optimiser (bounded multi-start Nelder-Mead, grid search for
# threshold parameters) and evaluated by leave-one-season-out (LOSO)
# cross-validation.
#
#   Multiplication : M1 doubling time, M2 continuous logistic,
#                    M3 discrete logistic (gp = 1..7)
#                    free: mu_max, N0, T_opt, rh_thresh_M (+ gp for M3)
#                    target: example multiplication index (ex_M)
#   Dispersal      : D1 constant K_d, D2 1 - exp(-a R), D3 1[R >= R_th]
#                    target: example dispersal day (ex_D, 0/1)
#   Infection      : I1 wetness W(rain, RH)            free: RH_th, Rain_th, k_I
#                    I2 W x f_T(T)                     + T_opt_I
#                    I3 min(1, LWD / L)                free: L (hours)
#                    target: example infection day (ex_I, 0/1)
#   Risk level     : Risk = M3 x D x I with M3 fixed at its calibration and
#                    D, I parameters fitted jointly to the example risk
#                    (fire blight: (BIR - 25) / 75; bacterial spot: ex_I)
#
# Run from the "2026_Generic infection models" folder after
# Compare_BGMP_components_hourly99.R (uses its daily example-model output).
#############################################

invisible(Sys.setlocale("LC_CTYPE", "C.UTF-8"))
library(parallel)
n_cores <- max(1, min(4, detectCores()))
in_dir  <- file.path("Output", "component_test_99")
out_dir <- file.path("Output", "component_test_99_calibrated")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
set.seed(2026)

# -------------------------------
# 1. DATA
# -------------------------------
hr <- read.csv(file.path("obs", "hourly_99"))
hr[hr == -999] <- NA
hr$date <- as.Date(substr(hr$data_tm, 1, 10))
hr$rn[is.na(hr$rn)] <- 0
for (v in c("ta", "hm")) {
  ok <- !is.na(hr[[v]])
  hr[[v]] <- approx(which(ok), hr[[v]][ok], seq_len(nrow(hr)), rule = 2)$y
}
day <- aggregate(cbind(temp = ta, rh = hm) ~ date, hr, mean)
day$rain <- aggregate(rn ~ date, hr, sum)$rn
lw <- aggregate(cbind(lwd = as.numeric(hm >= 90 | rn > 0)) ~ date, hr, sum)
day$lwd <- lw$lwd[match(day$date, lw$date)]

load_example <- function(file) {
  x <- read.csv(file.path(in_dir, file))
  x$date <- as.Date(x$date)
  x <- x[order(x$season, x$day), ]
  w <- day[match(x$date, day$date), c("temp", "rh", "rain", "lwd")]
  cbind(x[, intersect(names(x), c("season", "day", "date", "ex_M", "ex_D", "ex_I", "BIR"))], w)
}
DAT <- list(fb = load_example("fire_blight_daily.csv"),
            bs = load_example("bacterial_spot_daily.csv"))
DAT$fb$target_risk <- (DAT$fb$BIR - 25) / 75
DAT$bs$target_risk <- DAT$bs$ex_I
TLIM <- list(fb = c(4, 36), bs = c(5, 35))       # v8 T_min, T_max
dis_lab <- c(fb = "Pear fire blight (bloom)", bs = "Pepper bacterial spot (Jun-Aug)")

# -------------------------------
# 2. RESPONSE FUNCTIONS AND CANDIDATES
# -------------------------------
f_T <- function(T, a, o, b) {
  r <- rep(0, length(T)); v <- T > a & T < b
  e <- (b - o) / (o - a)
  r[v] <- ((T[v] - a) * (b - T[v])^e) / ((o - a) * (b - o)^e)
  pmin(r, 1)
}
lead <- 7
blocks <- function(df) lapply(split(df, df$season), function(s) {
  d <- day[day$date >= min(s$date) - lead & day$date <= max(s$date), ]
  list(d = d, k = (lead + 1):nrow(d))
})
BLK <- lapply(DAT, blocks)

# Multiplication: th = (mu_max, N0, T_opt, rh_thresh_M)
mult_sim <- function(alg, th, blk, Tlim, gp = 3) unlist(lapply(blk, function(b) {
  r <- th[1] * f_T(b$d$temp, Tlim[1], th[3], Tlim[2]) * plogis(0.3 * (b$d$rh - th[4]))
  k <- b$k; x <- th[2]
  if (alg == "M1") vapply(r[k], function(z) { x <<- min(1, x * 2^(z / log(2))); x }, 0)
  else if (alg == "M2") vapply(r[k], function(z) { x <<- min(1, x + z * x * (1 - x)); x }, 0)
  else vapply(k, function(t) {
    y <- th[2]
    for (s in (t - gp):t) y <- min(1, max(0, y + r[s] * y * (1 - y)))
    y
  }, 0)
}))

# Dispersal and infection: vectorised on a data frame
wet <- function(d, rh_th, rain_th, k) ifelse(d$rain >= rain_th, 1, plogis(k * (d$rh - rh_th)))
D_fun <- list(
  D1 = function(c, g, d) rep(c[1], nrow(d)),
  D2 = function(c, g, d) 1 - exp(-c[1] * d$rain),
  D3 = function(c, g, d) as.numeric(d$rain >= g[1]))
I_fun <- function(Tlim) list(
  I1 = function(c, g, d) wet(d, c[1], g[1], c[2]),
  I2 = function(c, g, d) wet(d, c[1], g[1], c[2]) * f_T(d$temp, Tlim[1], c[3], Tlim[2]),
  I3 = function(c, g, d) pmin(1, d$lwd / c[1]))

# -------------------------------
# 3. GENERIC CALIBRATION
# -------------------------------
# cont: list(lower, upper) of continuous parameters (logistic transform)
# grid: list of candidate vectors for threshold parameters
nm_fit <- function(obj, lo, hi, starts = 4, maxit = 400) {
  tf <- function(z) lo + (hi - lo) * plogis(z)
  best <- NULL
  for (i in seq_len(starts)) {
    z0 <- if (i == 1) rep(0, length(lo)) else rnorm(length(lo), 0, 1.5)
    o <- if (length(lo) == 1)
      optim(z0, function(z) obj(tf(z)), method = "Brent", lower = -10, upper = 10)
    else {
      o <- optim(z0, function(z) obj(tf(z)), control = list(maxit = maxit))
      optim(o$par, function(z) obj(tf(z)), control = list(maxit = maxit))
    }
    if (is.null(best) || o$value < best$value) best <- o
  }
  list(par = tf(best$par), value = best$value)
}
calibrate <- function(pred, y, rows, lo = NULL, hi = NULL, grid = NULL, starts = 4) {
  G <- if (is.null(grid)) data.frame(dummy = NA) else expand.grid(grid)
  fit_g <- function(g, starts) {
    g <- unlist(g)
    if (is.null(lo)) return(list(par = numeric(0), g = g,
                                 value = sum((pred(numeric(0), g)[rows] - y[rows])^2)))
    f <- nm_fit(function(c) sum((pred(c, g)[rows] - y[rows])^2), lo, hi, starts)
    c(f, list(g = g))
  }
  coarse <- lapply(seq_len(nrow(G)), function(i) fit_g(G[i, , drop = FALSE], 1))
  top <- order(sapply(coarse, `[[`, "value"))[seq_len(min(3, nrow(G)))]
  fine <- lapply(top, function(i) fit_g(G[i, , drop = FALSE], starts))
  fine[[which.min(sapply(fine, `[[`, "value"))]]
}
# Full-data fit + LOSO predictions
fit_eval <- function(pred, y, season, lo = NULL, hi = NULL, grid = NULL, starts = 4) {
  all_rows <- seq_along(y)
  full <- calibrate(pred, y, all_rows, lo, hi, grid, starts)
  loso <- numeric(length(y))
  for (s in unique(season)) {
    tr <- which(season != s)
    f <- calibrate(pred, y, tr, lo, hi, grid, max(2, starts - 2))
    loso[season == s] <- pred(f$par, f$g)[season == s]
  }
  list(full = full, fitted = pred(full$par, full$g), loso = loso)
}

# -------------------------------
# 4. METRICS
# -------------------------------
auc <- function(score, obs) {
  obs <- obs > 0
  if (all(obs) || !any(obs)) return(NA)
  rk <- rank(score)
  (sum(rk[obs]) - sum(obs) * (sum(obs) + 1) / 2) / (sum(obs) * sum(!obs))
}
rmse <- function(p, y) sqrt(mean((p - y)^2))
r2   <- function(p, y) 1 - sum((p - y)^2) / sum((y - mean(y))^2)
rho  <- function(p, y) suppressWarnings(cor(p, y, method = "spearman"))
decline_pct <- function(x, s) 100 * mean(unlist(tapply(x, s, function(z) diff(z) < -1e-6)))
bic <- function(p, y, k) length(y) * log(mean((p - y)^2)) + k * log(length(y))
fmt_par <- function(f) paste(signif(c(f$par, f$g), 4), collapse = ", ")

# -------------------------------
# 5. MULTIPLICATION
# -------------------------------
mult_eval <- function(dz) {
  df <- DAT[[dz]]; blk <- BLK[[dz]]; Tl <- TLIM[[dz]]
  lo <- c(0.01, 0.001, Tl[1] + 1, 0); hi <- c(2, 0.9, Tl[2] - 1, 100)
  y <- df$ex_M
  # gp of M3 chosen on the full data (RMSE), then fixed in LOSO
  gp_fits <- mclapply(1:7, function(g)
    calibrate(function(c, gg) mult_sim("M3", c, blk, Tl, g), y, seq_along(y), lo, hi, starts = 4),
    mc.cores = n_cores)
  gp_tab <- data.frame(disease = dz, gp = 1:7,
                       rmse = sqrt(sapply(gp_fits, `[[`, "value") / length(y)))
  gp_best <- gp_tab$gp[which.min(gp_tab$rmse)]
  algs <- c("M1", "M2", "M3")
  res <- mclapply(algs, function(a) {
    g <- if (a == "M3") gp_best else 3
    fit_eval(function(c, gg) mult_sim(a, c, blk, Tl, g), y, df$season, lo, hi, starts = 4)
  }, mc.cores = n_cores)
  names(res) <- algs
  list(res = res, gp_tab = gp_tab, gp_best = gp_best)
}

# Start-date sensitivity with the calibrated parameters
start_shift <- function(dz, alg, th, gp, shift = 10) {
  df <- DAT[[dz]]; Tl <- TLIM[[dz]]
  mean(sapply(split(df, df$season), function(s) {
    first <- min(s$date); last <- max(s$date)
    mk <- function(extra) {
      d <- day[day$date >= first - lead - extra & day$date <= last, ]
      list(list(d = d, k = (lead + extra + 1):nrow(d)))
    }
    mean(abs(mult_sim(alg, th, mk(0), Tl, gp) - mult_sim(alg, th, mk(shift), Tl, gp)))
  }))
}

# -------------------------------
# 6. DISPERSAL AND INFECTION (component level)
# -------------------------------
rain_grid <- function(df, n = 25) unique(round(c(0, quantile(df$rain[df$rain > 0],
                                                             seq(0, 0.95, length.out = n))), 2))
D_spec <- function(df) list(
  D1 = list(lo = 0, hi = 1, grid = NULL),
  D2 = list(lo = 0.001, hi = 10, grid = NULL),
  D3 = list(lo = NULL, hi = NULL, grid = list(R_th = rain_grid(df))))
I_spec <- function(df, Tl) list(
  I1 = list(lo = c(0, 0.01), hi = c(100, 2), grid = list(Rain_th = rain_grid(df))),
  I2 = list(lo = c(0, 0.01, Tl[1] + 1), hi = c(100, 2, Tl[2] - 1),
            grid = list(Rain_th = rain_grid(df))),
  I3 = list(lo = 0.5, hi = 24, grid = NULL))
n_par <- c(D1 = 1, D2 = 1, D3 = 1, I1 = 3, I2 = 4, I3 = 1)

comp_eval <- function(dz, comp) {
  df <- DAT[[dz]]; Tl <- TLIM[[dz]]
  if (comp == "D") { fun <- D_fun; spec <- D_spec(df); y <- df$ex_D }
  else { fun <- I_fun(Tl); spec <- I_spec(df, Tl); y <- df$ex_I }
  res <- mclapply(names(fun), function(a) {
    s <- spec[[a]]
    fit_eval(function(c, g) fun[[a]](c, g, df), y, df$season, s$lo, s$hi, s$grid)
  }, mc.cores = n_cores)
  names(res) <- names(fun)
  res
}

# -------------------------------
# 7. RISK LEVEL: Risk = M3 x D x I
# -------------------------------
# D-variants with I1, I-variants with D1; all D and I parameters refitted
# jointly to the example risk. Parameters: D part first, then I part.
risk_eval <- function(dz, M3) {
  df <- DAT[[dz]]; Tl <- TLIM[[dz]]; y <- df$target_risk
  Ds <- D_spec(df); Is <- I_spec(df, Tl); If <- I_fun(Tl)
  combo <- function(Dn, In) {
    d <- Ds[[Dn]]; i <- Is[[In]]
    nd <- length(d$lo); ndg <- length(d$grid)
    pred <- function(c, g) {
      cd <- c[seq_len(nd)]; ci <- c[nd + seq_along(i$lo)]
      gd <- g[seq_len(ndg)]; gi <- g[ndg + seq_along(i$grid)]
      M3 * D_fun[[Dn]](cd, gd, df) * If[[In]](ci, gi, df)
    }
    grid <- c(d$grid, i$grid)
    if (length(grid) == 2) grid <- lapply(grid, function(v) v[unique(round(seq(1, length(v), length.out = 12)))])
    fit_eval(pred, y, df$season, c(d$lo, i$lo), c(d$hi, i$hi),
             if (length(grid)) grid else NULL, starts = 3)
  }
  combos <- list(c("D1", "I1"), c("D2", "I1"), c("D3", "I1"), c("D1", "I2"), c("D1", "I3"))
  res <- mclapply(combos, function(cb) combo(cb[1], cb[2]), mc.cores = n_cores)
  names(res) <- sapply(combos, paste, collapse = "+")
  res
}

# -------------------------------
# 8. RUN
# -------------------------------
t0 <- Sys.time()
MULT <- lapply(c(fb = "fb", bs = "bs"), mult_eval)
cat("multiplication done", format(Sys.time() - t0), "\n")
DISP <- lapply(c(fb = "fb", bs = "bs"), comp_eval, comp = "D")
INF  <- lapply(c(fb = "fb", bs = "bs"), comp_eval, comp = "I")
cat("dispersal/infection done", format(Sys.time() - t0), "\n")
RISK <- lapply(c(fb = "fb", bs = "bs"), function(dz) risk_eval(dz, MULT[[dz]]$res$M3$fitted))
cat("risk level done", format(Sys.time() - t0), "\n")
save(MULT, DISP, INF, RISK, file = file.path(out_dir, "results.RData"))

# -------------------------------
# 9. TABLES
# -------------------------------
tab_M <- do.call(rbind, lapply(names(MULT), function(dz) {
  y <- DAT[[dz]]$ex_M; s <- DAT[[dz]]$season
  do.call(rbind, lapply(names(MULT[[dz]]$res), function(a) {
    r <- MULT[[dz]]$res[[a]]; gp <- if (a == "M3") MULT[[dz]]$gp_best else NA
    data.frame(disease = dz, algorithm = a, params = fmt_par(r$full), gp = gp,
               rho_fit = rho(r$fitted, y), rmse_fit = rmse(r$fitted, y), r2_fit = r2(r$fitted, y),
               rho_loso = rho(r$loso, y), rmse_loso = rmse(r$loso, y), r2_loso = r2(r$loso, y),
               saturated_pct = 100 * mean(r$fitted >= 0.99),
               decline_pct = decline_pct(r$fitted, s),
               start_shift = start_shift(dz, a, r$full$par, if (is.na(gp)) 3 else gp))
  }))
}))
tab_M <- rbind(tab_M, do.call(rbind, lapply(names(DAT), function(dz) data.frame(
  disease = dz, algorithm = "example", params = "", gp = NA, rho_fit = NA, rmse_fit = NA,
  r2_fit = NA, rho_loso = NA, rmse_loso = NA, r2_loso = NA,
  saturated_pct = 100 * mean(DAT[[dz]]$ex_M >= 0.99),
  decline_pct = decline_pct(DAT[[dz]]$ex_M, DAT[[dz]]$season), start_shift = NA))))

tab_comp <- function(L, target) do.call(rbind, lapply(names(L), function(dz) {
  y <- DAT[[dz]][[target]]
  do.call(rbind, lapply(names(L[[dz]]), function(a) {
    r <- L[[dz]][[a]]
    data.frame(disease = dz, algorithm = a, n_par = n_par[[a]], params = fmt_par(r$full),
               auc_fit = auc(r$fitted, y), brier_fit = mean((r$fitted - y)^2),
               auc_loso = auc(r$loso, y), brier_loso = mean((r$loso - y)^2),
               bic = bic(r$fitted, y, n_par[[a]]))
  }))
}))
tab_D <- tab_comp(DISP, "ex_D"); tab_I <- tab_comp(INF, "ex_I")

tab_R <- do.call(rbind, lapply(names(RISK), function(dz) {
  y <- DAT[[dz]]$target_risk
  do.call(rbind, lapply(names(RISK[[dz]]), function(cb) {
    r <- RISK[[dz]][[cb]]; k <- sum(n_par[strsplit(cb, "\\+")[[1]]])
    data.frame(disease = dz, combination = cb, n_par = k, params = fmt_par(r$full),
               rho_loso = rho(r$loso, y), rmse_loso = rmse(r$loso, y), r2_loso = r2(r$loso, y),
               rmse_fit = rmse(r$fitted, y), bic = bic(r$fitted, y, k))
  }))
}))
# Overlap of the calibrated rain-driven dispersal with the calibrated I1
tab_overlap <- do.call(rbind, lapply(names(DISP), function(dz) data.frame(
  disease = dz,
  D2_vs_I1 = rho(DISP[[dz]]$D2$fitted, INF[[dz]]$I1$fitted),
  D3_vs_I1 = rho(DISP[[dz]]$D3$fitted, INF[[dz]]$I1$fitted),
  fT_in_N_vs_I2_temp_term = NA)))

rnd <- function(t) { i <- vapply(t, is.numeric, TRUE); t[i] <- lapply(t[i], round, 3); t }
write.csv(rnd(tab_M), file.path(out_dir, "multiplication.csv"), row.names = FALSE)
write.csv(rbind(MULT$fb$gp_tab, MULT$bs$gp_tab) |> rnd(), file.path(out_dir, "m3_gp_grid.csv"), row.names = FALSE)
write.csv(rnd(tab_D), file.path(out_dir, "dispersal.csv"), row.names = FALSE)
write.csv(rnd(tab_I), file.path(out_dir, "infection.csv"), row.names = FALSE)
write.csv(rnd(tab_R), file.path(out_dir, "risk_level.csv"), row.names = FALSE)
write.csv(rnd(tab_overlap[, 1:3]), file.path(out_dir, "dispersal_infection_overlap.csv"), row.names = FALSE)
print(rnd(tab_M)); print(rnd(tab_D)); print(rnd(tab_I)); print(rnd(tab_R)); print(rnd(tab_overlap[, 1:3]))

# -------------------------------
# 10. FIGURES (calibrated, full-data fits)
# -------------------------------
for (dz in names(DAT)) {
  for (a in names(MULT[[dz]]$res)) DAT[[dz]][[a]] <- MULT[[dz]]$res[[a]]$fitted
  for (a in names(DISP[[dz]])) DAT[[dz]][[a]] <- DISP[[dz]][[a]]$fitted
  for (a in names(INF[[dz]])) DAT[[dz]][[a]] <- INF[[dz]][[a]]$fitted
}
col_ex <- "#444444"
col_c  <- c("#1b9e77", "#7570b3", "#d95f02")
labs <- list(
  Multiplication = c("Example model", "M1 Doubling time", "M2 Continuous logistic", "M3 Discrete logistic"),
  Dispersal = c("Example model", "D1 Constant K_d", "D2 Rain exponential", "D3 Rain threshold"),
  Infection = c("Example model", "I1 Wetness (rain, RH)", "I2 Wetness x f(T)", "I3 Leaf wetness duration"))
ex_title <- list(
  Multiplication = c(fb = "Maryblyt EIP / 100 (capped at 1)", bs = "log10 surface pop. / log10(5e3)"),
  Dispersal = c(fb = "Maryblyt rain wetting (0/1)", bs = "Rain > 4 mm/h, wind assumed (0/1)"),
  Infection = c(fb = "Maryblyt W and T met (0/1)", bs = "Leaf entry event (0/1)"))
example_year <- 2019

panel_lines <- function(df, k, col, main) {
  plot(NA, xlim = range(df$day), ylim = c(0, 1.05), xlab = "Day of test period",
       ylab = "Value (0-1)", main = main, cex.main = 0.95, las = 1)
  for (y in unique(df$season)) {
    s <- df[df$season == y, ]
    lines(s$day, s[[k]], col = adjustcolor(col, 0.25), lwd = 1)
  }
  mu <- tapply(df[[k]], df$day, mean); nn <- tapply(df[[k]], df$day, length)
  mu <- mu[nn >= 8]
  lines(as.numeric(names(mu)), mu, col = col, lwd = 2.6)
}
panel_heat <- function(df, k, col, main) {
  yrs <- sort(unique(df$season)); dys <- sort(unique(df$day))
  z <- matrix(NA, length(dys), length(yrs))
  z[cbind(match(df$day, dys), match(df$season, yrs))] <- df[[k]]
  image(dys, seq_along(yrs), z, zlim = c(0, 1), col = colorRampPalette(c("#f7f7f7", col))(50),
        xlab = "Day of test period", ylab = "Season", main = main, cex.main = 0.95,
        las = 1, useRaster = TRUE, yaxt = "n")
  axis(2, at = seq_along(yrs), labels = yrs, las = 1, cex.axis = 0.7)
  box()
}
single_year <- function(df, keys, lab, main, ex_key) {
  s <- df[df$season == example_year, ]
  plot(s$day, s[[ex_key]], type = "l", lwd = 2.6, col = col_ex, ylim = c(0, 1.1),
       xlab = "Day of test period", ylab = "Value (0-1)", main = main, cex.main = 1, las = 1)
  for (j in seq_along(keys)) lines(s$day, s[[keys[j]]], col = col_c[j], lwd = 2, lty = c(2, 4, 1)[j])
  legend("topleft", bty = "n", cex = 0.8, lwd = 2, ncol = 2,
         col = c(col_ex, col_c), lty = c(1, 2, 4, 1), legend = lab)
}
make_figure <- function(comp, keys, ex_key, file, heat) {
  png(file.path(out_dir, file), width = 2000, height = 1700, res = 150)
  layout(matrix(c(1, 1, 2, 2, 3:10), nrow = 3, byrow = TRUE), heights = c(1.1, 1, 1))
  par(mar = c(4, 4.2, 3, 1), oma = c(0, 0, 2, 0))
  single_year(DAT$fb, keys, labs[[comp]], sprintf("(a) %s, %d", dis_lab["fb"], example_year), ex_key)
  single_year(DAT$bs, keys, labs[[comp]], sprintf("(b) %s, %d", dis_lab["bs"], example_year), ex_key)
  for (dz in c("fb", "bs")) {
    df <- DAT[[dz]]; f <- if (heat) panel_heat else panel_lines
    f(df, ex_key, col_ex, paste0(dis_lab[dz], "\nExample: ", ex_title[[comp]][dz]))
    for (j in seq_along(keys)) f(df, keys[j], col_c[j], paste0(dis_lab[dz], "\n", labs[[comp]][j + 1]))
  }
  mtext(sprintf("%s: example models vs calibrated BGMP candidates, CF-99 hourly_99 (fire blight %d seasons, bacterial spot %d seasons)",
                comp, length(unique(DAT$fb$season)), length(unique(DAT$bs$season))),
        outer = TRUE, font = 2, cex = 1.0)
  dev.off()
}
make_figure("Multiplication", c("M1", "M2", "M3"), "ex_M", "fig_multiplication.png", FALSE)
make_figure("Dispersal", c("D1", "D2", "D3"), "ex_D", "fig_dispersal.png", TRUE)
make_figure("Infection", c("I1", "I2", "I3"), "ex_I", "fig_infection.png", TRUE)

cat("Outputs written to", out_dir, "| total", format(Sys.time() - t0), "\n")
