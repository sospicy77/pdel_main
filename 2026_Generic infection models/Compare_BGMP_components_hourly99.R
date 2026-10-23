#############################################
# BGMP basic-module algorithm selection test (hourly_99, 2010-2025)
#
# Example models : pear fire blight (Maryblyt type, pear_fire_blight_fixed.py)
#                  pepper bacterial spot (bacterial_spot_model.R)
# BGMP           : bacterialmodel_v8.R (fire_blight, bacterial_spot parameters)
#
# Test periods   : fire blight    - bloom, B1 <= day < PF (pear model)
#                  bacterial spot - 1 Jun - 31 Aug
#
# Candidates
#   Multiplication : M1 doubling time (exponential from N0, capped at 1)
#                    M2 continuous logistic (tracked from N0 at season start)
#                    M3 discrete logistic (v8: restart from N0, last gp days)
#   Dispersal      : D1 constant K_d (v8)
#                    D2 rain exponential 1 - exp(-a R) (v7 add-on)
#                    D3 rain threshold 1[R >= R_th]
#   Infection      : I1 wetness W(rain, RH) (v8)
#                    I2 wetness x f_T(T) (v7 add-on)
#                    I3 leaf-wetness duration min(1, LWD / 6 h) (hourly)
#
# hourly_99 has no wind speed: the bacterial spot wind condition
# (> 3 m/s) is assumed to be met (ASSUMPTION).
#############################################

# Run from the "2026_Generic infection models" folder after the pear
# model runs for hourly_99 (Output/pear_99/pear_daily_<year>.csv).

invisible(Sys.setlocale("LC_CTYPE", "C.UTF-8"))   # model files contain UTF-8 comments
seasons <- 2010:2025
out_dir <- file.path("Output", "component_test_99")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# -------------------------------
# 1. MODELS
# -------------------------------
v8 <- new.env()
ex <- parse("bacterialmodel_v8.R", encoding = "UTF-8")
keep <- vapply(ex, function(e) is.call(e) && identical(e[[1]], as.name("<-")) &&
                 (is.call(e[[3]]) && identical(e[[3]][[1]], as.name("function")) ||
                  as.character(e[[2]]) %in% c("disease_params", "global_params")),
               logical(1))
for (e in ex[keep]) eval(e, v8)
gp <- v8$global_params$growth_period

source("bacterial_spot_model.R", encoding = "UTF-8")

# -------------------------------
# 2. DATA
# -------------------------------
hr <- read.csv(file.path("obs", "hourly_99"))
hr[hr == -999] <- NA
hr$datetime <- as.POSIXct(hr$data_tm, format = "%Y-%m-%d %H:%M", tz = "Asia/Seoul")
hr$date <- as.Date(format(hr$datetime, "%Y-%m-%d"))
hr$rn[is.na(hr$rn)] <- 0
for (v in c("ta", "hm")) {                       # short gaps: linear fill
  ok <- !is.na(hr[[v]])
  hr[[v]] <- approx(which(ok), hr[[v]][ok], seq_len(nrow(hr)), rule = 2)$y
}
day <- aggregate(cbind(temp = ta, rh = hm) ~ date, hr, mean)
day$rain <- aggregate(rn ~ date, hr, sum)$rn
lw <- aggregate(cbind(lwd = as.numeric(hm >= 90 | rn > 0)) ~ date, hr, sum)
day$lwd <- lw$lwd[match(day$date, lw$date)]

# -------------------------------
# 3. CANDIDATE ALGORITHMS (daily)
# -------------------------------
growth_rate <- function(d, p) {
  p$mu_max * v8$temp_response(d$temp, p$T_min, p$T_opt, p$T_max) *
    v8$rh_response(d$rh, p$rh_thresh_M)
}
mult_doubling <- function(r, N0) {               # Td = ln2 / r (days)
  N <- numeric(length(r)); x <- N0
  for (t in seq_along(r)) { x <- min(1, x * 2^(r[t] / log(2))); N[t] <- x }
  N
}
mult_logistic_cont <- function(r, N0) {
  N <- numeric(length(r)); x <- N0
  for (t in seq_along(r)) { x <- min(1, x + r[t] * x * (1 - x)); N[t] <- x }
  N
}
mult_logistic_disc <- function(r, N0, gp) {      # v8 calc_population
  N <- numeric(length(r))
  for (t in seq_along(r)) {
    x <- N0
    for (s in max(1, t - gp):t) x <- min(1, max(0, x + r[s] * x * (1 - x)))
    N[t] <- x
  }
  N
}

run_candidates <- function(d, p, start_idx, rain_a, rain_th_D) {
  r <- growth_rate(d, p)
  idx <- start_idx:nrow(d)
  data.frame(
    date = d$date[idx],
    M1 = mult_doubling(r[idx], p$N0),
    M2 = mult_logistic_cont(r[idx], p$N0),
    M3 = mult_logistic_disc(r, p$N0, gp)[idx],
    D1 = p$K_d,
    D2 = 1 - exp(-rain_a * d$rain[idx]),
    D3 = as.numeric(d$rain[idx] >= rain_th_D),
    I1 = v8$wetness_response(d$rh[idx], d$rain[idx], p$rh_thresh_I, p$rain_thresh),
    I2 = v8$wetness_response(d$rh[idx], d$rain[idx], p$rh_thresh_I, p$rain_thresh) *
         v8$temp_response(d$temp[idx], p$T_min, p$T_opt, p$T_max),
    I3 = pmin(1, d$lwd[idx] / 6)
  )
}

# -------------------------------
# 4. FIRE BLIGHT (bloom period)
# -------------------------------
pfb <- v8$disease_params$fire_blight
fb <- do.call(rbind, lapply(seasons, function(y) {
  x <- read.csv(file.path("Output", "pear_99", sprintf("pear_daily_%d.csv", y)))
  x$date <- as.Date(x$tm)
  x$B <- x$B == "True"; x$W <- x$W == "True"; x$T <- x$T == "True"
  b <- which(x$B)
  d <- day[day$date >= min(x$date[b]) - gp & day$date <= max(x$date[b]), ]
  cand <- run_candidates(d, pfb, start_idx = gp + 1, rain_a = 0.055, rain_th_D = 1)
  e <- x[b, ]
  rn_prev <- c(0, x$rn)[b]
  data.frame(season = y, day = seq_along(b) - 1, cand,
             ex_M = pmin(1, e$EIP / 100), EIP = e$EIP,
             ex_D = as.numeric(e$rn >= 0.25 | rn_prev >= 2.5),
             ex_I = as.numeric(e$W & e$T), BIR = e$BIR)
}))

# -------------------------------
# 5. BACTERIAL SPOT (1 Jun - 31 Aug)
# -------------------------------
pbs <- v8$disease_params$bacterial_spot
bsp <- bs_default_params(); bsp$reinfect <- TRUE
# hourly_99 has no Jul-Aug data in 2012, 2015, 2018, 2021, 2024 and ends on
# 29 Jun 2025: only seasons with a complete 1 Jun - 31 Aug record are used.
bs_ok <- function(y) sum(hr$date >= as.Date(sprintf("%d-06-01", y)) &
                         hr$date <= as.Date(sprintf("%d-08-31", y))) >= 0.95 * 92 * 24
seasons_bs <- Filter(bs_ok, seasons)
bs <- do.call(rbind, lapply(seasons_bs, function(y) {
  st <- as.Date(sprintf("%d-06-01", y)); en <- as.Date(sprintf("%d-08-31", y))
  w <- hr[hr$date >= st & hr$date <= en, ]
  wx <- data.frame(datetime = w$datetime, temp = w$ta, rh = w$hm, rain = w$rn,
                   wind = 5)                     # ASSUMPTION: wind > 3 m/s
  res <- bs_run_model(wx, bsp, stop_at_warning = FALSE)$hourly
  res$date <- as.Date(format(res$datetime, "%Y-%m-%d"))
  logS <- aggregate(log10_surface ~ date, res, max)
  disp <- aggregate(cbind(x = rain > bsp$rain_thr) ~ date, res, max)
  ent  <- aggregate(cbind(x = event != "") ~ date, res, max)
  d <- day[day$date >= st - gp & day$date <= en, ]
  cand <- run_candidates(d, pbs, start_idx = gp + 1, rain_a = 0.070, rain_th_D = 4)
  data.frame(season = y, day = seq_len(nrow(cand)) - 1, cand,
             ex_M = pmin(1, logS$log10_surface / log10(bsp$infect_pop)),
             logS = logS$log10_surface,
             ex_D = disp$x, ex_I = ent$x)
}))

write.csv(fb, file.path(out_dir, "fire_blight_daily.csv"), row.names = FALSE)
write.csv(bs, file.path(out_dir, "bacterial_spot_daily.csv"), row.names = FALSE)

# -------------------------------
# 6. METRICS
# -------------------------------
auc <- function(score, obs) {                    # Mann-Whitney AUC
  obs <- obs > 0
  if (all(obs) || !any(obs)) return(NA)
  rk <- rank(score)
  (sum(rk[obs]) - sum(obs) * (sum(obs) + 1) / 2) / (sum(obs) * sum(!obs))
}
within_sd <- function(x, s) mean(tapply(x, s, sd))
decline_pct <- function(x, s) 100 * mean(unlist(tapply(x, s, function(z) diff(z) < -1e-6)))
metrics <- function(df, disease) {
  m <- do.call(rbind, lapply(c("M1", "M2", "M3"), function(k) data.frame(
    disease, component = "Multiplication", algorithm = k,
    spearman = cor(df[[k]], df$ex_M, method = "spearman"),
    auc = NA, within_season_sd = within_sd(df[[k]], df$season),
    saturated_pct = 100 * mean(df[[k]] >= 0.99),
    decline_pct = decline_pct(df[[k]], df$season))))
  d <- do.call(rbind, lapply(c("D1", "D2", "D3"), function(k) data.frame(
    disease, component = "Dispersal", algorithm = k,
    spearman = suppressWarnings(cor(df[[k]], df$ex_D, method = "spearman")),
    auc = auc(df[[k]], df$ex_D), within_season_sd = within_sd(df[[k]], df$season),
    saturated_pct = NA, decline_pct = NA)))
  i <- do.call(rbind, lapply(c("I1", "I2", "I3"), function(k) data.frame(
    disease, component = "Infection", algorithm = k,
    spearman = cor(df[[k]], df$ex_I, method = "spearman"),
    auc = auc(df[[k]], df$ex_I), within_season_sd = within_sd(df[[k]], df$season),
    saturated_pct = NA, decline_pct = NA)))
  ex <- data.frame(disease, component = c("Multiplication", "Dispersal", "Infection"),
                   algorithm = "example", spearman = 1, auc = NA,
                   within_season_sd = c(within_sd(df$ex_M, df$season),
                                        within_sd(df$ex_D, df$season),
                                        within_sd(df$ex_I, df$season)),
                   saturated_pct = c(100 * mean(df$ex_M >= 0.99), NA, NA),
                   decline_pct = c(decline_pct(df$ex_M, df$season), NA, NA))
  rbind(ex, m, d, i)
}
met <- rbind(metrics(fb, "fire_blight"), metrics(bs, "bacterial_spot"))

# Risk-level check: does the dispersal choice change the BGMP risk ranking?
risk_cor <- function(df, ref) sapply(c("D1", "D2", "D3"), function(k)
  cor(df$M3 * df[[k]] * df$I1, ref, method = "spearman"))
rc <- rbind(fire_blight = risk_cor(fb, fb$BIR), bacterial_spot = risk_cor(bs, bs$ex_I))
# Overlap of rain-driven dispersal with the rain term of infection (I1)
ov <- rbind(fire_blight = sapply(c("D2", "D3"), function(k) cor(fb[[k]], fb$I1, method = "spearman")),
            bacterial_spot = sapply(c("D2", "D3"), function(k) cor(bs[[k]], bs$I1, method = "spearman")))

# Same check for the infection candidates (Risk = M3 x D1 x I_k)
risk_cor_I <- function(df, ref) sapply(c("I1", "I2", "I3"), function(k)
  cor(df$M3 * df$D1 * df[[k]], ref, method = "spearman"))
rcI <- rbind(fire_blight = risk_cor_I(fb, fb$BIR), bacterial_spot = risk_cor_I(bs, bs$ex_I))

# Start-date sensitivity of the multiplication candidates: season start
# moved 10 days earlier; mean |N(start) - N(start - 10 d)| over the test period
start_shift <- function(d_all, p, first, last, shift = 10) {
  base <- day[day$date >= first - gp & day$date <= last, ]
  early <- day[day$date >= first - shift - gp & day$date <= last, ]
  a <- run_candidates(base, p, gp + 1, 0, 1)
  b <- run_candidates(early, p, gp + 1, 0, 1)
  b <- b[match(a$date, b$date), ]
  sapply(c("M1", "M2", "M3"), function(k) mean(abs(a[[k]] - b[[k]])))
}
ss_fb <- rowMeans(sapply(seasons, function(y) {
  w <- fb$season == y
  start_shift(day, pfb, min(fb$date[w]), max(fb$date[w]))
}))
ss_bs <- rowMeans(sapply(seasons_bs, function(y)
  start_shift(day, pbs, as.Date(sprintf("%d-06-01", y)), as.Date(sprintf("%d-08-31", y)))))
ss <- rbind(fire_blight = ss_fb, bacterial_spot = ss_bs)
write.csv(round(rcI, 3), file.path(out_dir, "risk_spearman_by_infection.csv"))
write.csv(round(ss, 3), file.path(out_dir, "multiplication_start_shift.csv"))
print(round(rcI, 3)); print(round(ss, 3))

met[, 4:8] <- round(met[, 4:8], 3)
write.csv(met, file.path(out_dir, "metrics.csv"), row.names = FALSE)
write.csv(round(rc, 3), file.path(out_dir, "risk_spearman_by_dispersal.csv"))
write.csv(round(ov, 3), file.path(out_dir, "dispersal_infection_overlap.csv"))
print(met); print(round(rc, 3)); print(round(ov, 3))

# -------------------------------
# 7. FIGURES
# -------------------------------
col_ex <- "#444444"
col_c  <- c("#1b9e77", "#7570b3", "#d95f02")
lab_M <- c(ex = "Example model", M1 = "M1 Doubling time", M2 = "M2 Continuous logistic",
           M3 = "M3 Discrete logistic (v8)")
lab_D <- c(ex = "Example model", D1 = "D1 Constant K_d (v8)", D2 = "D2 Rain exponential",
           D3 = "D3 Rain threshold")
lab_I <- c(ex = "Example model", I1 = "I1 Wetness (v8)", I2 = "I2 Wetness x f(T)",
           I3 = "I3 Leaf wetness duration")
ex_title <- list(
  Multiplication = c(fb = "Maryblyt EIP / 100 (capped at 1)",
                     bs = "log10 surface pop. / log10(5e3)"),
  Dispersal = c(fb = "Maryblyt rain wetting (0/1)", bs = "Rain > 4 mm/h, wind assumed (0/1)"),
  Infection = c(fb = "Maryblyt W and T met (0/1)", bs = "Leaf entry event (0/1)"))
dis_lab <- c(fb = "Pear fire blight (bloom)", bs = "Pepper bacterial spot (Jun-Aug)")
example_year <- 2019

panel_lines <- function(df, k, col, main) {
  plot(NA, xlim = range(df$day), ylim = c(0, 1.05), xlab = "Day of test period",
       ylab = "Value (0-1)", main = main, cex.main = 0.95, las = 1)
  for (y in unique(df$season)) {
    s <- df[df$season == y, ]
    lines(s$day, s[[k]], col = adjustcolor(col, 0.25), lwd = 1)
  }
  mu <- tapply(df[[k]], df$day, mean)
  nn <- tapply(df[[k]], df$day, length)
  mu <- mu[nn >= 8]                              # mean over >= 8 seasons only
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
single_year <- function(df, keys, labs, main, ex_key = "ex_M") {
  s <- df[df$season == example_year, ]
  plot(s$day, s[[ex_key]], type = "l", lwd = 2.6, col = col_ex, ylim = c(0, 1.1),
       xlab = "Day of test period", ylab = "Value (0-1)", main = main, cex.main = 1, las = 1)
  for (j in seq_along(keys)) lines(s$day, s[[keys[j]]], col = col_c[j], lwd = 2,
                                   lty = c(2, 4, 1)[j])
  legend("topleft", bty = "n", cex = 0.8, lwd = 2, ncol = 2,
         col = c(col_ex, col_c), lty = c(1, 2, 4, 1), legend = labs)
}

make_figure <- function(comp, keys, labs, ex_key, file, heat) {
  png(file.path(out_dir, file), width = 2000, height = 1700, res = 150)
  layout(matrix(c(1, 1, 2, 2, 3:10), nrow = 3, byrow = TRUE), heights = c(1.1, 1, 1))
  par(mar = c(4, 4.2, 3, 1), oma = c(0, 0, 2, 0))
  single_year(fb, keys, labs, sprintf("(a) %s, %d", dis_lab["fb"], example_year), ex_key)
  single_year(bs, keys, labs, sprintf("(b) %s, %d", dis_lab["bs"], example_year), ex_key)
  for (dz in c("fb", "bs")) {
    df <- if (dz == "fb") fb else bs
    f <- if (heat) panel_heat else panel_lines
    f(df, ex_key, col_ex, paste0(dis_lab[dz], "\nExample: ", ex_title[[comp]][dz]))
    for (j in seq_along(keys)) f(df, keys[j], col_c[j], paste0(dis_lab[dz], "\n", labs[j + 1]))
  }
  mtext(sprintf("%s: example models vs BGMP candidates, CF-99 hourly_99 (fire blight %d seasons, bacterial spot %d seasons)",
                comp, length(seasons), length(seasons_bs)), outer = TRUE, font = 2, cex = 1.05)
  dev.off()
}
make_figure("Multiplication", c("M1", "M2", "M3"), lab_M, "ex_M", "fig_multiplication.png", FALSE)
make_figure("Dispersal", c("D1", "D2", "D3"), lab_D, "ex_D", "fig_dispersal.png", TRUE)
make_figure("Infection", c("I1", "I2", "I3"), lab_I, "ex_I", "fig_infection.png", TRUE)

cat("Outputs written to", out_dir, "\n")
