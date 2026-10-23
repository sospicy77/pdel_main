###############################################################################
#  Bacterial spot (Xanthomonas campestris pv. vesicatoria, Xcv) on hot pepper
#  Primary-infection-date model — R implementation
#
#  Source: Kim, J.-H., Kang, W.-S., & Yun, S.-C. (2014).
#          Plant Pathol. J. 30(2): 125-135. doi:10.5423/PPJ.OA.09.2013.0090
#
#  Model structure (Fig. 5 of the paper), hourly time step:
#   [1] Leaf-surface (epiphytic) population
#         wet hour : N(t) = N(t-1) * 2^(W / g(T))
#         dry hour : N(t) = N(t-1) * (1/2)^(1/h)
#         g(T) = 0.0088*T^2 - 0.6839*T + 15.332   (doubling time, h)
#         h    = 4.52586599                        (half-life on dry leaf, h)
#         start: 1 cell on a leaf
#   [2] Infection (surface -> inside leaf), all in the same hour:
#         N_surface >= 5e3  AND  wind > 3 m/s  AND  rain > 4 mm/h
#         N_inside(start) = N_surface / 100
#   [3] Inside-leaf population grows with temperature only:
#         N_in(t) = N_in(t-1) * 2^(1 / g(T))
#   [4] Primary infection warning (first occurrence date) when
#         N_in >= 1e15
#
#  Items NOT specified in the paper are exposed as parameters and marked
#  "ASSUMPTION" below (leaf-wetness rule, start date, T limits, caps ...).
#
#  All populations are tracked on the log10 scale to avoid overflow.
###############################################################################


# -----------------------------------------------------------------------------
# 1. Parameters
# -----------------------------------------------------------------------------
bs_default_params <- function() {
  list(
    ## --- values taken from the paper ---------------------------------------
    g_a        = 0.0088,      # g(T) = a*T^2 + b*T + c   (Fig. 2, R2 = 0.3292)
    g_b        = -0.6839,
    g_c        = 15.332,
    half_life  = 4.52586599,  # h, dry leaf surface, temperature-independent
    N0_surface = 1,           # "calculated from one bacterium at the beginning"
    infect_pop = 5e3,         # min. surface population for entry (cells/ml)
    wind_thr   = 3,           # m/s   (strictly greater than)
    rain_thr   = 4,           # mm/h  (Results/Fig.5: 4; Methods says 5)
    entry_factor = 1/100,     # population diluted 100x on entering the leaf
    symptom_thr  = 1e15,      # inside-leaf threshold -> first infection date

    ## --- ASSUMPTIONS (not given in the paper) -----------------------------
    # Leaf wetness: synoptic data have no LW sensor. Used only when the
    # weather data have no 'lw' column. An hour is wet if rain > wet_rain
    # OR RH >= wet_rh.
    wet_rh   = 90,
    wet_rain = 0,
    # "binary"  : flowchart logic — wet hour -> growth with exponent W/g,
    #             dry hour (W == 0) -> decay.
    # "fraction": growth on the wet fraction AND decay on the dry fraction
    #             of each hour: 2^(W/g) * (1/2)^((1-W)/h)
    wet_mode = "binary",
    # Growth temperature limits (paper: no growth seen at 10 C in culture,
    # but the regression is applied without limits). NA = no limit.
    T_min_growth = NA,
    T_max_growth = NA,
    # Optional cap on the epiphytic population (cells/ml). NA = no cap
    # (paper has no carrying capacity).
    surface_cap = NA,
    # Lower bound of the epiphytic population (cells/ml). Without a floor,
    # dry periods drive N to ~1e-150 and infection never happens with real
    # weather; the paper's "start from one bacterium" implies a floor of 1.
    # NA = no floor (literal equations).
    surface_floor = 1,
    # FALSE (flowchart): after the first entry the surface submodel stops.
    # TRUE: surface keeps running and later entries add to the inside pop.
    reinfect = FALSE,
    # Inside-leaf growth also requires wetness? Paper: temperature only.
    inside_needs_wet = FALSE
  )
}


# -----------------------------------------------------------------------------
# 2. Component functions
# -----------------------------------------------------------------------------

# Doubling time (h) as a function of temperature (C)
bs_doubling_time <- function(T, p = bs_default_params()) {
  p$g_a * T^2 + p$g_b * T + p$g_c
}

# Growth allowed at temperature T?
bs_growth_ok <- function(T, p) {
  ok <- !is.na(T)
  if (!is.na(p$T_min_growth)) ok <- ok & T >= p$T_min_growth
  if (!is.na(p$T_max_growth)) ok <- ok & T <= p$T_max_growth
  ok
}

# Hourly wetness rate W (0-1).
#   - if 'lw' column exists: values > 1 are treated as minutes per hour,
#     otherwise as a fraction of the hour.
#   - else: rule-based estimate from RH and rain (ASSUMPTION).
bs_wetness <- function(weather, p = bs_default_params()) {
  if ("lw" %in% names(weather)) {
    w <- weather$lw
    if (any(w > 1, na.rm = TRUE)) w <- w / 60
    return(pmin(pmax(w, 0), 1))
  }
  rh   <- if ("rh"   %in% names(weather)) weather$rh   else rep(NA, nrow(weather))
  rain <- if ("rain" %in% names(weather)) weather$rain else rep(0,  nrow(weather))
  wet  <- (!is.na(rain) & rain > p$wet_rain) | (!is.na(rh) & rh >= p$wet_rh)
  as.numeric(wet)
}

# log10 of (10^a + 10^b), numerically safe
.log10_add <- function(a, b) {
  if (is.na(a)) return(b)
  if (is.na(b)) return(a)
  m <- max(a, b)
  m + log10(10^(a - m) + 10^(b - m))
}


# -----------------------------------------------------------------------------
# 3. Main simulation (one site, one season)
# -----------------------------------------------------------------------------
#  weather: data.frame with hourly rows, columns
#     datetime (POSIXct), temp (C), rh (%), rain (mm/h), wind (m/s)
#     optional lw (fraction of hour, or minutes)
#  start, end: optional POSIXct/character to subset the season
#  Returns a list:
#     $first_infection   POSIXct of the primary infection warning (NA if none)
#     $entry_time        POSIXct when the pathogen entered the leaf
#     $doy               Julian day of the warning
#     $hourly            data.frame of hourly state
bs_run_model <- function(weather, p = bs_default_params(),
                         start = NULL, end = NULL, stop_at_warning = TRUE) {

  stopifnot(all(c("datetime", "temp", "rain", "wind") %in% names(weather)))
  weather <- weather[order(weather$datetime), ]
  if (!is.null(start)) weather <- weather[weather$datetime >= as.POSIXct(start, tz = attr(weather$datetime, "tzone")), ]
  if (!is.null(end))   weather <- weather[weather$datetime <= as.POSIXct(end,   tz = attr(weather$datetime, "tzone")), ]
  n <- nrow(weather)
  if (n == 0) stop("No weather rows in the simulation window.")

  W    <- bs_wetness(weather, p)
  T    <- weather$temp
  g    <- bs_doubling_time(T, p)
  gok  <- bs_growth_ok(T, p) & g > 0
  rain <- ifelse(is.na(weather$rain), 0, weather$rain)
  wind <- ifelse(is.na(weather$wind), 0, weather$wind)

  L2       <- log10(2)
  log_inf  <- log10(p$infect_pop)
  log_thr  <- log10(p$symptom_thr)
  log_ent  <- log10(p$entry_factor)
  log_cap  <- if (is.na(p$surface_cap)) Inf else log10(p$surface_cap)
  log_flr  <- if (is.na(p$surface_floor)) -Inf else log10(p$surface_floor)
  decay_hr <- L2 / p$half_life              # log10 loss per dry hour

  logS  <- rep(NA_real_, n)   # surface population (log10)
  logI  <- rep(NA_real_, n)   # inside-leaf population (log10)
  event <- rep("",       n)

  s <- log10(p$N0_surface)    # current surface state
  i <- NA_real_               # current inside state
  surface_active <- TRUE
  entry_idx <- NA_integer_
  warn_idx  <- NA_integer_

  for (k in seq_len(n)) {

    ## [1] surface population ------------------------------------------------
    if (surface_active) {
      grow  <- if (gok[k]) W[k] * L2 / g[k] else 0
      if (p$wet_mode == "fraction") {
        s <- s + grow - (1 - W[k]) * decay_hr
      } else {                                   # "binary" (flowchart)
        if (W[k] > 0) s <- s + grow else s <- s - decay_hr
      }
      s <- min(max(s, log_flr), log_cap)
      logS[k] <- s

      ## [2] infection condition --------------------------------------------
      if (s >= log_inf && wind[k] > p$wind_thr && rain[k] > p$rain_thr) {
        new_in <- s + log_ent
        i <- if (is.na(i)) new_in else .log10_add(i, new_in)
        event[k] <- "entry"
        if (is.na(entry_idx)) entry_idx <- k
        if (!p$reinfect) surface_active <- FALSE
        logI[k] <- i
        if (i >= log_thr && is.na(warn_idx)) { warn_idx <- k; event[k] <- "entry+warning" }
        if (!is.na(warn_idx) && stop_at_warning) break
        next                                    # growth inside starts next hour
      }
    }

    ## [3] inside-leaf population --------------------------------------------
    if (!is.na(i)) {
      can_grow <- gok[k] && (!p$inside_needs_wet || W[k] > 0)
      if (can_grow) i <- i + L2 / g[k]
      logI[k] <- i

      ## [4] threshold -> primary infection warning ---------------------------
      if (i >= log_thr && is.na(warn_idx)) {
        warn_idx <- k
        event[k] <- "warning"
        if (stop_at_warning) break
      }
    }
  }

  hourly <- data.frame(
    datetime        = weather$datetime,
    temp            = T,
    rain            = rain,
    wind            = wind,
    W               = W,
    g               = g,
    log10_surface   = logS,
    log10_inside    = logI,
    event           = event,
    stringsAsFactors = FALSE
  )

  fi <- if (is.na(warn_idx)) as.POSIXct(NA) else weather$datetime[warn_idx]
  list(
    first_infection = fi,
    entry_time      = if (is.na(entry_idx)) as.POSIXct(NA) else weather$datetime[entry_idx],
    doy             = if (is.na(warn_idx)) NA_integer_ else as.integer(format(fi, "%j")),
    period          = bs_period_label(fi),
    hourly          = hourly,
    params          = p
  )
}


# -----------------------------------------------------------------------------
# 4. Output helpers
# -----------------------------------------------------------------------------

# 10-day period label (as in the maps of Fig. 6/7: "Jul 11-20" etc.)
bs_period_label <- function(dt) {
  if (length(dt) == 0 || all(is.na(dt))) return(NA_character_)
  d   <- as.integer(format(dt, "%d"))
  mon <- format(dt, "%b")
  last <- as.integer(format(as.Date(format(dt, "%Y-%m-01")) + 31 -
                            as.integer(format(as.Date(format(dt, "%Y-%m-01")) + 31, "%d")), "%d"))
  ifelse(is.na(dt), NA_character_,
         ifelse(d <= 10, paste(mon, "1-10"),
                ifelse(d <= 20, paste(mon, "11-20"), paste0(mon, " 21-", last))))
}

# Run several sites / years. 'weather' must contain a 'site' column
# (and optionally 'year'); the season is chosen per year by month-day strings.
bs_run_sites <- function(weather, p = bs_default_params(),
                         season_start = "05-01", season_end = "09-30") {
  if (!"year" %in% names(weather)) weather$year <- as.integer(format(weather$datetime, "%Y"))
  keys <- unique(weather[, c("site", "year")])
  tz   <- attr(weather$datetime, "tzone"); if (is.null(tz)) tz <- ""
  out  <- lapply(seq_len(nrow(keys)), function(j) {
    w  <- weather[weather$site == keys$site[j] & weather$year == keys$year[j], ]
    st <- as.POSIXct(paste0(keys$year[j], "-", season_start, " 00:00"), tz = tz)
    en <- as.POSIXct(paste0(keys$year[j], "-", season_end,   " 23:00"), tz = tz)
    r  <- bs_run_model(w, p, start = st, end = en)
    data.frame(site = keys$site[j], year = keys$year[j],
               entry_time = r$entry_time, first_infection = r$first_infection,
               doy = r$doy, period = r$period, at_risk = !is.na(r$doy))
  })
  do.call(rbind, out)
}

# Validation as in the paper: Agreement = (O-O + X-X) / total
#   observed : logical, disease observed in field
#   predicted: logical, model issued a warning
bs_agreement <- function(observed, predicted) {
  tab <- table(factor(ifelse(observed,  "O", "X"), levels = c("O", "X")),
               factor(ifelse(predicted, "O", "X"), levels = c("O", "X")),
               dnn = c("observed", "predicted"))
  list(table = tab, agreement = (tab["O", "O"] + tab["X", "X"]) / sum(tab))
}

# Month difference between observed and predicted dates (Fig. 8 categories)
bs_month_diff <- function(obs_date, pred_date) {
  m <- function(x) as.integer(format(x, "%Y")) * 12 + as.integer(format(x, "%m"))
  d <- abs(m(as.Date(obs_date)) - m(as.Date(pred_date)))
  cut(d, c(-Inf, 0, 1, Inf), labels = c("same month", "1 month", ">=2 months"))
}


# -----------------------------------------------------------------------------
# 5. Plot of one run
# -----------------------------------------------------------------------------
bs_plot <- function(res, main = "Xcv population dynamics") {
  h <- res$hourly
  p <- res$params
  yl <- range(c(h$log10_surface, h$log10_inside, log10(p$symptom_thr)), na.rm = TRUE)
  plot(h$datetime, h$log10_surface, type = "l", col = "steelblue",
       ylim = c(min(-1, yl[1]), yl[2] + 1), xlab = "", ylab = "log10 population (cells/ml)",
       main = main)
  lines(h$datetime, h$log10_inside, col = "firebrick", lwd = 2)
  abline(h = log10(p$infect_pop),  lty = 2, col = "steelblue")
  abline(h = log10(p$symptom_thr), lty = 2, col = "firebrick")
  if (!is.na(res$entry_time))      abline(v = res$entry_time, col = "grey40", lty = 3)
  if (!is.na(res$first_infection)) abline(v = res$first_infection, col = "firebrick", lty = 3)
  legend("topleft", bty = "n", cex = 0.8,
         legend = c("leaf surface", "inside leaf", "entry threshold 5e3", "warning threshold 1e15"),
         col = c("steelblue", "firebrick", "steelblue", "firebrick"),
         lty = c(1, 1, 2, 2), lwd = c(1, 2, 1, 1))
}


###############################################################################
# 6. Demo with synthetic hourly weather (replace with KMA ASOS/AWS data)
###############################################################################
if (sys.nframe() == 0) {

  set.seed(2014)
  tz  <- "Asia/Seoul"
  dt  <- seq(as.POSIXct("2011-05-01 00:00", tz = tz),
             as.POSIXct("2011-09-30 23:00", tz = tz), by = "hour")
  n   <- length(dt)
  doy <- as.integer(format(dt, "%j"))
  hr  <- as.integer(format(dt, "%H"))

  # seasonal + diurnal temperature
  temp <- 21 + 6 * sin((doy - 130) / 365 * 2 * pi) + 4 * sin((hr - 9) / 24 * 2 * pi) + rnorm(n, 0, 1)

  # rain spells (2-state Markov chain): longer/more frequent in Jul-Aug
  monsoon <- doy >= 175 & doy <= 240
  p_start <- ifelse(monsoon, 0.012, 0.006)   # dry -> rain
  p_stay  <- ifelse(monsoon, 0.93,  0.85)    # rain -> rain
  state   <- logical(n)
  for (k in 2:n) state[k] <- if (state[k-1]) runif(1) < p_stay[k] else runif(1) < p_start[k]
  rain    <- ifelse(state, rgamma(n, shape = 0.9, scale = ifelse(monsoon, 5, 2.5)), 0)
  rh     <- pmin(100, ifelse(rain > 0, runif(n, 92, 100),
                             ifelse(monsoon, 84, 76) - 12 * sin((hr - 9) / 24 * 2 * pi) + rnorm(n, 0, 5)))
  wind   <- pmax(0, rgamma(n, shape = 2, scale = 1) + ifelse(rain > 0, 1.5, 0))

  wx <- data.frame(datetime = dt, temp = temp, rh = rh, rain = rain, wind = wind)

  res <- bs_run_model(wx)
  cat("Entry into leaf      :", format(res$entry_time), "\n")
  cat("Primary infection    :", format(res$first_infection),
      "| DOY", res$doy, "|", res$period, "\n")

  # sensitivity: rain threshold 4 vs 5 mm/h, binary vs fraction wetness
  for (rt in c(4, 5)) for (wm in c("binary", "fraction")) {
    pp <- bs_default_params(); pp$rain_thr <- rt; pp$wet_mode <- wm
    r  <- bs_run_model(wx, pp)
    cat(sprintf("rain_thr=%d, wet_mode=%-8s -> %s\n", rt, wm,
                ifelse(is.na(r$doy), "no warning", paste("DOY", r$doy, r$period))))
  }

  # sanity check: hours needed inside the leaf at constant 25 C
  g25 <- bs_doubling_time(25)
  cat(sprintf("g(25C) = %.2f h; from 50 cells to 1e15 takes %.0f h (%.1f d)\n",
              g25, log2(1e15 / 50) * g25, log2(1e15 / 50) * g25 / 24))

  # validation check reproducing the paper's 2x2 table (21, 60, 4, 9)
  obs  <- c(rep(TRUE, 21), rep(TRUE, 60), rep(FALSE, 4), rep(FALSE, 9))
  pred <- c(rep(TRUE, 21), rep(FALSE, 60), rep(TRUE, 4), rep(FALSE, 9))
  v <- bs_agreement(obs, pred)
  print(v$table); cat(sprintf("Agreement = %.1f%%\n", 100 * v$agreement))

  png("bacterial_spot_demo.png", width = 1100, height = 550, res = 120)
  bs_plot(res, main = "Bacterial spot model - synthetic 2011 weather")
  dev.off()
}
