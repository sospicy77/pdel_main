###############################################################################
# Standalone Simple Generic Infection Model
#
# Based on:
#   - Run.simpleinfection(jjy).R
#   - Required functions extracted from Functions_simpleinfection(jjy).R
#
# Modification applied:
#   - Only the handling of unfavorable temperature was changed.
#   - When ft[i] == 0, the previously accumulated infection progress is retained.
#   - Accumulated progress is reset only when wethrs[i] == 0.
#
# All other model logic, including the splash logic, is retained.
###############################################################################

rm(list = ls())

###############################################################################
# Required class and utility functions from Functions_simpleinfection(jjy).R
###############################################################################

setClass(
  "weather",
  representation(
    stn = "character",
    rmk = "character",
    lon = "numeric",
    lat = "numeric",
    alt = "numeric",
    vars = "data.frame",
    w = "data.frame"
  ),
  prototype(
    stn = "Station name",
    rmk = paste(Sys.time()),
    lon = 0,
    lat = 0,
    alt = 0,
    vars = data.frame(
      varcode = character(0),
      varname = character(0),
      unit = character(0),
      stringsAsFactors = FALSE
    ),
    w = data.frame()
  )
)


doyFromDate <- function(date) {
  date <- as.character(date)
  as.numeric(format(as.Date(date), "%j"))
}


SVP <- function(temp) {
  .611 * 10^(7.5 * temp / (237.7 + temp))
}


rhMinMax <- function(rhavg, tmin, tmax, tavg = (tmin + tmax) / 2) {
  tmin <- pmax(tmin, -5)
  tmax <- pmax(tmax, -5)
  tavg <- pmax(tavg, -5)

  es <- SVP(tavg)
  vp <- rhavg / 100 * es

  es <- SVP(tmax)
  rhmin <- 100 * vp / es
  rhmin <- pmax(0, pmin(100, rhmin))

  es <- SVP(tmin)
  rhmax <- 100 * vp / es
  rhmax <- pmax(0, pmin(100, rhmax))

  cbind(rhmin, rhmax)
}


daylength <- function(lat, doy) {
  if (class(doy) == "Date" | class(doy) == "character") {
    doy <- doyFromDate(doy)
  }

  lat[lat > 90 | lat < -90] <- NA

  doy[doy == 366] <- 365
  doy[doy < 1] <- 365 + doy[doy < 1]
  doy[doy > 365] <- doy[doy > 365] - 365

  if (isTRUE(any(doy < 1)) | isTRUE(any(doy > 365))) {
    stop("cannot understand value for doy")
  }

  P <- asin(
    0.39795 * cos(
      0.2163108 +
        2 * atan(0.9671396 * tan(0.00860 * (doy - 186)))
    )
  )

  a <- (
    sin(0.8333 * pi / 180) +
      sin(lat * pi / 180) * sin(P)
  ) / (
    cos(lat * pi / 180) * cos(P)
  )

  a <- pmin(pmax(a, -1), 1)
  24 - (24 / pi) * acos(a)
}


diurnalTemp <- function(lat, date, tmin, tmax) {
  TC <- 4.0
  P <- 1.5

  dayl <- daylength(lat, doyFromDate(date))
  nigthl <- 24 - dayl
  sunris <- 12 - 0.5 * dayl
  sunset <- 12 + 0.5 * dayl

  hrtemp <- vector(length = 24)

  for (hr in 1:24) {
    if (hr < sunris) {
      tsunst <- tmin + (tmax - tmin) * sin(pi * (dayl / (dayl + 2 * P)))
      hrtemp[hr] <- (
        tmin - tsunst * exp(-nigthl / TC) +
          (tsunst - tmin) * exp(-(hr + 24 - sunset) / TC)
      ) / (1 - exp(-nigthl / TC))
    } else if (hr < (12 + P)) {
      hrtemp[hr] <- tmin +
        (tmax - tmin) * sin(pi * (hr - sunris) / (dayl + 2 * P))
    } else if (hr < sunset) {
      hrtemp[hr] <- tmin +
        (tmax - tmin) * sin(pi * (hr - sunris) / (dayl + 2 * P))
    } else {
      tsunst <- tmin + (tmax - tmin) * sin(pi * (dayl / (dayl + 2 * P)))
      hrtemp[hr] <- (
        tmin - tsunst * exp(-nigthl / TC) +
          (tsunst - tmin) * exp(-(hr - sunset) / TC)
      ) / (1 - exp(-nigthl / TC))
    }
  }

  hrtemp
}


diurnalRH.temp <- function(lat, date, rhavg, tmin, tmax,
                           tavg = (tmin + tmax) / 2) {
  temp <- diurnalTemp(lat, date, tmin, tmax)
  vp <- SVP(tavg) * rhavg / 100

  hr <- 1:24
  es <- SVP(temp[hr])
  rh <- 100 * vp / es
  rh <- pmin(100, pmax(0, rh))

  data.frame(rh, temp)
}


hourly.weather <- function(wth, simple = FALSE) {
  days <- length(wth@w$date)
  hw <- data.frame()

  wth@w$rhavg <- (wth@w$rhmin + wth@w$rhmax) / 2

  for (d in 1:days) {
    rhtemp <- diurnalRH.temp(
      wth@lat,
      wth@w$date[d],
      wth@w$rhavg[d],
      wth@w$tmin[d],
      wth@w$tmax[d],
      wth@w$tavg[d]
    )

    rh <- rhtemp[, 1]
    lw <- rh

    if (simple) {
      lw[] <- 0
      lw[rh >= 90] <- 1
    } else {
      x <- (rh - 80) / (95 - 80)
      lw[rh >= 95] <- 1
      lw[rh < 95] <- x[rh < 95]
      lw[rh < 80] <- 0
    }

    rhtemplw <- data.frame(rhtemp, lw)
    hw <- rbind(
      hw,
      cbind(date = rep(wth@w$date[d], 24), hr = 1:24, rhtemplw)
    )
  }

  hw
}


.subsetwth <- function(wth, emergence, duration) {
  emergence <- as.Date(emergence)
  wth@w <- subset(wth@w, wth@w$date >= emergence)

  if (dim(wth@w)[1] < duration) {
    duration <- dim(wth@w)[1]
  }

  wth@w <- wth@w[1:duration, ]

  if (sum(is.na(wth@w)) > 0) {
    stop("There are missing values (NA) in your weather data")
  }

  wth
}


Get.wth <- function(wdir, stnid, stnlon, stnlat) {
  srchstr <- paste0("*", stnid, "*.csv")
  
  obsfile <- list.files(
    path = wdir,
    pattern = glob2rx(srchstr),
    full.names = TRUE
  )
  
  if (length(obsfile) == 0) {
    stop("Weather file not found for station: ", stnid)
  }
  
  if (length(obsfile) > 1) {
    stop(
      "Multiple weather files found for station ",
      stnid,
      ": ",
      paste(basename(obsfile), collapse = ", ")
    )
  }
  
  obs <- read.csv(
    obsfile,
    sep = ",",
    header = TRUE,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  
  # ID101 daily weather structure
  obs <- obs[, 2:9]
  
  colnames(obs) <- c("year","mon","day","prec","tmax",
    "tmin", "wspd", "rhum")
  
  # Convert -99 missing-value codes to NA
  weather_vars <- c("prec", "tmax", "tmin", "wspd",
    "rhum")
  
  obs[weather_vars] <- lapply(
    obs[weather_vars],
    function(x) {
      x[x <= -90] <- NA
      x
    }
  )
  
  sdate <- as.Date(
    sprintf(
      "%04d-%02d-%02d",
      obs$year,
      obs$mon,
      obs$day
    )
  )
  
  obs$tavg <- (
    obs$tmax +
      obs$tmin
  ) / 2
  
  doy <- doyFromDate(sdate)
  
  lns <- data.frame(
    date = sdate,
    year = obs$year,
    doy = doy,
    tavg = obs$tavg,
    tmax = obs$tmax,
    tmin = obs$tmin,
    rhum = obs$rhum,
    prec = obs$prec
  )
  
  if (anyNA(lns)) {
    missing_rows <- which(
      !complete.cases(lns)
    )
    
    stop(
      "Missing or invalid weather data found in rows: ",
      paste(head(missing_rows, 20), collapse = ", "),
      if (length(missing_rows) > 20) " ..." else ""
    )
  }
  
  rhnx <- rhMinMax(
    lns$rhum,
    lns$tmin,
    lns$tmax,
    lns$tavg
  )
  
  lns <- cbind(
    lns,
    rhnx
  )
  
  colnames(lns) <- c("date","year","doy","tavg","tmax",
    "tmin","rhum","prec","rhmin","rhmax")
  
  wth <- new("weather")
  wth@lon <- stnlon
  wth@lat <- stnlat
  wth@w <- lns
  
  return(wth)
}


Get.daily.wth.korea <- function(wdir, stnid, stnlon, stnlat) {
  wdir <- stndir
  
  setwd(wdir)

  srchstr <- paste("*", stnid, "*.csv", sep = "")
  obsfile <- list.files(
    wdir,
    pattern = glob2rx(srchstr),
    full.names = FALSE
  )

  obs <- read.csv(
    file = obsfile,
    sep = ",",
    header = TRUE,
    stringsAsFactors = FALSE,
    fileEncoding = "CP949",
    check.names = FALSE
  )

  obs <- obs[, c(2, 3, 6, 9)]
  colnames(obs) <- c("Date", "tmp", "prec", "rhum")

  sdate <- substring(obs$Date, 1, 10)
  sdate <- as.Date(unique(sdate))
  doy <- doyFromDate(sdate)

  days <- length(sdate)
  obsdata <- data.frame()

  for (i in 1:days) {
    imsirows <- ((i * 24) - 23):(i * 24)
    imsi <- obs[imsirows, ]

    tmax <- max(imsi$tmp)
    tmin <- min(imsi$tmp)
    tavg <- mean(imsi$tmp)
    rhum <- mean(imsi$rhum)
    rhmin <- min(imsi$rhum)
    rhmax <- max(imsi$rhum)
    prec <- sum(imsi$prec)

    obsdata <- rbind(
      obsdata,
      c(tavg, tmax, tmin, rhum, prec, rhmin, rhmax)
    )
  }

  lns <- data.frame(sdate, doy, obsdata)
  colnames(lns) <- c(
    "date", "doy", "tavg", "tmax", "tmin",
    "rhum", "prec", "rhmin", "rhmax"
  )

  wth <- new("weather")
  wth@lon <- stnlon
  wth@lat <- stnlat
  wth@w <- lns

  wth
}


Get.hourly.wth.korea <- function(wdir, stnid, emergence, duration) {
  
  srchstr <- paste0("*", stnid, "*.csv")
  
  obsfile <- list.files(
    path = wdir,
    pattern = glob2rx(srchstr),
    full.names = TRUE
  )
  
  if (length(obsfile) == 0) {
    stop("Weather file not found for station: ", stnid)
  }
  
  if (length(obsfile) > 1) {
    stop(
      "Multiple weather files found for station ", stnid, ": ",
      paste(basename(obsfile), collapse = ", ")
    )
  }
  
  obs <- read.csv(
    file = obsfile,
    sep = ",",
    header = TRUE,
    stringsAsFactors = FALSE,
    fileEncoding = "CP949",
    check.names = FALSE
  )
  
  # ID1001.csv의 실제 열 구조:
  # 2 = 일시, 3 = 기온, 6 = 강수량, 9 = 습도
  obs <- obs[, c(2, 3, 6, 9)]
  
  colnames(obs) <- c(
    "Datetime",
    "tmp",
    "prec",
    "rhum"
  )
  
  # 날짜·시간 형식 변환
  obs$Datetime <- as.POSIXct(
    obs$Datetime,
    format = "%Y-%m-%d %H:%M",
    tz = "Asia/Seoul"
  )
  
  if (any(is.na(obs$Datetime))) {
    stop(
      "Some datetime values could not be parsed. ",
      "Check the datetime format in the weather file."
    )
  }
  
  # 강수량 공란은 무강수로 처리
  obs$prec[is.na(obs$prec)] <- 0
  
  # 시간순 정렬
  obs <- obs[order(obs$Datetime), ]
  
  emergence <- as.Date(emergence)
  
  obs <- obs[
    as.Date(obs$Datetime) >= emergence,
  ]
  
  required_hours <- duration * 24
  
  if (nrow(obs) < required_hours) {
    warning(
      "Only ",
      nrow(obs),
      " hourly records are available. Requested: ",
      required_hours
    )
    
    duration <- floor(nrow(obs) / 24)
    required_hours <- duration * 24
  }
  
  if (required_hours == 0) {
    stop("No complete daily hourly records are available.")
  }
  
  obs <- obs[seq_len(required_hours), ]
  
  date <- as.Date(obs$Datetime)
  hr <- as.integer(format(obs$Datetime, "%H")) + 1
  
  # 기존 R 코드의 leaf wetness 계산 유지
  lw <- numeric(nrow(obs))
  
  x <- (obs$rhum - 80) / (95 - 80)
  
  lw[obs$rhum >= 95] <- 1
  lw[obs$rhum < 95] <- x[obs$rhum < 95]
  lw[obs$rhum < 80] <- 0
  
  lw[lw >= 0.8] <- 1
  lw[lw < 0.8] <- 0
  
  hw <- data.frame(
    date = date,
    hr = hr,
    pr = obs$prec,
    rh = obs$rhum,
    tp = obs$tmp,
    lw = lw
  )
  
  return(hw)
}
###############################################################################
# User settings
###############################################################################

prjdir <- getwd()
stnfile <- "Station-Info.csv"
stnid <- "ID1001"
wthhourly <- TRUE

# Generic infection model parameters
emergence <- "2019-06-01"
duration <- 60

tmn <- 6
topt <- 20
tmx <- 30
dryhr <- 2
splash <- 0
lwmn <- 3
lwmx <- 12

###############################################################################
# Read weather data
###############################################################################

stndir <- paste(prjdir, "/obs", sep = "")
stninfo <- read.csv(
  paste(stndir, stnfile, sep = "/"),
  header = TRUE
)

stnlon <- stninfo$Lon[stninfo$ID == stnid]
stnlat <- stninfo$Lat[stninfo$ID == stnid]

if (wthhourly == TRUE) {
  wth <- Get.daily.wth.korea(stndir, stnid, stnlon, stnlat)
  wth <- .subsetwth(wth, emergence, duration)
  hwth <- Get.hourly.wth.korea(stndir, stnid, emergence, duration)
} else {
  wth <- Get.wth(stndir, stnid, stnlon, stnlat)
  wth <- .subsetwth(wth, emergence, duration)
  hwth <- hourly.weather(wth, simple = FALSE)
  
  # Standardize column names and structure
  hwth$pr <- 0
  hwth <- hwth[, c(
    "date",
    "hr",
    "pr",
    "rh",
    "temp",
    "lw"
  )]
  
  colnames(hwth) <- c(
    "date",
    "hr",
    "pr",
    "rh",
    "tp",
    "lw"
  )
}

###############################################################################
# Generic infection model
###############################################################################

Texp <- (topt - tmn) / (tmx - topt)
lwmnmx <- lwmn / lwmx

hrs <- length(hwth$date)

if (hrs != length(wth@w$date) * 24) {
  stop("Wrong hourly weather data")
}

ft <- dryhrs <- wethrs <- ftlw <- splsh <- sum_ftlw <-
  scld_sum_ftlw <- risk <- rep(0, times = hrs)

for (i in 2:hrs) {

  # Temperature response
  if (hwth$tp[i] < tmn) {
    ft[i] <- 0
  } else {
    if (hwth$tp[i] > tmx) {
      ft[i] <- 0
    } else {
      ft[i] <- (
        ((tmx - hwth$tp[i]) / (tmx - topt)) *
          ((hwth$tp[i] - tmn) / (topt - tmn))
      )^Texp
    }
  }

  # Consecutive dry hours
  if (hwth$lw[i] == 0) {
    dryhrs[i] <- dryhrs[i - 1] + 1
  } else {
    dryhrs[i] <- 0
  }

  # Wet-hour accumulation logic
  if (dryhrs[i] >= dryhr && hwth$lw[i] == 0) {
    wethrs[i] <- 0
  } else {
    wethrs[i] <- wethrs[i - 1] + hwth$lw[i]
  }

  # Temperature response weighted by leaf wetness
  ftlw[i] <- hwth$lw[i] * ft[i]

  # Splash logic
  if (hwth$lw[i] >= splash || scld_sum_ftlw[i - 1] > 0) {
    splsh[i] <- 1
  } else {
    splsh[i] <- 0
  }


  if (wethrs[i] == 0) {
    sum_ftlw[i] <- 0
  } else {
    sum_ftlw[i] <- (sum_ftlw[i - 1] + ftlw[i]) * splsh[i]
  }

  scld_sum_ftlw[i] <- sum_ftlw[i] / lwmn

  if (scld_sum_ftlw[i] < lwmnmx) {
    imsirisk <- 0
  } else {
    imsirisk <- (scld_sum_ftlw[i] - lwmnmx) * lwmnmx
  }

  risk[i] <- min(imsirisk, 1)
}

###############################################################################
# Save hourly and daily output
###############################################################################

res <- cbind(
  hwth,
  ft,
  dryhrs,
  wethrs,
  ftlw,
  splsh,
  sum_ftlw,
  scld_sum_ftlw,
  risk
)

colnames(res) <- c(
  "date",
  "hr",
  "prec",
  "rhum",
  "temp",
  "lf_wtness",
  "T_response",
  "dryhours",
  "wethours",
  "function_T(ft)",
  "splash",
  "Sum_ft",
  "Scaled_sum_ft",
  "disease_risk"
)

output_dir <- file.path(prjdir, "Output")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

write.csv(
  res,
  file.path(output_dir, "hourlyinfectionrisk.csv"),
  row.names = FALSE
)


d_res <- data.frame(
  wth@w,
  daily_risk = rep(0, nrow(wth@w))
)

for (d in wth@w$date) {
  d_res$daily_risk[d_res$date == d] <- mean(
    res$disease_risk[which(res$date == d)]
  )
}

write.csv(
  d_res,
  file.path(output_dir, "dailyinfectionrisk.csv"),
  row.names = FALSE
)

###############################################################################
# Plot results
###############################################################################

setwd(prjdir)

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  stop("Package 'ggplot2' is required for plotting")
}

library(ggplot2)

p_daily <- ggplot(
  d_res,
  aes(x = date, y = daily_risk)
) +
  labs(
    title = "Daily infection risk",
    x = "Date",
    y = "Infection risk"
  ) +
  geom_line() +
  geom_point()

print(p_daily)

p_hourly <- ggplot(
  res,
  aes(x = date, y = disease_risk)
) +
  labs(
    title = "Hourly infection risk",
    x = "Date",
    y = "Infection risk"
  ) +
  geom_line()

print(p_hourly)

###############################################################################
# Calibration to weekly field severity observations
#
# Purpose
#   The generic infection model produces an environmental infection risk (R_t),
#   not disease severity itself. This section adds a calibration layer that
#   converts daily infection risk into expected disease severity and estimates
#   a small number of field-calibration parameters.
#
# Severity model
#   lambda_t = kappa * R_(t-lag)^beta
#   h_t      = 1 - exp(-lambda_t)
#   S_t      = S_(t-1) + (1 - S_(t-1)) * h_t
#
# where
#   R_t   : daily infection risk from the generic infection model (0-1)
#   lambda_t: latent daily infection pressure (non-negative)
#   h_t   : daily fraction of the remaining healthy tissue becoming diseased
#   S_t   : expected disease severity as a proportion (0-1)
#   kappa : overall field infection-pressure scale
#   beta  : nonlinear response to generic infection risk
#   lag   : delay in days between infection conditions and visible symptoms
#
# The biological parameters tmn, topt, tmx, lwmn, lwmx, and dryhr remain fixed.
###############################################################################

# -------------------------------
# 1. Prepare daily infection risk
# -------------------------------

daily_cal <- data.frame(
  date = as.Date(d_res$date),
  risk = pmax(0, pmin(as.numeric(d_res$daily_risk), 1))
)

# -------------------------------
# 2. Severity simulation function
# -------------------------------

simulate_severity <- function(daily_risk, kappa, beta, lag_days, s0 = 0) {
  n <- length(daily_risk)
  
  if (n == 0) {
    stop("daily_risk is empty.")
  }
  if (kappa < 0 || beta <= 0 || lag_days < 0) {
    stop("kappa must be >= 0, beta must be > 0, and lag_days must be >= 0.")
  }
  
  severity <- numeric(n)
  lambda <- numeric(n)
  hazard <- numeric(n)
  
  previous_severity <- pmax(0, pmin(s0, 1))
  
  for (t in seq_len(n)) {
    source_day <- t - lag_days
    lagged_risk <- if (source_day >= 1) daily_risk[source_day] else 0
    
    lambda[t] <- kappa * lagged_risk^beta
    hazard[t] <- 1 - exp(-lambda[t])
    
    severity[t] <- previous_severity +
      (1 - previous_severity) * hazard[t]
    previous_severity <- severity[t]
  }
  
  data.frame(
    severity = severity,
    lambda = lambda,
    hazard = hazard
  )
}

# -------------------------------
# 3. Create example weekly severity observations
# -------------------------------
# Replace this block later with a read.csv() call for actual field observations.
# Required columns for real data:
#   date, severity
# where severity is expressed as percent (0-100).

set.seed(20260724)

true_parameter <- list(
  kappa = 0.20,
  beta = 1.40,
  lag_days = 7,
  s0 = 0.01
)

true_curve <- simulate_severity(
  daily_risk = daily_cal$risk,
  kappa = true_parameter$kappa,
  beta = true_parameter$beta,
  lag_days = true_parameter$lag_days,
  s0 = true_parameter$s0
)

# Weekly observations beginning 7 days after the simulation start.
weekly_index <- seq(from = 7, to = nrow(daily_cal), by = 7)

field_severity <- data.frame(
  date = daily_cal$date[weekly_index],
  severity = 100 * true_curve$severity[weekly_index]
)

# Add realistic observation error and constrain severity to 0-100%.
field_severity$severity <- field_severity$severity +
  rnorm(nrow(field_severity), mean = 0, sd = 2.5)
field_severity$severity <- pmax(0, pmin(field_severity$severity, 100))

# Severity assessed repeatedly on the same plants is usually non-decreasing.
# Remove cummax() when independent plants are sampled at each assessment or
# when apparent decreases are biologically/observationally plausible.
field_severity$severity <- cummax(field_severity$severity)

write.csv(
  field_severity,
  file.path(output_dir, "example_weekly_field_severity.csv"),
  row.names = FALSE
)

# To use real field data, replace the synthetic data above with, for example:
# field_severity <- read.csv(
#   file.path(prjdir, "FieldData", "weekly_severity.csv"),
#   stringsAsFactors = FALSE
# )
# field_severity$date <- as.Date(field_severity$date)
# field_severity$severity <- as.numeric(field_severity$severity)

# -------------------------------
# 4. Objective function for fitting
# -------------------------------

severity_objective <- function(par, lag_days, daily_cal, field_severity) {
  kappa <- par[1]
  beta <- par[2]
  s0 <- par[3]
  
  predicted <- simulate_severity(
    daily_risk = daily_cal$risk,
    kappa = kappa,
    beta = beta,
    lag_days = lag_days,
    s0 = s0
  )$severity * 100
  
  obs_index <- match(field_severity$date, daily_cal$date)
  
  if (anyNA(obs_index)) {
    stop("Some field observation dates are outside the model simulation period.")
  }
  
  residual <- field_severity$severity - predicted[obs_index]
  
  # Sum of squared errors on the percentage-severity scale.
  sum(residual^2, na.rm = TRUE)
}

# -------------------------------
# 5. Fit kappa, beta, s0, and lag
# -------------------------------
# lag_days is discrete and is selected by grid search.
# For each candidate lag, kappa, beta, and s0 are optimized with bounds.

candidate_lags <- 0:21
fit_by_lag <- vector("list", length(candidate_lags))

for (j in seq_along(candidate_lags)) {
  current_lag <- candidate_lags[j]
  
  fit_by_lag[[j]] <- optim(
    par = c(kappa = 0.10, beta = 1.00, s0 = 0.01),
    fn = severity_objective,
    lag_days = current_lag,
    daily_cal = daily_cal,
    field_severity = field_severity,
    method = "L-BFGS-B",
    lower = c(kappa = 1e-6, beta = 0.10, s0 = 0.00),
    upper = c(kappa = 5.00, beta = 5.00, s0 = 0.95)
  )
}

fit_summary <- data.frame(
  lag_days = candidate_lags,
  kappa = vapply(fit_by_lag, function(x) x$par[1], numeric(1)),
  beta = vapply(fit_by_lag, function(x) x$par[2], numeric(1)),
  s0 = vapply(fit_by_lag, function(x) x$par[3], numeric(1)),
  SSE = vapply(fit_by_lag, function(x) x$value, numeric(1)),
  convergence = vapply(fit_by_lag, function(x) x$convergence, integer(1))
)

best_row <- which.min(fit_summary$SSE)
best_fit <- fit_summary[best_row, ]

best_curve <- simulate_severity(
  daily_risk = daily_cal$risk,
  kappa = best_fit$kappa,
  beta = best_fit$beta,
  lag_days = best_fit$lag_days,
  s0 = best_fit$s0
)

calibration_result <- data.frame(
  date = daily_cal$date,
  daily_risk = daily_cal$risk,
  lambda = best_curve$lambda,
  daily_hazard = best_curve$hazard,
  predicted_severity = 100 * best_curve$severity
)

obs_index <- match(field_severity$date, calibration_result$date)
obs_prediction <- calibration_result$predicted_severity[obs_index]

RMSE <- sqrt(mean((field_severity$severity - obs_prediction)^2))
MAE <- mean(abs(field_severity$severity - obs_prediction))
R2 <- 1 - sum((field_severity$severity - obs_prediction)^2) /
  sum((field_severity$severity - mean(field_severity$severity))^2)

fit_statistics <- data.frame(
  parameter = c("kappa", "beta", "lag_days", "s0", "RMSE", "MAE", "R2"),
  estimate = c(
    best_fit$kappa,
    best_fit$beta,
    best_fit$lag_days,
    best_fit$s0,
    RMSE,
    MAE,
    R2
  )
)

write.csv(
  fit_summary,
  file.path(output_dir, "severity_fit_by_lag.csv"),
  row.names = FALSE
)

write.csv(
  calibration_result,
  file.path(output_dir, "daily_fitted_severity.csv"),
  row.names = FALSE
)

write.csv(
  fit_statistics,
  file.path(output_dir, "severity_fit_statistics.csv"),
  row.names = FALSE
)

cat("\nBest severity calibration parameters\n")
print(fit_statistics)

# -------------------------------
# 6. Plot observed and fitted severity
# -------------------------------

p_fit <- ggplot() +
  geom_line(
    data = calibration_result,
    aes(x = date, y = predicted_severity),
    linewidth = 0.9
  ) +
  geom_point(
    data = field_severity,
    aes(x = date, y = severity),
    size = 2.5
  ) +
  labs(
    title = "Calibration of the fungal generic infection model",
    subtitle = paste0(
      "kappa = ", round(best_fit$kappa, 3),
      ", beta = ", round(best_fit$beta, 3),
      ", lag = ", best_fit$lag_days, " days",
      ", RMSE = ", round(RMSE, 2)
    ),
    x = "Date",
    y = "Disease severity (%)"
  ) +
  coord_cartesian(ylim = c(0, 100))

print(p_fit)

ggsave(
  filename = file.path(output_dir, "severity_calibration_plot.png"),
  plot = p_fit,
  width = 8,
  height = 5,
  dpi = 300
)

###############################################################################
# Notes for real field data
#
# 1. Keep the biological infection parameters fixed when reliable literature or
#    controlled-environment estimates are available.
# 2. Fit only kappa, beta, lag_days, and optionally s0 to field severity data.
# 3. Use independent site-year data for validation rather than evaluating the
#    model only with the data used for calibration.
# 4. When severity is recorded on an ordinal scale, use an ordinal observation
#    model rather than treating the scores as exact percentages.
# 5. When repeated observations are collected from multiple fields or years,
#    consider a nonlinear mixed-effects or hierarchical model with a shared beta
#    and lag but site-year-specific kappa and/or s0.
###############################################################################