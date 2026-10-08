#############################################
# Run bacterialmodel_v8 (fire blight) with observed weather
# Station : ID101 (Attapeu)
# Period  : 2019-03-01 ~ 2019-06-30
#############################################

# Run from the "2026_Generic infection models" folder.

model_file <- "bacterialmodel_v8.R"
wth_file   <- file.path("obs", "ID101.csv")
out_dir    <- "Output"

disease    <- "fire_blight"
start_date <- as.Date("2019-03-01")
end_date   <- as.Date("2019-06-30")

# -------------------------------
# 1. LOAD MODEL FUNCTIONS / PARAMETERS
# -------------------------------

# Evaluate only the parameter lists and function definitions so that the
# example data, example run and plot in the model file are not executed.
model_exprs <- parse(model_file, encoding = "UTF-8")
for(e in model_exprs){
  if(is.call(e) && identical(e[[1]], as.name("<-")) &&
     (as.character(e[[2]]) %in% c("global_params", "disease_params") ||
      (is.call(e[[3]]) && identical(e[[3]][[1]], as.name("function"))))){
    eval(e, envir = globalenv())
  }
}

# -------------------------------
# 2. WEATHER DATA
# -------------------------------

wth <- read.csv(wth_file, check.names = FALSE)
wth[wth == -99] <- NA

wth$date <- as.Date(sprintf("%d-%02d-%02d", wth$Year, wth$Mon, wth$Day))
wth <- wth[wth$date >= start_date & wth$date <= end_date, ]

# Tavg is missing (-99) in this file, so use (Tmax + Tmin) / 2.
tavg <- ifelse(is.na(wth$`Tavg(c)`),
               (wth$`Tmax(c)` + wth$`Tmin(c)`) / 2,
               wth$`Tavg(c)`)

data <- data.frame(
  date = wth$date,
  temp = tavg,
  rh   = wth$`RHumidity(fr)`,   # values are in % despite the "(fr)" label
  rain = wth$`Pcp(mm)`,
  wind = wth$`WSpeed(m/s)`
)

if(anyNA(data)){
  stop("Weather data contain missing values in the simulation period.")
}

# -------------------------------
# 3. RUN MODEL
# -------------------------------

disease_param <- disease_params[[disease]]

res <- run_model(data, disease_param, global_params$growth_period)

dir.create(out_dir, showWarnings = FALSE)
write.csv(cbind(res["date"], data[, c("temp", "rh", "rain", "wind")], res[-1]),
          file.path(out_dir, "fireblight_ID101_2019_MarJun_v8.csv"),
          row.names = FALSE)

cat("Max daily Risk  :", round(max(res$Risk), 3), "\n")
cat("Mean daily Risk :", round(mean(res$Risk), 3), "\n")

# -------------------------------
# 4. PLOT
# -------------------------------

png(file.path(out_dir, "fireblight_ID101_2019_MarJun_v8.png"),
    width = 2400, height = 2400, res = 250)

par(mfrow = c(4, 1), mar = c(2.5, 4.5, 2, 4.5), oma = c(2, 0, 2, 0))
month_ticks <- seq(start_date, end_date + 1, by = "month")

# (a) Weather: temperature and rainfall
plot(data$date, data$temp, type = "n", ylim = range(data$temp) + c(-1, 1),
     xaxt = "n", xlab = "", ylab = expression(T[avg]~(degree*C)))
rect(data$date - 0.4, par("usr")[3], data$date + 0.4,
     par("usr")[3] + data$rain / max(data$rain) * diff(par("usr")[3:4]) * 0.9,
     col = "#9ecae1", border = NA)
lines(data$date, data$temp, col = "#d7301f", lwd = 1.5)
abline(h = c(disease_param$T_opt, disease_param$T_max), lty = 3, col = "grey40")
axis(4, at = par("usr")[3] + pretty(c(0, max(data$rain))) / max(data$rain) *
       diff(par("usr")[3:4]) * 0.9,
     labels = pretty(c(0, max(data$rain))))
mtext("Rain (mm)", side = 4, line = 2.5, cex = 0.8)
axis.Date(1, at = month_ticks, format = "%b")
title("(a) Weather: daily mean temperature (line) and rainfall (bars)",
      adj = 0, cex.main = 1)

# (b) Relative humidity and wind
plot(data$date, data$rh, type = "l", col = "#2171b5", lwd = 1.5,
     ylim = c(0, 100), xaxt = "n", xlab = "", ylab = "RH (%)")
abline(h = disease_param$rh_thresh_M, lty = 3, col = "#238b45")
abline(h = disease_param$rh_thresh_I, lty = 3, col = "#d94801")
par(new = TRUE)
plot(data$date, data$wind, type = "l", col = "#636363", lwd = 1,
     axes = FALSE, xlab = "", ylab = "", ylim = c(0, max(data$wind) * 1.1))
axis(4)
mtext("Wind (m/s)", side = 4, line = 2.5, cex = 0.8)
axis.Date(1, at = month_ticks, format = "%b")
legend("bottomleft", c("RH", "Wind", "RH_th (M)", "RH_th (I)"),
       col = c("#2171b5", "#636363", "#238b45", "#d94801"),
       lty = c(1, 1, 3, 3), bty = "n", horiz = TRUE, cex = 0.85)
title("(b) Weather: relative humidity and wind speed", adj = 0, cex.main = 1)

# (c) Module outputs
plot(res$date, res$N, type = "l", col = "#238b45", lwd = 1.5, ylim = c(0, 1.25),
     xaxt = "n", yaxt = "n", xlab = "", ylab = "Module value")
axis(2, at = seq(0, 1, 0.2))
lines(res$date, res$D, col = "#6a51a3", lwd = 1.5)
lines(res$date, res$I, col = "#d94801", lwd = 1.5)
axis.Date(1, at = month_ticks, format = "%b")
legend("topleft", c("N (multiplication)", "D (dispersal)", "I (infection)"),
       col = c("#238b45", "#6a51a3", "#d94801"), lty = 1, lwd = 1.5,
       bty = "n", horiz = TRUE, cex = 0.85)
title("(c) Module outputs", adj = 0, cex.main = 1)

# (d) Daily infection risk
plot(res$date, res$Risk, type = "h", col = "#fc9272", lwd = 2,
     ylim = c(0, max(res$Risk) * 1.15), xaxt = "n", xlab = "", ylab = "Risk")
lines(res$date, res$Risk, col = "#cb181d", lwd = 1)
axis.Date(1, at = month_ticks, format = "%b")
title("(d) Daily infection risk (Risk = N x D x I)", adj = 0, cex.main = 1)

mtext("Date (2019)", side = 1, outer = TRUE, line = 0.5)
mtext("Fire blight (model v8) - ID101 (Attapeu), 2019-03-01 ~ 2019-06-30",
      side = 3, outer = TRUE, line = 0.5, font = 2)

dev.off()
