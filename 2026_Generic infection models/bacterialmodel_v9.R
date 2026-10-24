#############################################
# Generic Bacterial Infection Model (v9)
# - Multiplication–Dispersal–Infection structure, Risk = N × D × I
# - Base modules (all diseases):
#     N : discrete logistic growth, r = mu_max × f_T(T) × f_RH(RH)
#     D : D = K_d (constant, fittable), used when no dispersal add-on is on
#     I : I = W(RH, Rain)
# - Add-on modules (switched on per disease):
#     N : degree-hour driven growth, r = mu_max × (1 - exp(-DH / DH_scale)) × f_RH(RH)
#         (DH = degree-hours > 18.3 °C; Maryblyt EIP, Cougarblight)
#     D : weather-driven dispersal (rain, wind, rain × wind) and insect pathway
#         replace the base constant: D = D_add = 1 - (1 - D_weather)(1 - D_insect)
#     I : wetting on the day after rain >= Rain_prev (Maryblyt),
#         temperature response f_T(T) and host vulnerable period V
#         I = max(W(RH, Rain), 1[Rain_{t-1} >= Rain_prev]) × f_T(T) × V
#   With every add-on switched off (and k_M = 0.3, k_I = 1), v9 is identical to v8.
#############################################

# -------------------------------
# 1. INPUT DATA (example)
# -------------------------------

set.seed(1)

data <- data.frame(
  date = seq(as.Date("2024-04-01"), as.Date("2024-04-30"), by = "day"),
  temp = runif(30, 5, 35),      # daily mean temperature (°C)
  rh   = runif(30, 50, 100),    # relative humidity (%)
  rain = runif(30, 0, 20),      # rainfall amount (mm)
  wind = runif(30, 0, 10),      # wind speed (m/s), needed only if use_wind / use_interaction
  degree_hours = runif(30, 0, 120),  # daily degree-hours > 18.3 °C from hourly temperature,
                                     # needed only if use_dh_growth
  vulnerable_period = ifelse(1:30 %in% 10:20, 1, 0)
  # vulnerable_period (needed only if use_vulnerable_period):
  # 1 = host tissue / growth stage is vulnerable (flowering, young tissue, wounds)
  # 0 = host is not in a vulnerable stage
)

# -------------------------------
# 2. GLOBAL PARAMETERS
# -------------------------------

global_params <- list(
  growth_period = 3         # Number of preceding days used to calculate N
                            # (a disease may override it with its own growth_period)
)

# -------------------------------
# 3. DISEASE-SPECIFIC PARAMETERS
# -------------------------------

disease_params <- list(

  fire_blight = list(  ##화상병
    # Calibrated to the pear fire blight model (Maryblyt-type BIR):
    # hourly_99 / CF-99, 2010-2025 bloom periods, target (BIR - 25) / 75,
    # parameters and add-ons selected by leave-one-season-out RMSE.
    # [v8: ...] = value in bacterialmodel_v8.R

    growth_period = 4,     # Days of growth history for N (overrides global 3)

    # ---- base: multiplication (N) ----
    N0 = 0.1529,           # Initial inoculum pressure (0-1)        [v8: 0.15]
    mu_max = 2.00,         # Maximum bacterial growth rate          [v8: 0.30]
                           # (2 = stability limit of the discrete logistic)
    T_min = 4,             # Minimum temperature (°C), f_T in I
    T_opt = 21.2062,       # Optimum temperature (°C), f_T in I     [v8: 28]
    T_max = 36,            # Maximum temperature (°C), f_T in I
    rh_thresh_M = 0,       # RH threshold for growth (%)            [v8: 50]
                           # fitted 0: no RH limitation of growth
    k_M = 0.3,             # Logistic slope of f_RH in N

    # ---- base: dispersal (D) ----
    K_d = 0.20,            # Dispersal coefficient (constant, 0-1)
                           # used only when no dispersal add-on is on

    # ---- base: infection (I) ----
    rh_thresh_I = 41.1785, # RH threshold for infection wetness (%)  [v8: 90]
    rain_thresh = 0.67,    # Rainfall threshold (mm), W = 1 above    [v8: 0.25]
    k_I = 0.05,            # Logistic slope of W in RH                [v8: 1]
                           # (gradual RH response of wetness)

    # ---- add-on: degree-hour driven growth (N) ----
    use_dh_growth = TRUE,  # needs data$degree_hours
    DH_scale = 49.3842,    # Degree-hours giving 63 % of the maximum growth rate

    # ---- add-on: weather-driven / insect dispersal (D) ----
    use_rain = TRUE,       # rain splash pathway
    use_wind = FALSE,      # wind pathway (needs wind data; not calibrated: no wind in hourly_99)
    use_interaction = FALSE,  # rain × wind pathway (needs wind data; not calibrated)
    use_insect = TRUE,     # insect pathway (weather independent)
    rain_coeff = 4.9875,   # a_rain: rain-driven dispersal coefficient [v7: 0.055]
    wind_coeff = 0.030,    # a_wind: wind-driven dispersal coefficient (v7 value)
    interaction_coeff = 0.0030,  # a_int: rain × wind interaction coefficient (v7 value)
    insect_coeff = 0.9026, # D_insect: insect-mediated dispersal probability (0-1) [v7: 0.10]

    # ---- add-on: infection (I) ----
    use_prev_rain_wetness = TRUE,  # wetting on the day after rain (Maryblyt)
    rain_prev_thresh = 2.3001,     # previous-day rain giving W = 1 (mm) [Maryblyt: 2.5]
    use_temp_I = TRUE,     # temperature response f_T(T) in I
    use_vulnerable_period = TRUE   # host vulnerable period V in I (bloom)
                                   # (V = 1 on all calibration days)
  ),

  bacterial_spot = list(  ##고추세균성점무늬병
    N0 = 0.08,

    # Xanthomonas arboricola pv. pruni:
    # infection tested from 5–35 C;
    # infection optimum estimated at 28.9 C;
    # in vitro growth reported from 5–34 C, not above 35 C.
    mu_max = 0.26,
    T_min = 5,
    T_opt = 29,
    T_max = 35,
    rh_thresh_M = 50,
    k_M = 0.3,

    K_d = 0.20,            # used only when no dispersal add-on is on

    rh_thresh_I = 90,
    rain_thresh = 0.25,
    k_I = 1,

    # Mainly rain splash / wetness-driven.
    use_rain = TRUE,
    use_wind = FALSE,
    use_interaction = FALSE,
    use_insect = FALSE,
    rain_coeff = 0.070,
    wind_coeff = 0.005,
    interaction_coeff = 0.0005,
    insect_coeff = 0.00,

    use_dh_growth = FALSE,
    DH_scale = 50,
    use_prev_rain_wetness = FALSE,
    rain_prev_thresh = 2.5,
    use_temp_I = TRUE,
    use_vulnerable_period = TRUE
  ),

  bacterial_blight = list( ##벼 흰잎마름병
    N0 = 0.12,

    # Rice bacterial blight / Xanthomonas oryzae pv. oryzae:
    # disease-favoring environment commonly reported as 25–34 C
    # and RH > 70%.
    mu_max = 0.27,
    T_min = 10,
    T_opt = 30,
    T_max = 38,
    rh_thresh_M = 50,
    k_M = 0.3,

    K_d = 0.20,            # used only when no dispersal add-on is on

    rh_thresh_I = 90,
    rain_thresh = 0.25,
    k_I = 1,

    # Strong winds and continuous heavy rain favor spread.
    use_rain = TRUE,
    use_wind = TRUE,
    use_interaction = TRUE,
    use_insect = FALSE,
    rain_coeff = 0.060,
    wind_coeff = 0.030,
    interaction_coeff = 0.0030,
    insect_coeff = 0.00,

    use_dh_growth = FALSE,
    DH_scale = 50,
    use_prev_rain_wetness = FALSE,
    rain_prev_thresh = 2.5,
    use_temp_I = TRUE,
    use_vulnerable_period = TRUE
  ),

  ralstonia = list(  ##고추풋마름병
    N0 = 0.30,

    # Ralstonia solanacearum:
    # warm-climate soilborne pathogen;
    # in vitro maximum growth often around 30 C,
    # bacterial wilt intensity can be high at 30–35 C depending on host/root injury.
    mu_max = 0.32,
    T_min = 10,
    T_opt = 30,
    T_max = 40,

    # RH is only a weak proxy here; later soil moisture/root infection module needed.
    rh_thresh_M = 50,
    k_M = 0.3,

    # No aboveground dispersal limitation (D = 1): base module only.
    K_d = 1.00,

    rh_thresh_I = 90,
    rain_thresh = 0.25,
    k_I = 1,

    use_rain = FALSE,
    use_wind = FALSE,
    use_interaction = FALSE,
    use_insect = FALSE,
    rain_coeff = 0.000,
    wind_coeff = 0.000,
    interaction_coeff = 0.000,
    insect_coeff = 0.00,

    use_dh_growth = FALSE,
    DH_scale = 50,
    use_prev_rain_wetness = FALSE,
    rain_prev_thresh = 2.5,
    use_temp_I = TRUE,
    use_vulnerable_period = FALSE
  )
)

# -------------------------------
# 4. PARAMETER VALIDATION
# -------------------------------

validate_disease_params <- function(p){

  if(!(p$T_min < p$T_opt && p$T_opt < p$T_max)){
    stop("Temperature parameters must satisfy T_min < T_opt < T_max.")
  }

  if(p$N0 < 0 || p$N0 > 1){
    stop("N0 must be between 0 and 1.")
  }

  if(p$mu_max < 0){
    stop("mu_max must be non-negative.")
  }

  if(p$rh_thresh_M < 0 || p$rh_thresh_M > 100 ||
     p$rh_thresh_I < 0 || p$rh_thresh_I > 100){
    stop("rh_thresh_M and rh_thresh_I must be between 0 and 100.")
  }

  if(p$rain_thresh < 0){
    stop("rain_thresh must be non-negative.")
  }

  if(p$K_d < 0 || p$K_d > 1){
    stop("K_d must be between 0 and 1.")
  }

  if(p$rain_coeff < 0 || p$wind_coeff < 0 || p$interaction_coeff < 0){
    stop("Dispersal coefficients must be non-negative.")
  }

  if(p$insect_coeff < 0 || p$insect_coeff > 1){
    stop("insect_coeff must be between 0 and 1.")
  }

  if(p$k_M <= 0 || p$k_I <= 0){
    stop("k_M and k_I must be positive.")
  }

  if(p$use_dh_growth && p$DH_scale <= 0){
    stop("DH_scale must be positive.")
  }

  if(p$use_prev_rain_wetness && p$rain_prev_thresh < 0){
    stop("rain_prev_thresh must be non-negative.")
  }
}

validate_data <- function(data, p){
  need <- c("date", "temp", "rh", "rain")
  if(p$use_wind || p$use_interaction) need <- c(need, "wind")
  if(p$use_vulnerable_period) need <- c(need, "vulnerable_period")
  if(p$use_dh_growth) need <- c(need, "degree_hours")
  missing <- setdiff(need, names(data))
  if(length(missing) > 0){
    stop("Input data are missing column(s): ", paste(missing, collapse = ", "))
  }
}

# -------------------------------
# 5. RESPONSE FUNCTIONS
# -------------------------------

# Cardinal temperature response (0-1), used in N and (add-on) in I
temp_response <- function(T, T_min, T_opt, T_max){

  response <- rep(0, length(T))

  valid <- T > T_min & T < T_max

  exponent <- (T_max - T_opt) / (T_opt - T_min)

  response[valid] <-
    ((T[valid] - T_min) *
       (T_max - T[valid])^exponent) /
    ((T_opt - T_min) *
       (T_max - T_opt)^exponent)

  pmin(response, 1)
}

# Humidity response, used in the multiplication module
# f_RH(RH) = 1 / (1 + exp[-k * (RH - RH_th)])
# RH_th: RH where the response reaches 0.5, k: logistic slope
rh_response <- function(RH, RH_th, k = 0.3){
  plogis(k * (RH - RH_th))
}

# Wetness response, used in the infection module
# W(RH, Rain) = 1                                  if Rain >= Rain_th
#             = 1 / (1 + exp[-k * (RH - RH_th)])   if Rain <  Rain_th
wetness_response <- function(RH, Rain, RH_th, Rain_th, k = 1){
  ifelse(Rain >= Rain_th, 1, rh_response(RH, RH_th, k))
}

# -------------------------------
# 6. MULTIPLICATION (N)
# -------------------------------

calc_population <- function(data, disease_param, growth_period){

  if(length(growth_period) != 1 || is.na(growth_period) ||
     growth_period < 0 || growth_period != as.integer(growth_period)){
    stop("growth_period must be a non-negative integer.")
  }

  n <- nrow(data)

  # Daily growth rate r_t = mu_max * g * f_RH(RH)
  #   base   : g = f_T(daily mean T)
  #   add-on : g = 1 - exp(-DH / DH_scale), DH = degree-hours > 18.3 °C
  if(disease_param$use_dh_growth){
    temp_eff <- 1 - exp(-data$degree_hours / disease_param$DH_scale)
  } else {
    temp_eff <- temp_response(
      data$temp,
      disease_param$T_min,
      disease_param$T_opt,
      disease_param$T_max
    )
  }
  rh_eff <- rh_response(data$rh, disease_param$rh_thresh_M, disease_param$k_M)
  r <- disease_param$mu_max * temp_eff * rh_eff

  N <- numeric(n)

  # For each date, restart from N0 and apply the daily growth rates
  # from t - growth_period through t.
  for(t in seq_len(n)){
    current_N <- disease_param$N0

    for(s in max(1, t - growth_period):t){
      current_N <- current_N + r[s] * current_N * (1 - current_N)
      current_N <- max(0, min(current_N, 1))
    }

    N[t] <- current_N
  }

  N
}

# -------------------------------
# 7. DISPERSAL (D)
# -------------------------------

# Add-on: weather-driven and insect dispersal
#   E_total   = a_rain × Rain + a_wind × Wind + a_int × Rain × Wind
#   D_weather = 1 - exp(-E_total)
#   D_add     = 1 - (1 - D_weather)(1 - D_insect)
calc_dispersal_addon <- function(data, disease_param){

  p <- disease_param
  total_effect <- rep(0, nrow(data))
  if(p$use_rain){
    total_effect <- total_effect + p$rain_coeff * data$rain
  }
  if(p$use_wind){
    total_effect <- total_effect + p$wind_coeff * data$wind
  }
  if(p$use_interaction){
    total_effect <- total_effect + p$interaction_coeff * data$rain * data$wind
  }
  D_weather <- 1 - exp(-total_effect)

  if(p$use_insect){
    1 - (1 - D_weather) * (1 - p$insect_coeff)
  } else {
    D_weather
  }
}

use_dispersal_addon <- function(p){
  p$use_rain || p$use_wind || p$use_interaction || p$use_insect
}

# Base: D = K_d when no dispersal add-on is used.
# Add-on: D = D_add (weather / insect pathways), K_d is not used.
calc_dispersal <- function(data, disease_param){
  if(use_dispersal_addon(disease_param)){
    calc_dispersal_addon(data, disease_param)
  } else {
    rep(disease_param$K_d, nrow(data))
  }
}

# -------------------------------
# 8. INFECTION (I)
# -------------------------------

# Base W(RH, Rain) × add-ons (previous-day rain wetting, f_T(T), V)
calc_infection <- function(data, disease_param){

  I <- wetness_response(
    data$rh,
    data$rain,
    disease_param$rh_thresh_I,
    disease_param$rain_thresh,
    disease_param$k_I
  )

  if(disease_param$use_prev_rain_wetness){
    rain_prev <- c(0, head(data$rain, -1))
    I <- pmax(I, as.numeric(rain_prev >= disease_param$rain_prev_thresh))
  }

  if(disease_param$use_temp_I){
    I <- I * temp_response(
      data$temp,
      disease_param$T_min,
      disease_param$T_opt,
      disease_param$T_max
    )
  }

  if(disease_param$use_vulnerable_period){
    I <- I * data$vulnerable_period
  }

  I
}

# -------------------------------
# 9. RUN MODEL
# -------------------------------

run_model <- function(data, disease_param, growth_period = 3){

  validate_disease_params(disease_param)
  validate_data(data, disease_param)

  if(!is.null(disease_param$growth_period)) growth_period <- disease_param$growth_period
  N <- calc_population(data, disease_param, growth_period)
  D <- calc_dispersal(data, disease_param)
  I <- calc_infection(data, disease_param)

  data.frame(
    date = data$date,
    N = N,
    D = D,
    I = I,
    Risk = N * D * I
  )
}

# -------------------------------
# 10. RUN
# -------------------------------

disease <- "fire_blight"

disease_param <- disease_params[[disease]]

res <- run_model(data, disease_param, global_params$growth_period)

print(res)

# -------------------------------
# 11. VISUALIZATION
# -------------------------------

plot(
  res$date,
  res$Risk,
  type = "l",
  col = "red",
  main = paste("Risk -", disease),
  ylab = "Risk",
  xlab = "Date"
)
