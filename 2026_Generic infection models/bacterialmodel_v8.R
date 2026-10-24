#############################################
# Generic Bacterial Infection Model (v8)
# - Multiplication–Dispersal–Infection structure
# - Risk = N × D × I
# - Dispersal as a constant, fittable parameter (D = K_d)
#############################################

# -------------------------------
# 1. INPUT DATA (example)
# -------------------------------

set.seed(1)

data <- data.frame(
  date = seq(as.Date("2024-04-01"), as.Date("2024-04-30"), by = "day"),
  temp = runif(30, 5, 35),      # daily mean temperature (°C)
  rh   = runif(30, 50, 100),    # relative humidity (%)
  rain = runif(30, 0, 20)       # rainfall amount (mm)
)

# -------------------------------
# 2. GLOBAL PARAMETERS
# -------------------------------

global_params <- list(
  growth_period = 3         # Number of preceding days used to calculate N
                            # (used when a disease has no growth_period)
)

# -------------------------------
# 3. DISEASE-SPECIFIC PARAMETERS
# -------------------------------

disease_params <- list(

  # Initial values from the literature review of the four bacterial
  # diseases (pear fire blight, pepper bacterial spot, kiwifruit canker,
  # bacterial wilt). Hourly thresholds of the source models were converted
  # to daily means with CF-99 hourly_99 data (logistic fit of the daily
  # fraction of qualifying hours on daily mean RH or daily rain).
  # mu_max: literature doubling times at T_opt are a few hours
  # (ln2 * 24 / Td > 2 per day); the discrete logistic is stable only for
  # r <= 2, so mu_max = min(2, ln2 * 24 / Td).
  # N0 and K_d have no literature values and are calibration targets.

  fire_blight = list(  ##화상병 (Erwinia amylovora)
    N0 = 0.15,             # Initial inoculum pressure (0-1); no literature value

    mu_max = 2.00,         # Maximum bacterial growth rate (per day, capped at 2)

    T_min = 4,             # Growth 4-37 C, optimum 28 C (Santander & Biosca 2017)
    T_opt = 28,
    T_max = 37,

    rh_thresh_M = 50,      # Maryblyt EIP has no RH term: f_RH >= 0.95 above 60 % RH

    rh_thresh_I = 80,      # Maryblyt-type wetting: daily mean RH >= 80 %

    rain_thresh = 0.25,    # Maryblyt wetting: daily rain >= 0.25 mm (Steiner 1990)

    growth_period = 5,     # Cougarblight 4 d window (Smith 1993); Maryblyt EIP
                           # 66.7 DD > 4.4 C ~ 6.3 d at 15 C bloom; mean 5 d

    K_d = 0.20             # Dispersal coefficient (constant, 0-1); no literature value
  ),

  bacterial_spot = list(  ##고추세균점무늬병 (Xanthomonas euvesicatoria)
    N0 = 0.08,

    # No growth at 10 C in culture (Kim et al. 2014); optimum 24-30 C
    # (approximate, compendium value).
    # Doubling time g(27 C) = 3.28 h (Kim et al. 2014) -> 5.07 per day -> cap 2.
    mu_max = 2.00,
    T_min = 10,
    T_opt = 27,
    T_max = 35,

    rh_thresh_M = 85,      # Growth in wet hours (RH >= 90 % or rain): daily mean
                           # RH at which half of the hours are wet = 85.1 %
    rh_thresh_I = 90,      # Leaf wetness assumption of the source model
    rain_thresh = 10.4,    # Entry needs rain > 4 mm/h: daily rain at which a
                           # > 4 mm/h hour occurs on half of the days = 10.4 mm

    growth_period = 3,     # No literature value

    K_d = 0.20
  ),

  kiwi_canker = list(  ##키위궤양병 (Pseudomonas syringae pv. actinidiae)
    N0 = 0.10,             # No literature value

    # M-index valid -6.5 to 24.5 C, maximum at 15.0 C, zero when daily
    # T_max > 26 C (PSA-V Risk model); conducive 12-18 C, weakened at 25 C
    # (Serizawa & Ichikawa 1993). T_max = mean(24.5, 26, 25).
    mu_max = 2.00,
    T_min = -6.5,
    T_opt = 15,
    T_max = 25,

    rh_thresh_M = 76,      # M counted in hours with RH > 81 % (Beresford et al.
                           # 2017): daily mean RH at which half of the hours
                           # exceed 81 % = 76.2 %
    rh_thresh_I = 76,      # Same RH criterion in the R' index
    rain_thresh = 1.05,    # Rain > 1 mm (Beresford et al. 2017), >= 1.1 mm
                           # (PSA-V Risk model); mean

    growth_period = 3,     # R = 3-day sum of M (Beresford et al. 2017)

    K_d = 0.20             # No literature value
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
    rh_thresh_I = 90,
    rain_thresh = 0.25,

    K_d = 0.20
  ),

  ralstonia = list(  ##세균성시들음병 (Ralstonia solanacearum)
    N0 = 0.30,

    # Bacterial wilt model: r_max 0.35-0.40 per day, 10/30/42 C (Rosso et
    # al. 1995 type cardinal model); growth about 10-41 C, optimum about
    # 30 C (Hayward 1991, approximate); T_max = mean(42, 41, 40).
    mu_max = 0.38,
    T_min = 10,
    T_opt = 30,
    T_max = 41,

    # Soil-borne: air RH is only a weak proxy; no literature value.
    rh_thresh_M = 50,
    rh_thresh_I = 90,
    rain_thresh = 0.25,

    growth_period = 3,     # No literature value

    # No aboveground dispersal limitation (D = 1).
    K_d = 1.00
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
}

# -------------------------------
# 5. RESPONSE FUNCTIONS
# -------------------------------

# Cardinal temperature response (0-1), used in the multiplication module
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

  # Daily growth rate r_t = mu_max * f_T(T) * f_RH(RH)
  temp_eff <- temp_response(
    data$temp,
    disease_param$T_min,
    disease_param$T_opt,
    disease_param$T_max
  )
  rh_eff <- rh_response(data$rh, disease_param$rh_thresh_M)
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

# D = K_d (constant dispersal coefficient)
calc_dispersal <- function(data, disease_param){
  rep(disease_param$K_d, nrow(data))
}

# -------------------------------
# 8. INFECTION (I)
# -------------------------------

# I = W(RH, Rain)
calc_infection <- function(data, disease_param){

  wetness_response(
    data$rh,
    data$rain,
    disease_param$rh_thresh_I,
    disease_param$rain_thresh
  )
}

# -------------------------------
# 9. RUN MODEL
# -------------------------------

run_model <- function(data, disease_param, growth_period = 3){

  validate_disease_params(disease_param)

  # A disease-specific growth_period overrides the global value
  if(!is.null(disease_param$growth_period)){
    growth_period <- disease_param$growth_period
  }

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
