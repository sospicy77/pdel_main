#############################################
# Generic Bacterial Infection Model (v8)
# - Multiplication–Dispersal–Infection structure
# - Risk = N × D × I
# - Different dispersal pathways per disease
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
  wind = runif(30, 0, 10)       # wind speed (m/s)
)

# -------------------------------
# 2. GLOBAL PARAMETERS
# -------------------------------

global_params <- list(
  growth_period = 3         # Number of preceding days used to calculate N
)

# -------------------------------
# 3. DISEASE-SPECIFIC PARAMETERS
# -------------------------------

disease_params <- list(

  fire_blight = list(  ##화상병
    N0 = 0.15,             # Initial inoculum pressure (0-1)

    mu_max = 0.30,         # Maximum bacterial growth rate
                           # Growth rate under optimal temperature and RH

    T_min = 4,             # Minimum temperature for growth (°C)
    T_opt = 28,            # Optimum temperature for growth (°C)
    T_max = 36,            # Maximum temperature for growth (°C)

    rh_thresh_M = 50,      # RH threshold for multiplication module (%)
                           # f_RH(RH) in growth rate reaches 0.5 at this RH

    rh_thresh_I = 90,      # RH threshold for infection module (%)
                           # Wetness response W(RH, Rain) reaches 0.5 at this RH

    rain_thresh = 0.25,    # Rainfall threshold (mm)
                           # Wetness response = 1 when daily rain >= this value

    rain_coeff = 0.055,    # Rain-driven dispersal coefficient
    wind_coeff = 0.030,    # Wind-driven dispersal coefficient
    interaction_coeff = 0.0030,  # Rain × wind interaction coefficient
    insect_coeff = 0.10,   # Insect-mediated dispersal probability
                           # (independent of weather)

    use_rain = TRUE,       # Include rain dispersal pathway
    use_wind = TRUE,       # Include wind dispersal pathway
    use_interaction = TRUE,# Include rain × wind interaction pathway
    use_insect = TRUE,     # Include insect-mediated dispersal
    use_dispersal = TRUE   # Calculate dispersal module D (FALSE: D = 1)
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
    rh_thresh_I = 90,
    rain_thresh = 0.25,

    # Mainly rain splash / wetness-driven.
    rain_coeff = 0.070,
    wind_coeff = 0.005,
    interaction_coeff = 0.0005,
    insect_coeff = 0.00,

    use_rain = TRUE,
    use_wind = FALSE,
    use_interaction = FALSE,
    use_insect = FALSE,
    use_dispersal = TRUE
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

    # Strong winds and continuous heavy rain favor spread.
    rain_coeff = 0.060,
    wind_coeff = 0.030,
    interaction_coeff = 0.0030,
    insect_coeff = 0.00,

    use_rain = TRUE,
    use_wind = TRUE,
    use_interaction = TRUE,
    use_insect = FALSE,
    use_dispersal = TRUE
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
    rh_thresh_I = 90,
    rain_thresh = 0.25,

    # Base module keeps aboveground dispersal off.
    rain_coeff = 0.000,
    wind_coeff = 0.000,
    interaction_coeff = 0.000,
    insect_coeff = 0.00,

    use_rain = FALSE,
    use_wind = FALSE,
    use_interaction = FALSE,
    use_insect = FALSE,
    use_dispersal = FALSE
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

  if(p$rain_coeff < 0 || p$wind_coeff < 0 || p$interaction_coeff < 0){
    stop("Dispersal coefficients must be non-negative.")
  }

  if(p$insect_coeff < 0 || p$insect_coeff > 1){
    stop("insect_coeff must be between 0 and 1.")
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

calc_dispersal <- function(data, disease_param){

  if(!disease_param$use_dispersal){
    return(rep(1, nrow(data)))
  }

  total_effect <- rep(0, nrow(data))

  if(disease_param$use_rain){
    total_effect <- total_effect +
      disease_param$rain_coeff * data$rain
  }

  if(disease_param$use_wind){
    total_effect <- total_effect +
      disease_param$wind_coeff * data$wind
  }

  if(disease_param$use_interaction){
    total_effect <- total_effect +
      disease_param$interaction_coeff * data$rain * data$wind
  }

  D <- 1 - exp(-total_effect)

  if(disease_param$use_insect){
    D <- 1 - (1 - D) * (1 - disease_param$insect_coeff)
  }

  D
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
