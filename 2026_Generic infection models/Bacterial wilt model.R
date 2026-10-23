##references
#Elphinstone, J. G. (1996). Survival and possible mechanisms of survival of Ralstonia solanacearum. BulEPPO/EPPO Bulletin, 26(3‐4), 663-668.
#Caruso, P., et al. (2005). Seasonal variation in population size of Ralstonia solanacearum race 3 biovar 2 in a Spanish river and associated irrigation water. Applied and Environmental Microbiology, 71(3), 140-148.
#Rosso, L., Lobry, J. R., Bajard, S., & Flandrois, J. P. (1995). Convenient model to describe the combined effects of temperature and pH on microbial growth. Applied and Environmental Microbiology, 61(2), 610-616.

# 필요한 패키지 로드 (없을 시 설치)
if (!requireNamespace("deSolve", quietly = TRUE)) {
  install.packages("deSolve")
}
library(deSolve)

# 1. 온도 의존적 세균 증식률 함수 r(T) 정의
calc_growth_rate <- function(T, r_max = 0.35, T_min = 10, T_opt = 30, T_max = 42) {
  if (T <= T_min || T >= T_max) {
    return(0)
  }
  # 생물학적 온도 반응 함수 (Cardinal Temperature Response Model)
  r_T <- r_max * (1 - ((T - T_opt) / (T_opt - T_min))^2)
  return(max(0, r_T))
}

# 2. 세균성 시듦병 SEIR-Pathogen 연계 동역학 모형 정의
bacterial_wilt_model <- function(time, state, parameters) {
  with(as.list(c(state, parameters)), {
    
    # a. 세균 증식: 온도 의존적 로지스틱 성장
    r_T <- calc_growth_rate(Temp, r_max, T_min, T_opt, T_max)
    dB <- r_T * B * (1 - B / K_B) - decay_rate * B
    
    # b & c. 세균 확산 및 병원체 부하 의존적 전이단계 (S -> E -> I -> R)
    # 병원체 부하 B와 토양수분(moisture)에 따른 감염 압력(Infection pressure) 계산
    total_plants = S + E + I + R
    pathogen_effect = B / (K_half + B) # Monod-type saturation response
    
    # 토양수분 및 뿌리 접촉 확산을 반영한 감염 전이율
    infection_rate = (beta_soil * moisture * pathogen_effect + beta_root * (I / total_plants)) * S
    
    # 구획별 변화율 (ODEs)
    dS_dt = -infection_rate
    dE_dt = infection_rate - sigma * E
    dI_dt = sigma * E - gamma * I
    dR_dt = gamma * I
    
    list(c(dS_dt, dE_dt, dI_dt, dR_dt, dB))
  })
}

# 3. 모델 파라미터 및 초기 조건 설정
parameters <- c(
  r_max = 0.40,      # 최대 비증식률 (1/day)
  T_min = 10.0,      # 최저 생육 온도 (°C)
  T_opt = 30.0,      # 최적 온도 (°C)
  T_max = 42.0,      # 최고 생육 온도 (°C)
  Temp = 28.0,       # 환경 온도 (°C) - 최적온도에 가까운 조건
  K_B = 1e9,         # 병원체 최대 수용력 (CFU 또는 밀도 한계)
  decay_rate = 0.05, # 병원체 자연 사멸률
  beta_soil = 1e-6,  # 토양수 매개 전파 계수
  beta_root = 1e-3,  # 감염 주와의 뿌리 접촉 전파 계수
  moisture = 0.75,   # 토양 수분 함량 (0 ~ 1)
  K_half = 1e5,      # 반포화 상수 (병원체 부하 임계값)
  sigma = 0.2,       # 잠복기(E)에서 감염기(I)로의 전이율 (평균 잠복 5일)
  gamma = 0.1        # 감염기(I)에서 고사/제거(R)로의 전이율
)

# 초기 상태 값 (총 기주 식물 N = 1000, 초기 병원체 부하 B0 = 1e4)
initial_state <- c(
  S = 990,
  E = 10,
  I = 0,
  R = 0,
  B = 1e4
)

# 시뮬레이션 기간 설정 (총 60일, 일 단위)
time_steps <- seq(0, 60, by = 1)

# 모형 실행 (Standard ODE solver)
output <- ode(y = initial_state, times = time_steps, func = bacterial_wilt_model, parms = parameters)
output_df <- as.data.frame(output)
