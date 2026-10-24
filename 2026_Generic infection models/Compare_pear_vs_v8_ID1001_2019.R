#############################################
# Compare pear fire blight model (Maryblyt-type, gold standard)
# with bacterialmodel_v8 (fire_blight) at ID1001, 2019 bloom period
#############################################

# Run from the "2026_Generic infection models" folder after:
#   python3 pear_fire_blight_fixed.py obs/ID1001.csv 2019 Output/pear_ID1001_2019

model_file <- "bacterialmodel_v8.R"
pear_file  <- file.path("Output", "pear_ID1001_2019", "pear_daily_2019.csv")
out_dir    <- file.path("Output", "pear_ID1001_2019")
disease    <- "fire_blight"

# -------------------------------
# 1. LOAD v8 FUNCTIONS / PARAMETERS
# -------------------------------

model_exprs <- parse(model_file, encoding = "UTF-8")
for(e in model_exprs){
  if(is.call(e) && identical(e[[1]], as.name("<-")) &&
     (as.character(e[[2]]) %in% c("global_params", "disease_params") ||
      (is.call(e[[3]]) && identical(e[[3]][[1]], as.name("function"))))){
    eval(e, envir = globalenv())
  }
}

# -------------------------------
# 2. RUN v8 WITH THE SAME DAILY WEATHER
# -------------------------------

disease_param <- disease_params[[disease]]

merge_with_v8 <- function(file){
  pear <- read.csv(file)
  pear$tm <- as.Date(pear$tm)
  # Same daily values the pear model used (aggregated from hourly ID1001)
  wx <- data.frame(date = pear$tm, temp = pear$ta, rh = pear$hm, rain = pear$rn)
  v8 <- run_model(wx, disease_param, global_params$growth_period)
  data.frame(
    date = pear$tm, ta = pear$ta, rn = pear$rn, hm = pear$hm,
    B = pear$B == "True", H = pear$H == "True",
    W = pear$W == "True", T = pear$T == "True",
    EIP = pear$EIP, BIR = pear$BIR, level = pear$level,
    infection = pear$infection == "True",
    N = v8$N, D = v8$D, I = v8$I, Risk = v8$Risk
  )
}

# -------------------------------
# 3. MERGE, BLOOM PERIOD (pear B = TRUE: B1 <= day < PF)
# -------------------------------

cmp <- merge_with_v8(pear_file)
bloom <- cmp[cmp$B, ]
write.csv(bloom, file.path(out_dir, "compare_bloom_pear_vs_v8.csv"), row.names = FALSE)

# Supplementary what-if: pear BHWT with bloom forced to 03-01..06-30
force_file <- file.path(out_dir, "pear_daily_2019_forcebloom.csv")
whatif <- if(file.exists(force_file)) {
  x <- merge_with_v8(force_file); x[x$B, ]
} else NULL
if(!is.null(whatif)){
  write.csv(whatif, file.path(out_dir, "compare_forcebloom_pear_vs_v8.csv"), row.names = FALSE)
}

# -------------------------------
# 4. PERFORMANCE OF v8 AGAINST THE GOLD STANDARD
# -------------------------------

auc <- function(score, label){
  pos <- score[label]; neg <- score[!label]
  if(length(pos) == 0 || length(neg) == 0) return(NA)
  mean(outer(pos, neg, ">") + 0.5 * outer(pos, neg, "=="))
}

confusion <- function(pred, label){
  tp <- sum(pred & label);  fp <- sum(pred & !label)
  fn <- sum(!pred & label); tn <- sum(!pred & !label)
  c(TP = tp, FP = fp, FN = fn, TN = tn,
    sensitivity = tp / (tp + fn), specificity = tn / (tn + fp),
    precision = if(tp + fp > 0) tp / (tp + fp) else NA,
    accuracy = (tp + tn) / length(label))
}

best_threshold <- function(score, label){
  cand <- sort(unique(score))
  j <- sapply(cand, function(th){
    cm <- confusion(score >= th, label)
    cm["sensitivity"] + cm["specificity"] - 1
  })
  cand[which.max(j)]
}

binary_report <- function(x, label, name){
  cat("\n-- Target:", name, "| positives =", sum(label), "of", length(label), "--\n")
  if(sum(label) == 0 || sum(!label) == 0){
    cat("Only one class present: AUC / sensitivity cannot be estimated.\n")
    return(invisible())
  }
  cat("AUC (v8 Risk):", round(auc(x$Risk, label), 3), "\n")
  th <- best_threshold(x$Risk, label)
  cat("Youden-optimal threshold:", signif(th, 3), "\n")
  print(round(confusion(x$Risk >= th, label), 3))
  cat("Fixed threshold Risk >= 0.01:\n")
  print(round(confusion(x$Risk >= 0.01, label), 3))
}

evaluate <- function(x, title){
  cat("\n==================================================\n")
  cat(title, ":", format(min(x$date)), "~", format(max(x$date)),
      "(", nrow(x), "days )\n")
  cat("==================================================\n")
  cat("Gold standard BIR distribution:\n"); print(table(BIR = x$BIR))
  cat("Days with H:", sum(x$H), " W:", sum(x$W), " T:", sum(x$T),
      " infection (BIR = 100):", sum(x$infection), "\n")
  cat("\nv8 summary:\n")
  print(round(sapply(x[, c("N", "D", "I", "Risk")], summary), 4))

  cat("\nAgreement of v8 Risk with BIR\n")
  cat("Pearson r   :", round(cor(x$Risk, x$BIR), 3), "\n")
  cat("Spearman rho:", round(cor(x$Risk, x$BIR, method = "spearman"), 3), "\n")
  cat("Mean v8 Risk by BIR class:\n"); print(round(tapply(x$Risk, x$BIR, mean), 4))

  binary_report(x, x$BIR >= 50,  "BIR >= 50 (M or higher)")
  binary_report(x, x$BIR >= 75,  "BIR >= 75 (H or higher)")
  binary_report(x, x$infection,  "infection day (BIR = 100)")

  cat("\nComponent agreement\n")
  cat("pear W vs v8 I >= 0.5:\n"); print(round(confusion(x$I >= 0.5, x$W), 3))
  cat("AUC (v8 I for pear W):", round(auc(x$I, x$W), 3), "\n")
  cat("Pearson r, v8 N vs pear EIP      :", round(cor(x$N, x$EIP), 3), "\n")
  cat("Pearson r, v8 N vs daily mean temp:", round(cor(x$N, x$ta), 3), "\n")
  if(sum(x$H) > 0 && sum(!x$H) > 0)
    cat("AUC (v8 N for pear H):", round(auc(x$N, x$H), 3), "\n")
  if(sum(x$T) > 0 && sum(!x$T) > 0)
    cat("AUC (v8 N for pear T):", round(auc(x$N, x$T), 3), "\n")
}

sink(file.path(out_dir, "compare_metrics.txt"), split = TRUE)
evaluate(bloom, "MAIN: pear bloom period")
if(!is.null(whatif)) evaluate(whatif, "SUPPLEMENTARY what-if: bloom forced 03-01..06-30")
sink()

# -------------------------------
# 5. PLOT
# -------------------------------

png(file.path(out_dir, "compare_pear_vs_v8.png"), width = 2400, height = 2200, res = 250)
par(mfrow = c(3, 1), mar = c(3, 4.8, 2.4, 4.8), oma = c(1.5, 0, 2, 0))
xr <- range(bloom$date) + c(-0.5, 0.5)
day_ticks <- seq(min(bloom$date), max(bloom$date), by = 2)

# (a) weather
plot(bloom$date, bloom$ta, type = "n", xlim = xr,
     ylim = c(min(0, bloom$ta), max(20, bloom$ta) + 1), xaxt = "n",
     xlab = "", ylab = expression(T[avg]~(degree*C)))
rect(bloom$date - 0.35, 0, bloom$date + 0.35, bloom$rn / 25 * 20,
     col = "#9ecae1", border = NA)
lines(bloom$date, bloom$ta, col = "#d7301f", lwd = 2, type = "o", pch = 16, cex = 0.7)
lines(bloom$date, bloom$hm / 5, col = "#2171b5", lwd = 1.5, lty = 2)
abline(h = 15.6, lty = 3, col = "grey30")
axis(4, at = seq(0, 20, 5), labels = seq(0, 100, 25))
mtext("RH (%)  /  Rain (mm x 1.25)", side = 4, line = 2.8, cex = 0.75)
axis.Date(1, at = day_ticks, format = "%m-%d")
legend("topleft", c("Tavg", "RH", "Rain", "T threshold (15.6)"),
       col = c("#d7301f", "#2171b5", "#9ecae1", "grey30"),
       lty = c(1, 2, NA, 3), pch = c(16, NA, 15, NA), lwd = 2,
       bty = "n", horiz = TRUE, cex = 0.85)
title("(a) Weather in the bloom period", adj = 0, cex.main = 1)

# (b) gold standard vs v8 risk (raw values, two axes)
plot(bloom$date, bloom$BIR, type = "h", lwd = 9, col = "#fdae6b", xlim = xr,
     ylim = c(0, 100), xaxt = "n", xlab = "", ylab = "Pear BIR (0-100)")
text(bloom$date, bloom$BIR + 5, bloom$level, cex = 0.8, col = "#a63603")
par(new = TRUE)
plot(bloom$date, bloom$Risk, type = "o", pch = 16, col = "#54278f", lwd = 2,
     xlim = xr, ylim = c(0, max(bloom$Risk) * 1.6), axes = FALSE, xlab = "", ylab = "")
axis(4); mtext("v8 Risk", side = 4, line = 2.8, cex = 0.75)
axis.Date(1, at = day_ticks, format = "%m-%d")
legend("topleft", c("Pear BIR (gold standard)", "v8 Risk"),
       col = c("#fdae6b", "#54278f"), lwd = c(9, 2), pch = c(NA, 16),
       bty = "n", horiz = TRUE, cex = 0.85)
title("(b) Daily infection risk: pear model (bars) vs v8 (line)", adj = 0, cex.main = 1)

# (c) components
plot(bloom$date, bloom$I, type = "o", pch = 16, col = "#d94801", lwd = 2,
     xlim = xr, ylim = c(0, 1.15), xaxt = "n", xlab = "", ylab = "Value (0-1)")
lines(bloom$date, bloom$N, type = "o", pch = 15, col = "#238b45", lwd = 2)
points(bloom$date[bloom$W], rep(1.1, sum(bloom$W)), pch = 25, bg = "#2171b5", col = "#2171b5")
lines(bloom$date, pmin(bloom$EIP / 100, 1), type = "s", col = "grey40", lty = 2)
axis.Date(1, at = day_ticks, format = "%m-%d")
legend("topleft", c("v8 I (wetness)", "v8 N", "pear W = TRUE", "pear EIP / 100"),
       col = c("#d94801", "#238b45", "#2171b5", "grey40"),
       lty = c(1, 1, NA, 2), pch = c(16, 15, 25, NA), pt.bg = "#2171b5",
       lwd = 2, bty = "n", horiz = TRUE, cex = 0.85)
title("(c) Components: v8 I vs pear W, v8 N vs pear EIP", adj = 0, cex.main = 1)

mtext("Date (2019)", side = 1, outer = TRUE, line = 0.3)
mtext("Pear fire blight (Maryblyt-type) vs bacterialmodel_v8 - ID1001 (Gwangyang), bloom 2019",
      side = 3, outer = TRUE, line = 0.4, font = 2)
dev.off()

# (supplementary) what-if season plot
if(!is.null(whatif)){
  png(file.path(out_dir, "compare_forcebloom_pear_vs_v8.png"), width = 2400, height = 1500, res = 250)
  layout(matrix(c(1, 1, 2), nrow = 1))
  par(mar = c(4, 4.8, 3, 4.8))
  plot(whatif$date, whatif$BIR, type = "h", lwd = 3, col = "#fdae6b",
       ylim = c(0, 100), xlab = "Date (2019)", ylab = "Pear BIR (0-100)")
  points(whatif$date[whatif$infection], rep(100, sum(whatif$infection)),
         pch = 25, bg = "#a63603", col = "#a63603", cex = 0.8)
  par(new = TRUE)
  plot(whatif$date, whatif$Risk, type = "l", col = "#54278f", lwd = 1.8,
       ylim = c(0, max(whatif$Risk) * 1.6), axes = FALSE, xlab = "", ylab = "")
  axis(4); mtext("v8 Risk", side = 4, line = 2.8, cex = 0.75)
  legend("topleft", c("Pear BIR", "Pear infection day", "v8 Risk"),
         col = c("#fdae6b", "#a63603", "#54278f"), lwd = c(3, NA, 1.8),
         pch = c(NA, 25, NA), pt.bg = "#a63603", bty = "n", horiz = TRUE, cex = 0.85)
  title("(a) What-if: bloom forced 03-01 ~ 06-30", adj = 0, cex.main = 1)

  par(mar = c(4, 4.5, 3, 1))
  boxplot(Risk ~ BIR, data = whatif, col = c("#fee6ce", "#fdae6b", "#f16913", "#a63603"),
          xlab = "Pear BIR (gold standard)", ylab = "v8 Risk")
  title("(b) v8 Risk by pear BIR class", adj = 0, cex.main = 1)
  dev.off()
}
