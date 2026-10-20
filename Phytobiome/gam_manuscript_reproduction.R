# Reproduction of the manuscript GAM (Fig. 3c) and robustness checks
# ===================================================================
# Manuscript spec: beta regression, logit link, mgcv; CLR abundances;
#   logit(FHB) = b0 + g(GIBSIM) + sum_j s_j(genus_j) + sum_(j,k in SPIEC-EASI) f_jk(genus_j, genus_k)
# Unstated settings recovered by grid search (all 11 significance calls match, R2 = 0.686 vs 0.69):
#   s(..., k = 3, bs = "cr"), ti(..., k = 3), method = "REML", select = TRUE,
#   response = FHB/100 used as-is (0 and 1 squeezed by betar's default eps).
#
# Checks:
#   A. response handling for the 5 zeros / 1 one: raw (betar eps), clamp [0.001, 0.999],
#      Smithson-Verkuilen (y(n-1)+0.5)/n, and dropping the 6 boundary samples
#   B. M2 = M1 + ti(GIBSIM, Alternaria)
#   C. bootstrap stability selection (term selected when p < 0.05), B = 200
#
# Run:  Rscript Phytobiome/gam_manuscript_reproduction.R

suppressMessages({library(mgcv); library(readxl); library(parallel)})
here <- "Phytobiome"
out <- file.path(here, "output", "gam_manuscript")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
B <- 200

d <- as.data.frame(read_excel(file.path(here, "using_data.xlsx")))
names(d)[names(d) == "GIBSIM_pred"] <- "GIBSIM"
n <- nrow(d)
y0 <- d$FHB_incidence / 100
boundary <- y0 == 0 | y0 == 1

ms_table <- data.frame(
  term = c("s(GIBSIM)", "s(Alternaria)", "s(Cladosporium)", "s(Epicoccum)", "s(Hannaella)",
           "s(Papiliotrema)", "s(Periconia)", "ti(Alternaria,Cladosporium)",
           "ti(Alternaria,Papiliotrema)", "ti(Alternaria,Periconia)", "ti(Cladosporium,Epicoccum)",
           "ti(GIBSIM,Alternaria)"),
  ms_edf = c(1.917, 1.776, 0, 0.832, 0.919, 0, 1.482, 0, 1.796, 1.901, 0.807, NA),
  ms_p = c(0, 0, 0.861, 0.011, 0, 0.365, 0, 0.981, 0.002, 0.004, 0.074, NA))

rhs_m1 <- paste(
  "s(GIBSIM, k = 3, bs = 'cr') + s(Alternaria, k = 3, bs = 'cr') + s(Cladosporium, k = 3, bs = 'cr') +",
  "s(Epicoccum, k = 3, bs = 'cr') + s(Hannaella, k = 3, bs = 'cr') + s(Papiliotrema, k = 3, bs = 'cr') +",
  "s(Periconia, k = 3, bs = 'cr') + ti(Alternaria, Cladosporium, k = 3) + ti(Alternaria, Papiliotrema, k = 3) +",
  "ti(Alternaria, Periconia, k = 3) + ti(Cladosporium, Epicoccum, k = 3)")
rhs_m2 <- paste(rhs_m1, "+ ti(GIBSIM, Alternaria, k = 3)")

responses <- list(
  raw     = list(y = y0, keep = rep(TRUE, n)),
  clamp   = list(y = pmin(pmax(y0, 0.001), 0.999), keep = rep(TRUE, n)),
  SV      = list(y = (y0 * (n - 1) + 0.5) / n, keep = rep(TRUE, n)),
  drop0_1 = list(y = y0, keep = !boundary))

fit <- function(rhs, data)
  suppressWarnings(gam(as.formula(paste("y ~", rhs)), data = data, family = betar(link = "logit"),
                       method = "REML", select = TRUE))
tab <- function(m) {
  s <- summary(m)$s.table
  data.frame(term = gsub(", k = [0-9]+|, bs = 'cr'| ", "", rownames(s)), edf = s[, "edf"],
             p = s[, "p-value"], row.names = NULL)
}

sink(file.path(out, "gam_manuscript_log.txt"), split = TRUE)
cat(sprintf("n = %d; FHB = 0%%: %d samples, FHB = 100%%: %d samples\n", n, sum(y0 == 0), sum(y0 == 1)))
cat(sprintf("GIBSIM-only R2 (lm FHB ~ GIBSIM_pred) = %.3f   [manuscript: 0.49]\n",
            summary(lm(FHB_incidence ~ GIBSIM, d))$r.squared))

full <- list()
for (rn in names(responses)) for (mod in c("M1", "M2")) {
  r <- responses[[rn]]
  dd <- d[r$keep, ]; dd$y <- r$y[r$keep]
  m <- fit(if (mod == "M1") rhs_m1 else rhs_m2, dd)
  t <- merge(tab(m), ms_table, by = "term", all.x = TRUE, sort = FALSE)
  r2 <- cor(dd$y, fitted(m))^2
  cat(sprintf("\n=== response = %s | %s | n = %d | R2(obs, fitted) = %.3f | AIC = %.1f\n",
              rn, mod, nrow(dd), r2, AIC(m)))
  print(transform(t, edf = round(edf, 3), p = round(p, 4)), row.names = FALSE)
  full[[paste(rn, mod)]] <- cbind(response = rn, model = mod, n = nrow(dd), R2 = r2, t)
}
full <- do.call(rbind, full)
write.csv(full, file.path(out, "gam_manuscript_fullsample.csv"), row.names = FALSE)

# Epicoccum p-value at a glance
cat("\n=== s(Epicoccum) across response handling\n")
print(reshape(full[full$term == "s(Epicoccum)", c("response", "model", "edf", "p")],
              idvar = "response", timevar = "model", direction = "wide"), row.names = FALSE)

# ---------------------------------------------------------------- stability selection
boot_one <- function(b, rn, rhs) {
  set.seed(1000 + b)
  r <- responses[[rn]]
  dd <- d[r$keep, ]; dd$y <- r$y[r$keep]
  dd <- dd[sample(nrow(dd), replace = TRUE), ]
  m <- try(fit(rhs, dd), silent = TRUE)
  if (inherits(m, "try-error")) return(NULL)
  cbind(b = b, tab(m))
}
stab <- list()
for (rn in c("raw", "SV")) for (mod in c("M1", "M2")) {
  res <- do.call(rbind, mclapply(seq_len(B), boot_one, rn = rn,
                                 rhs = if (mod == "M1") rhs_m1 else rhs_m2,
                                 mc.cores = max(1, detectCores())))
  nb <- length(unique(res$b))
  agg <- do.call(rbind, lapply(split(res, res$term), function(r)
    data.frame(term = r$term[1], sel_p05_pct = 100 * mean(r$p < 0.05),
               median_edf = median(r$edf), median_p = median(r$p))))
  agg <- agg[order(-agg$sel_p05_pct), ]
  cat(sprintf("\n=== Stability selection: response = %s, %s (B = %d)\n", rn, mod, nb))
  print(transform(agg, sel_p05_pct = round(sel_p05_pct, 1), median_edf = round(median_edf, 2),
                  median_p = round(median_p, 3)), row.names = FALSE)
  stab[[paste(rn, mod)]] <- cbind(response = rn, model = mod, B_ok = nb, agg)
  write.csv(do.call(rbind, stab), file.path(out, "gam_manuscript_stability.csv"), row.names = FALSE)
}
sink()

stab <- do.call(rbind, stab)
png(file.path(out, "gam_manuscript_stability.png"), width = 2600, height = 2000, res = 200)
par(mfrow = c(2, 2), mar = c(4, 13, 3, 1))
for (rn in c("raw", "SV")) for (mod in c("M1", "M2")) {
  s <- stab[stab$response == rn & stab$model == mod, ]
  s <- s[order(s$sel_p05_pct), ]
  bp <- barplot(s$sel_p05_pct, names.arg = s$term, horiz = TRUE, las = 1, xlim = c(0, 105),
                col = ifelse(grepl("Epicoccum", s$term), "#eb6834", "#2a78d6"), border = NA,
                cex.names = .7, xlab = "% of bootstrap fits with p < 0.05",
                main = sprintf("%s, response = %s", if (mod == "M1") "Manuscript model" else
                               "+ ti(GIBSIM, Alternaria)", rn), cex.main = .9)
  text(s$sel_p05_pct + 1.5, bp, sprintf("%.0f%%", s$sel_p05_pct), adj = 0, cex = .65)
}
invisible(dev.off())
cat("Outputs in", out, "\n")
