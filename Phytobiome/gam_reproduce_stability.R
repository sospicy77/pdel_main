# GAM re-analysis of FHB_incidence
# =================================
# M1 : main smooths (GIBSIM + 6 fungi) + 4 fungus-pair ti() terms of the reference table
# M2 : M1 + ti(GIBSIM, Alternaria)
# Specs (shrinkage via select = TRUE, REML):
#   gauss_k3  : gaussian, k = 3 (closest to the reference table's edf), primary
#   betar_k3  : beta regression on FHB/100 (logit link), k = 3, sensitivity
#   gauss_k10 : gaussian, default k, sensitivity
# Stability selection: B bootstrap resamples; a term counts as "selected" when p < 0.05
# (also reported: edf > 0.5, i.e. not shrunk away).
#
# Run:  Rscript Phytobiome/gam_reproduce_stability.R

suppressMessages({library(mgcv); library(readxl); library(parallel)})
set.seed(42)
here <- "Phytobiome"
out  <- file.path(here, "output", "gam")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
B <- 200

d <- as.data.frame(read_excel(file.path(here, "using_data.xlsx")))
names(d)[names(d) == "GIBSIM_pred"] <- "GIBSIM"
d$p <- pmin(pmax(d$FHB_incidence / 100, 0.001), 0.999)

terms_m1 <- function(k, kt) {
  ks <- if (is.na(k)) "" else sprintf(", k = %d", k)
  kti <- if (is.na(kt)) "" else sprintf(", k = %d", kt)
  c(sprintf("s(%s%s)", c("GIBSIM", "Alternaria", "Cladosporium", "Epicoccum",
                          "Hannaella", "Papiliotrema", "Periconia"), ks),
    sprintf("ti(%s%s)", c("Alternaria, Cladosporium", "Alternaria, Papiliotrema",
                           "Alternaria, Periconia", "Cladosporium, Epicoccum"), kti))
}
specs <- list(
  gauss_k3  = list(fam = gaussian(), y = "FHB_incidence", k = 3,  kt = 3),
  betar_k3  = list(fam = betar(),    y = "p",             k = 3,  kt = 3),
  gauss_k10 = list(fam = gaussian(), y = "FHB_incidence", k = NA, kt = NA)
)
formula_for <- function(sp, with_gxa) {
  tt <- terms_m1(sp$k, sp$kt)
  if (with_gxa) tt <- c(tt, sprintf("ti(GIBSIM, Alternaria%s)",
                                    if (is.na(sp$kt)) "" else sprintf(", k = %d", sp$kt)))
  as.formula(paste(sp$y, "~", paste(tt, collapse = " + ")))
}
clean <- function(x) gsub(", k = [0-9]+", "", gsub(" ", "", x))
fit <- function(sp, with_gxa, data)
  suppressWarnings(gam(formula_for(sp, with_gxa), data = data, family = sp$fam,
                       method = "REML", select = TRUE))
tab <- function(m) {
  s <- summary(m)$s.table
  data.frame(term = clean(rownames(s)), edf = s[, "edf"], p = s[, "p-value"], row.names = NULL)
}

sink(file.path(out, "gam_log.txt"), split = TRUE)
cat("Reference table (user): s(GIBSIM) 1.917/0.000, s(Alternaria) 1.776/0.000, s(Cladosporium) 0/0.861,",
    "s(Epicoccum) 0.832/0.011, s(Hannaella) 0.919/0.000, s(Papiliotrema) 0/0.365, s(Periconia) 1.482/0.000,",
    "ti(A,C) 0/0.981, ti(A,Pap) 1.796/0.002, ti(A,Per) 1.901/0.004, ti(C,E) 0.807/0.074\n")

full_rows <- list()
for (nm in names(specs)) for (gx in c(FALSE, TRUE)) {
  m <- fit(specs[[nm]], gx, d)
  t <- tab(m)
  lab <- paste(nm, if (gx) "M2 (+ti(GIBSIM,Alternaria))" else "M1")
  cat(sprintf("\n=== %s | dev.expl = %.3f | adj.R2 = %.3f | AIC = %.1f\n", lab,
              summary(m)$dev.expl, summary(m)$r.sq, AIC(m)))
  print(transform(t, edf = round(edf, 3), p = round(p, 4)), row.names = FALSE)
  full_rows[[lab]] <- cbind(spec = nm, model = if (gx) "M2" else "M1", t,
                            dev_expl = summary(m)$dev.expl, AIC = AIC(m))
}
full <- do.call(rbind, full_rows)
write.csv(full, file.path(out, "gam_fullsample.csv"), row.names = FALSE)

# ---------------------------------------------------------------- stability selection
boot_one <- function(b, sp, gx) {
  set.seed(1000 + b)
  db <- d[sample(nrow(d), replace = TRUE), ]
  m <- try(fit(sp, gx, db), silent = TRUE)
  if (inherits(m, "try-error")) return(NULL)
  cbind(b = b, tab(m))
}
stab_rows <- list()
for (nm in names(specs)) for (gx in c(FALSE, TRUE)) {
  res <- do.call(rbind, mclapply(seq_len(B), boot_one, sp = specs[[nm]], gx = gx,
                                 mc.cores = max(1, detectCores())))
  nb <- length(unique(res$b))
  agg <- do.call(rbind, lapply(split(res, res$term), function(r)
    data.frame(term = r$term[1],
               sel_p05_pct = 100 * sum(r$p < 0.05) / nb,
               edf_gt_0.5_pct = 100 * sum(r$edf > 0.5) / nb,
               median_edf = median(r$edf), median_p = median(r$p))))
  agg <- agg[order(-agg$sel_p05_pct), ]
  cat(sprintf("\n=== Stability selection: %s %s (B = %d successful fits)\n", nm,
              if (gx) "M2" else "M1", nb))
  print(transform(agg, sel_p05_pct = round(sel_p05_pct, 1), edf_gt_0.5_pct = round(edf_gt_0.5_pct, 1),
                  median_edf = round(median_edf, 2), median_p = round(median_p, 3)), row.names = FALSE)
  stab_rows[[paste(nm, gx)]] <- cbind(spec = nm, model = if (gx) "M2" else "M1", B_ok = nb, agg)
}
stab <- do.call(rbind, stab_rows)
write.csv(stab, file.path(out, "gam_stability_selection.csv"), row.names = FALSE)
sink()

# ---------------------------------------------------------------- figure
png(file.path(out, "gam_stability_selection.png"), width = 2400, height = 1150, res = 200)
par(mfrow = c(1, 2), mar = c(4, 13, 3, 1))
for (mod in c("M1", "M2")) {
  s <- stab[stab$spec == "gauss_k3" & stab$model == mod, ]
  s <- s[order(s$sel_p05_pct), ]
  col <- ifelse(grepl("Epicoccum", s$term), "#eb6834", "#2a78d6")
  bp <- barplot(s$sel_p05_pct, names.arg = s$term, horiz = TRUE, las = 1, xlim = c(0, 100),
                col = col, border = NA, cex.names = .75,
                xlab = "% of bootstrap fits with p < 0.05",
                main = if (mod == "M1") "M1: reference terms" else "M2: + ti(GIBSIM, Alternaria)")
  text(s$sel_p05_pct + 2, bp, sprintf("%.0f%%", s$sel_p05_pct), adj = 0, cex = .7)
  abline(v = c(60, 80), lty = 3, col = "#52514e")
}
invisible(dev.off())
cat("\nOutputs in", out, "\n")
