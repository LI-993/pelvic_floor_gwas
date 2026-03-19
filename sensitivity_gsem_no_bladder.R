#!/usr/bin/env Rscript
# Sensitivity analysis: gSEM without Bladder
# Addresses Reviewer 2 comment 3 (low h² of Bladder)

suppressPackageStartupMessages({
  library(lavaan)
  library(data.table)
})

BASE_DIR <- "d:/Nproject/gwas/pelvic_floor_gwas"
RESULTS_DIR <- file.path(BASE_DIR, "results", "genomic_sem_proper")

# Load pre-computed LDSC results
ldsc_results <- fread(file.path(BASE_DIR, "results", "ldsc", "genetic_correlation_summary.tsv"))

# Use 5 phenotypes (exclude Bladder)
phenotypes <- c("POP", "BPH", "Constipation", "FemaleProlapse", "Incontinence")
n_traits <- length(phenotypes)

# Build S matrix
h2_values <- list()
for (row_idx in 1:nrow(ldsc_results)) {
  p1 <- ldsc_results$phenotype1[row_idx]
  p2 <- ldsc_results$phenotype2[row_idx]
  if (!(p1 %in% names(h2_values))) h2_values[[p1]] <- ldsc_results$h2_p1[row_idx]
  if (!(p2 %in% names(h2_values))) h2_values[[p2]] <- ldsc_results$h2_p2[row_idx]
}

S <- matrix(0, n_traits, n_traits)
rownames(S) <- colnames(S) <- phenotypes
for (i in seq_along(phenotypes)) S[i, i] <- h2_values[[phenotypes[i]]]

for (row_idx in 1:nrow(ldsc_results)) {
  p1 <- ldsc_results$phenotype1[row_idx]
  p2 <- ldsc_results$phenotype2[row_idx]
  if (p1 %in% phenotypes && p2 %in% phenotypes) {
    i <- which(phenotypes == p1)
    j <- which(phenotypes == p2)
    rg <- ldsc_results$rg[row_idx]
    cov_g <- rg * sqrt(S[i, i] * S[j, j])
    S[i, j] <- cov_g
    S[j, i] <- cov_g
  }
}

cat("S matrix (no Bladder):\n")
print(round(S, 6))

# Model 1: Single factor
model_1f <- paste0("PelvicFloor =~ ", paste(phenotypes, collapse = " + "))

# Model 2: Two factor (same structure as main analysis, minus Bladder)
model_2f <- '
  FemalePelvic =~ POP + FemaleProlapse + Incontinence
  Urinary =~ BPH + Incontinence
  FemalePelvic ~~ Urinary
'

models <- list(
  "1_SingleFactor" = model_1f,
  "2_TwoFactor" = model_2f
)

results <- data.frame()

for (name in names(models)) {
  cat("\nFitting", name, "...\n")
  tryCatch({
    fit <- cfa(
      model = models[[name]],
      sample.cov = S,
      sample.nobs = 100000,
      std.lv = TRUE,
      estimator = "ML"
    )
    fm <- fitMeasures(fit, c("chisq", "df", "pvalue", "cfi", "srmr", "aic"))
    row <- data.frame(
      Model = name,
      chisq = fm["chisq"], df = fm["df"], pvalue = fm["pvalue"],
      CFI = fm["cfi"], SRMR = fm["srmr"], AIC = fm["aic"],
      stringsAsFactors = FALSE
    )
    results <- rbind(results, row)
    cat("  CFI =", round(fm["cfi"], 4), ", SRMR =", round(fm["srmr"], 4), "\n")

    params <- parameterEstimates(fit, standardized = TRUE)
    loadings <- params[params$op == "=~", c("lhs", "rhs", "est", "se", "z", "pvalue", "std.all")]
    cat("  Loadings:\n")
    print(loadings)

    residuals_var <- params[params$op == "~~" & params$lhs == params$rhs & params$lhs %in% phenotypes,
                            c("lhs", "est", "std.all")]
    cat("  Residual variances:\n")
    print(residuals_var)

  }, error = function(e) {
    cat("  Error:", conditionMessage(e), "\n")
  })
}

cat("\n\n=== Model Comparison (No Bladder) ===\n")
print(results)

write.csv(results, file.path(RESULTS_DIR, "sensitivity_no_bladder.csv"), row.names = FALSE)
cat("\nSaved to:", file.path(RESULTS_DIR, "sensitivity_no_bladder.csv"), "\n")
