#!/usr/bin/env Rscript
# Sensitivity analysis: gSEM without FemaleProlapse
# Addresses Reviewer 2 comments 6-7 (POP-FemaleProlapse redundancy, Heywood case)

suppressPackageStartupMessages({
  library(lavaan)
  library(data.table)
})

BASE_DIR <- "d:/Nproject/gwas/pelvic_floor_gwas"
RESULTS_DIR <- file.path(BASE_DIR, "results", "genomic_sem_proper")

# Load pre-computed LDSC results
ldsc_results <- fread(file.path(BASE_DIR, "results", "ldsc", "genetic_correlation_summary.tsv"))

# Use 5 phenotypes (exclude FemaleProlapse)
phenotypes <- c("POP", "BPH", "Bladder", "Constipation", "Incontinence")
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

cat("S matrix (no FemaleProlapse):\n")
print(round(S, 6))

# Model 1: Single factor
model_1f <- paste0("PelvicFloor =~ ", paste(phenotypes, collapse = " + "))

# Model 2: Two factor (Prolapse/Anatomical vs Urinary)
model_2f <- '
  Pelvic =~ POP + Incontinence
  Urinary =~ BPH + Bladder + Incontinence
  Pelvic ~~ Urinary
'

# Model 3: Two factor variant
model_2f_v2 <- '
  Prolapse =~ POP + Constipation
  Urinary =~ BPH + Bladder + Incontinence
  Prolapse ~~ Urinary
'

models <- list(
  "1_SingleFactor" = model_1f,
  "2_TwoFactor" = model_2f,
  "2b_TwoFactor_v2" = model_2f_v2
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

    # Print parameter estimates for best inspection
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

cat("\n\n=== Model Comparison (No FemaleProlapse) ===\n")
print(results)

# Save
write.csv(results, file.path(RESULTS_DIR, "sensitivity_no_femaleprolapse.csv"), row.names = FALSE)
cat("\nSaved to:", file.path(RESULTS_DIR, "sensitivity_no_femaleprolapse.csv"), "\n")
