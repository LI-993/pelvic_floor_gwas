# LAVA Full Analysis - All 6 Phenotypes
# Using UK Biobank reference (100k individuals, much better power)

library(LAVA)

BASE_DIR <- "D:/Nproject/gwas/pelvic_floor_gwas"
REF_PREFIX <- file.path(BASE_DIR, "reference/lava_ukb/lava-ukb-v1.1")
INPUT_INFO <- file.path(BASE_DIR, "data/lava/input.info.txt")
SAMPLE_OVERLAP <- file.path(BASE_DIR, "data/lava/sample.overlap.txt")
LOCUS_FILE <- file.path(BASE_DIR, "LAVA-main/support_data/blocks_s2500_m25_f1_w200.GRCh37_hg19.locfile")
OUT_DIR <- file.path(BASE_DIR, "results/lava")

PHENOTYPES <- c("POP", "BPH", "Bladder", "Constipation", "FemaleProlapse", "Incontinence")

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat(paste(rep("=", 60), collapse = ""), "\n")
cat("LAVA Local Genetic Correlation Analysis\n")
cat("Phenotypes:", paste(PHENOTYPES, collapse = ", "), "\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

# Load input
cat("Loading input data...\n")
input <- process.input(
  input.info.file = INPUT_INFO,
  sample.overlap.file = SAMPLE_OVERLAP,
  ref.prefix = REF_PREFIX,
  phenos = PHENOTYPES
)

cat("\nSNPs shared:", "~10M\n")
cat("Reading loci...\n")
loci <- read.loci(LOCUS_FILE)
n_loci <- nrow(loci)
cat("Total loci:", n_loci, "\n\n")

# Analysis
cat("Running LAVA analysis...\n")
cat("(This will take several hours)\n\n")

univ_thresh <- 0.05 / n_loci  # Bonferroni correction
univ_results <- list()
bivar_results <- list()

# Progress tracking
progress_points <- ceiling(seq(n_loci * 0.1, n_loci, length.out = 10))
start_time <- Sys.time()

for (i in 1:n_loci) {
  # Progress
  if (i %in% progress_points) {
    elapsed <- difftime(Sys.time(), start_time, units = "mins")
    pct <- round(100 * i / n_loci)
    cat(sprintf("[%d%%] Locus %d/%d (%.1f min elapsed)\n", pct, i, n_loci, elapsed))
  }

  # Process locus
  locus <- tryCatch(
    suppressMessages(process.locus(loci[i, ], input)),
    error = function(e) NULL
  )

  if (!is.null(locus)) {
    loc_info <- data.frame(
      locus = locus$id,
      chr = locus$chr,
      start = locus$start,
      stop = locus$stop,
      n.snps = locus$n.snps,
      n.pcs = locus$K
    )

    # Run tests
    loc_out <- tryCatch(
      suppressMessages(run.univ.bivar(locus, univ.thresh = univ_thresh)),
      error = function(e) NULL
    )

    if (!is.null(loc_out)) {
      if (!is.null(loc_out$univ)) {
        univ_results[[length(univ_results) + 1]] <- cbind(loc_info, loc_out$univ)
      }
      if (!is.null(loc_out$bivar)) {
        bivar_results[[length(bivar_results) + 1]] <- cbind(loc_info, loc_out$bivar)
      }
    }
  }
}

# Combine results
cat("\nCombining results...\n")
univ_df <- if (length(univ_results) > 0) do.call(rbind, univ_results) else data.frame()
bivar_df <- if (length(bivar_results) > 0) do.call(rbind, bivar_results) else data.frame()

# Save
cat("Saving results...\n")
write.table(univ_df, file.path(OUT_DIR, "lava_univariate.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)
write.table(bivar_df, file.path(OUT_DIR, "lava_bivariate.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)

# Summary
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("LAVA Analysis Complete!\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

cat("Results:\n")
cat("  Univariate tests:", nrow(univ_df), "\n")
cat("  Bivariate tests:", nrow(bivar_df), "\n")
cat("  Output:", OUT_DIR, "\n")

# Significant results
if (nrow(bivar_df) > 0) {
  sig_bivar <- bivar_df[bivar_df$p < 0.05, ]
  cat("\nSignificant local rg (p < 0.05):", nrow(sig_bivar), "\n")

  if (nrow(sig_bivar) > 0) {
    cat("\nTop local genetic correlations:\n")
    top <- head(bivar_df[order(bivar_df$p), ], 20)
    print(top[, c("locus", "chr", "phen1", "phen2", "rho", "p")])
  }
}

cat("\nTotal time:", round(difftime(Sys.time(), start_time, units = "hours"), 2), "hours\n")
