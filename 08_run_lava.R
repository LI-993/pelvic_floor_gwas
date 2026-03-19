# LAVA Local Genetic Correlation Analysis
# Run this script in R to perform LAVA analysis

library(LAVA)

# ============================================================
# Configuration
# ============================================================

BASE_DIR <- "D:/Nproject/gwas/pelvic_floor_gwas"
DATA_DIR <- file.path(BASE_DIR, "data/ldsc")
OUT_DIR <- file.path(BASE_DIR, "results/lava")

# Reference data
REF_PREFIX <- file.path(BASE_DIR, "reference/lava/g1000_eur")

# Input files
INPUT_INFO <- file.path(BASE_DIR, "data/lava/input.info.txt")
SAMPLE_OVERLAP <- file.path(BASE_DIR, "data/lava/sample.overlap.txt")
LOCUS_FILE <- file.path(BASE_DIR, "LAVA-main/support_data/blocks_s2500_m25_f1_w200.GRCh37_hg19.locfile")

# Phenotypes
PHENOTYPES <- c("POP", "BPH", "Bladder", "Constipation", "FemaleProlapse", "Incontinence")

# Create output directory
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat(paste(rep("=", 60), collapse=""), "\n")
cat("LAVA Local Genetic Correlation Analysis\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

# ============================================================
# Step 1: Convert LDSC sumstats to LAVA format
# ============================================================

cat("Step 1: Preparing summary statistics...\n")

for (pheno in PHENOTYPES) {
  infile <- file.path(DATA_DIR, paste0(pheno, ".sumstats.gz"))
  outfile <- file.path(BASE_DIR, "data/lava", paste0(pheno, ".sumstats.txt"))

  if (!file.exists(outfile)) {
    cat("  Processing", pheno, "...\n")
    df <- read.table(gzfile(infile), header = TRUE, stringsAsFactors = FALSE)
    # LAVA needs: SNP A1 A2 N Z
    df <- df[, c("SNP", "A1", "A2", "N", "Z")]
    write.table(df, outfile, sep = "\t", row.names = FALSE, quote = FALSE)
    cat("    Saved", nrow(df), "SNPs\n")
  } else {
    cat("  ", pheno, "already exists\n")
  }
}

# ============================================================
# Step 2: Load input data
# ============================================================

cat("\nStep 2: Loading input data...\n")

input <- process.input(
  input.info.file = INPUT_INFO,
  sample.overlap.file = SAMPLE_OVERLAP,
  ref.prefix = REF_PREFIX,
  phenos = PHENOTYPES
)

cat("  SNPs in reference:", nrow(input$ref$bim), "\n")
cat("  Phenotypes loaded:", length(input$info$phenotype), "\n")

# ============================================================
# Step 3: Read loci
# ============================================================

cat("\nStep 3: Reading locus definitions...\n")
loci <- read.loci(LOCUS_FILE)
n_loci <- nrow(loci)
cat("  Total loci:", n_loci, "\n")

# ============================================================
# Step 4: Run univariate and bivariate tests
# ============================================================

cat("\nStep 4: Running LAVA analysis...\n")
cat("  This may take a while...\n\n")

# Set significance threshold for univariate test
univ_thresh <- 0.05 / n_loci  # Bonferroni correction

# Initialize results lists
univ_results <- list()
bivar_results <- list()

# Progress tracking
progress <- ceiling(quantile(1:n_loci, seq(0.1, 1, 0.1)))

for (i in 1:n_loci) {
  # Print progress
  if (i %in% progress) {
    pct <- names(progress[which(progress == i)])
    cat("  Progress:", pct, "\n")
  }

  # Process locus
  locus <- tryCatch(
    process.locus(loci[i, ], input),
    error = function(e) NULL
  )

  if (!is.null(locus)) {
    # Extract locus info
    loc_info <- data.frame(
      locus = locus$id,
      chr = locus$chr,
      start = locus$start,
      stop = locus$stop,
      n.snps = locus$n.snps,
      n.pcs = locus$K
    )

    # Run univariate and bivariate tests
    loc_out <- tryCatch(
      run.univ.bivar(locus, univ.thresh = univ_thresh),
      error = function(e) NULL
    )

    if (!is.null(loc_out)) {
      # Store univariate results
      if (!is.null(loc_out$univ)) {
        univ_results[[length(univ_results) + 1]] <- cbind(loc_info, loc_out$univ)
      }

      # Store bivariate results
      if (!is.null(loc_out$bivar)) {
        bivar_results[[length(bivar_results) + 1]] <- cbind(loc_info, loc_out$bivar)
      }
    }
  }
}

# ============================================================
# Step 5: Combine and save results
# ============================================================

cat("\nStep 5: Saving results...\n")

# Combine results
univ_df <- do.call(rbind, univ_results)
bivar_df <- do.call(rbind, bivar_results)

# Save results
univ_file <- file.path(OUT_DIR, "lava_univariate.tsv")
bivar_file <- file.path(OUT_DIR, "lava_bivariate.tsv")

write.table(univ_df, univ_file, sep = "\t", row.names = FALSE, quote = FALSE)
write.table(bivar_df, bivar_file, sep = "\t", row.names = FALSE, quote = FALSE)

cat("  Univariate results:", nrow(univ_df), "entries\n")
cat("  Bivariate results:", nrow(bivar_df), "entries\n")
cat("  Saved to:", OUT_DIR, "\n")

# ============================================================
# Summary
# ============================================================

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("LAVA Analysis Complete!\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

# Summarize significant bivariate correlations
if (nrow(bivar_df) > 0) {
  sig_bivar <- bivar_df[bivar_df$p < 0.05, ]
  cat("Significant local rg (p < 0.05):", nrow(sig_bivar), "\n")

  # Top 10 by p-value
  cat("\nTop 10 local genetic correlations:\n")
  top10 <- head(bivar_df[order(bivar_df$p), ], 10)
  print(top10[, c("locus", "chr", "phen1", "phen2", "rho", "p")])
}

cat("\nDone!\n")
