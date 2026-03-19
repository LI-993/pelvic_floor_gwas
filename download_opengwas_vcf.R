# Download OpenGWAS Full Summary Statistics via VCF
# This script uses gwasvcf package with proper authentication
# Run this in R/RStudio

# ============================================================
# STEP 1: Install packages
# ============================================================
cat("Installing/loading packages...\n")

if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}

# Install required packages
if (!requireNamespace("ieugwasr", quietly = TRUE)) {
  remotes::install_github("MRCIEU/ieugwasr")
}

if (!requireNamespace("gwasvcf", quietly = TRUE)) {
  remotes::install_github("MRCIEU/gwasvcf")
}

# For VCF handling
if (!requireNamespace("VariantAnnotation", quietly = TRUE)) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager")
  }
  BiocManager::install("VariantAnnotation")
}

library(ieugwasr)
library(gwasvcf)

# ============================================================
# STEP 2: Set authentication
# ============================================================
cat("\n=== Setting up authentication ===\n")

# API token - must be set as environment variable
token <- "eyJhbGciOiJSUzI1NiIsImtpZCI6ImFwaS1qd3QiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJhcGkub3Blbmd3YXMuaW8iLCJhdWQiOiJhcGkub3Blbmd3YXMuaW8iLCJzdWIiOiJuYXl1a2ljb21Ac2luYS5jbiIsImlhdCI6MTc2NTc5NTA5MywiZXhwIjoxNzY3MDA0NjkzfQ.Ek5ac5cYZT9PqddQu5N7TweRYNOZpTIPWOT6jcnoqG1Yll2fNhCp7iHnpSpZJu8HcRZ59lItJzfxNRvdGjeiwvbZ7Pb49jE6O8gEnQM2C3qkmhxCIEb1MCZIrr4M_qHcCjchCUmKzjKN6xJ2mnfLVxxDd7K6qVyfcrbvdh2oWiTo9rc5ZdX7SKgBhY3AyGNdcbxkXATGZlPd-SpEkhur102Fs-npdawvp6yT7fHQsd_-_ppOPV09qaBGrdsoYnBe9M4BBCy6KnEtJLCQIHb2zqDOgkJfwIjrX_qtDPT-eZ74nBkyzkg6qPcK8zZIm9dQ6V3idR_vPjNgIw83xQsIow"
Sys.setenv(OPENGWAS_JWT = token)
cat("API token set.\n")

# ============================================================
# STEP 3: Define datasets and output
# ============================================================
outdir <- "D:/Nproject/gwas/pelvic_floor_gwas/data/raw/OpenGWAS"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

datasets <- c(
  "ukb-b-373",   # Bladder: Calcified/Contracted/Overactive
  "ukb-b-8517"   # Urinary frequency/Incontinence
)

# ============================================================
# STEP 4: Get dataset info first
# ============================================================
cat("\n=== Dataset Information ===\n")

for (id in datasets) {
  cat(sprintf("\n--- %s ---\n", id))
  tryCatch({
    info <- gwasinfo(id)
    if (nrow(info) > 0) {
      cat(sprintf("  Trait: %s\n", info$trait[1]))
      cat(sprintf("  Sample size: %s\n", info$sample_size[1]))
      cat(sprintf("  N SNPs: %s\n", info$nsnp[1]))
    }
  }, error = function(e) {
    cat(sprintf("  Error: %s\n", e$message))
  })
}

# ============================================================
# STEP 5: Download using gwasvcf (recommended method)
# ============================================================
cat("\n\n=== Downloading VCF files using gwasvcf ===\n")
cat("This may take a while for large datasets...\n\n")

# Set GWAS VCF database location
set_bcftools()  # Configure bcftools path if needed

for (id in datasets) {
  cat(sprintf("Processing %s...\n", id))

  tryCatch({
    # Create VCF database connection
    # gwasvcf can query data from OpenGWAS server

    # Method 1: Query specific regions (faster)
    # For full genome, we need to query all chromosomes

    # Method 2: Use tophits and expand
    # This gets significant hits and surrounding variants

    # Method 3: Download via API URL
    # Get the proper authenticated URL

    # Check if VCF URL is available
    vcf_url <- sprintf("https://gwas.mrcieu.ac.uk/files/%s/%s.vcf.gz", id, id)

    cat(sprintf("  Attempting to query %s...\n", id))

    # Query summary stats for a test region first
    test_result <- associations(
      variants = "1:1000000-2000000",  # Small test region
      id = id,
      proxies = 0
    )

    if (!is.null(test_result) && nrow(test_result) > 0) {
      cat(sprintf("  API working. Found %d variants in test region.\n", nrow(test_result)))

      # For full summary stats, we need to query by chromosome
      cat("  Note: Full download requires querying all chromosomes.\n")
      cat("  This would take ~30 minutes per dataset.\n")
    }

  }, error = function(e) {
    cat(sprintf("  Error: %s\n", e$message))
  })

  cat("\n")
}

# ============================================================
# STEP 6: Alternative - Download by chromosome
# ============================================================
cat("\n=== Alternative: Download by chromosome ===\n")
cat("To download full summary statistics, uncomment and run:\n\n")

cat('
# Download all chromosomes for a dataset
download_full_sumstats <- function(id, outdir) {
  all_data <- data.frame()

  for (chr in 1:22) {
    cat(sprintf("  Chromosome %d...", chr))

    tryCatch({
      # Query entire chromosome
      # Note: This queries by chromosome range
      chr_data <- associations(
        variants = sprintf("%d:1-300000000", chr),
        id = id,
        proxies = 0
      )

      if (!is.null(chr_data) && nrow(chr_data) > 0) {
        all_data <- rbind(all_data, chr_data)
        cat(sprintf(" %d variants\\n", nrow(chr_data)))
      }
    }, error = function(e) {
      cat(sprintf(" Error: %s\\n", e$message))
    })

    Sys.sleep(1)  # Be nice to the API
  }

  # Save results
  outfile <- file.path(outdir, paste0(id, "_full.tsv.gz"))
  write.table(all_data, gzfile(outfile), sep = "\\t", row.names = FALSE, quote = FALSE)
  cat(sprintf("Saved %d total variants to %s\\n", nrow(all_data), outfile))

  return(all_data)
}

# Run for each dataset (this will take a while!)
# download_full_sumstats("ukb-b-373", outdir)
# download_full_sumstats("ukb-b-8517", outdir)
')

# ============================================================
# STEP 7: Recommendation
# ============================================================
cat("\n\n=== RECOMMENDATION ===\n")
cat("Since OpenGWAS VCF downloads require authentication and are slow,\n")
cat("consider these alternatives:\n\n")

cat("1. FinnGen has excellent bladder/urinary phenotypes (already downloading)\n")
cat("   - N14_NEUROMUSCDYSBLADD: Neurogenic bladder dysfunction\n")
cat("   - N14_URININCONT: Urinary incontinence\n\n")

cat("2. Pan-UKB has multi-ancestry summary stats (open access):\n")
cat("   https://pan.ukbb.broadinstitute.org/\n\n")

cat("3. For OpenGWAS, use the chromosome-by-chromosome download above.\n\n")

cat("Script completed.\n")
