# Download OpenGWAS Data
# This script uses ieugwasr to download GWAS summary statistics
# Run this in R/RStudio

# ============================================================
# STEP 1: Install packages
# ============================================================
cat("Installing/loading packages...\n")

if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}

# Install ieugwasr from GitHub (latest version)
if (!requireNamespace("ieugwasr", quietly = TRUE)) {
  remotes::install_github("MRCIEU/ieugwasr")
}

library(ieugwasr)

# ============================================================
# STEP 2: Check API status and authenticate
# ============================================================
cat("\nChecking API status...\n")

# Check if API is accessible
tryCatch({
  api_status <- api_status()
  cat("API Status: ", api_status$status, "\n")
}, error = function(e) {
  cat("Warning: Could not check API status\n")
  cat("Error: ", e$message, "\n")
})

# ============================================================
# STEP 3: Set up authentication (IMPORTANT!)
# ============================================================
cat("\n=== AUTHENTICATION ===\n")
cat("OpenGWAS now requires authentication for most queries.\n")
cat("Please visit: https://api.opengwas.io/profile/\n")
cat("Get your API token and run:\n")
cat('  Sys.setenv(OPENGWAS_JWT = "your_token_here")\n')
cat("Or permanently in your .Renviron file\n\n")

# Set API token - IMPORTANT: must set as environment variable!
token <- "eyJhbGciOiJSUzI1NiIsImtpZCI6ImFwaS1qd3QiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJhcGkub3Blbmd3YXMuaW8iLCJhdWQiOiJhcGkub3Blbmd3YXMuaW8iLCJzdWIiOiJuYXl1a2ljb21Ac2luYS5jbiIsImlhdCI6MTc2NTc5NTA5MywiZXhwIjoxNzY3MDA0NjkzfQ.Ek5ac5cYZT9PqddQu5N7TweRYNOZpTIPWOT6jcnoqG1Yll2fNhCp7iHnpSpZJu8HcRZ59lItJzfxNRvdGjeiwvbZ7Pb49jE6O8gEnQM2C3qkmhxCIEb1MCZIrr4M_qHcCjchCUmKzjKN6xJ2mnfLVxxDd7K6qVyfcrbvdh2oWiTo9rc5ZdX7SKgBhY3AyGNdcbxkXATGZlPd-SpEkhur102Fs-npdawvp6yT7fHQsd_-_ppOPV09qaBGrdsoYnBe9M4BBCy6KnEtJLCQIHb2zqDOgkJfwIjrX_qtDPT-eZ74nBkyzkg6qPcK8zZIm9dQ6V3idR_vPjNgIw83xQsIow"

# Set the environment variable so ieugwasr can use it
Sys.setenv(OPENGWAS_JWT = token)
cat("API token set in environment.\n\n")

# ============================================================
# STEP 4: Define target datasets
# ============================================================
outdir <- "D:/Nproject/gwas/pelvic_floor_gwas/data/raw/OpenGWAS"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# Target datasets
datasets <- c(
  "ukb-b-373",   # Bladder: Calcified/Contracted/Overactive
  "ukb-b-8517"   # Urinary frequency/Incontinence
)

# ============================================================
# STEP 5: Get dataset info
# ============================================================
cat("Fetching dataset information...\n\n")

dataset_info <- list()
for (id in datasets) {
  cat(sprintf("--- %s ---\n", id))
  tryCatch({
    # Use gwasinfo() to get metadata
    info <- gwasinfo(id)
    if (nrow(info) > 0) {
      cat(sprintf("  Trait: %s\n", info$trait[1]))
      cat(sprintf("  Sample size: %s\n", info$sample_size[1]))
      cat(sprintf("  N SNPs: %s\n", info$nsnp[1]))
      dataset_info[[id]] <- info
    }
  }, error = function(e) {
    cat(sprintf("  Error: %s\n", e$message))
    cat("  Try setting API token first.\n")
  })
  cat("\n")
}

# ============================================================
# STEP 6: Try to get tophits (may be empty for some phenotypes)
# ============================================================
cat("=== Getting top hits (genome-wide significant) ===\n")
cat("Note: Some phenotypes may have NO genome-wide significant hits.\n\n")

for (id in datasets) {
  cat(sprintf("Getting tophits for %s...\n", id))
  tryCatch({
    hits <- tophits(id = id)

    # Handle case where no hits are returned
    if (is.null(hits) || nrow(hits) == 0) {
      cat("  No genome-wide significant hits found (p < 5e-8).\n")
      cat("  This is normal for some phenotypes - will need to download full summary stats.\n")
    } else {
      cat(sprintf("  Found %d genome-wide significant hits\n", nrow(hits)))

      # Save tophits
      outfile <- file.path(outdir, paste0(id, "_tophits.tsv"))
      write.table(hits, outfile, sep = "\t", row.names = FALSE, quote = FALSE)
      cat(sprintf("  Saved to: %s\n", outfile))
    }
  }, error = function(e) {
    # This may happen if no hits exist - the API returns empty result
    if (grepl("length|empty|zero", e$message, ignore.case = TRUE)) {
      cat("  No genome-wide significant hits found.\n")
    } else {
      cat(sprintf("  Error: %s\n", e$message))
    }
  })
  cat("\n")
}

# ============================================================
# STEP 7: Download full summary statistics via VCF
# ============================================================
cat("\n=== Downloading full summary statistics ===\n")
cat("For full summary stats, the best method is to download VCF files.\n\n")

# Install gwasvcf if needed
if (!requireNamespace("gwasvcf", quietly = TRUE)) {
  cat("Installing gwasvcf package...\n")
  tryCatch({
    remotes::install_github("MRCIEU/gwasvcf")
  }, error = function(e) {
    cat("Note: gwasvcf installation may fail. Alternative: download manually.\n")
  })
}

# Alternative: Use direct download URLs
# OpenGWAS VCF files can be accessed via:
# https://gwas.mrcieu.ac.uk/files/{id}/{id}.vcf.gz

cat("\n=== Manual download instructions ===\n")
cat("OpenGWAS VCF files are available at:\n\n")

for (id in datasets) {
  vcf_url <- sprintf("https://gwas.mrcieu.ac.uk/files/%s/%s.vcf.gz", id, id)
  tbi_url <- sprintf("https://gwas.mrcieu.ac.uk/files/%s/%s.vcf.gz.tbi", id, id)
  cat(sprintf("Dataset: %s\n", id))
  cat(sprintf("  VCF: %s\n", vcf_url))
  cat(sprintf("  TBI: %s\n", tbi_url))
  cat("\n")
}

cat("You can download these using wget or curl.\n")
cat("Example:\n")
cat("  wget https://gwas.mrcieu.ac.uk/files/ukb-b-373/ukb-b-373.vcf.gz\n")
cat("  wget https://gwas.mrcieu.ac.uk/files/ukb-b-373/ukb-b-373.vcf.gz.tbi\n")

# ============================================================
# STEP 8: Alternative - Query specific p-value threshold
# ============================================================
cat("\n\n=== Alternative: Query with p-value threshold ===\n")
cat("Warning: Querying all variants is VERY slow and may timeout.\n")
cat("Only use this for specific regions or small datasets.\n\n")

# Example: Get variants with p < 0.001 (still many variants but manageable)
# Uncomment to run:
# for (id in datasets) {
#   cat(sprintf("Querying %s with p < 0.001...\n", id))
#   tryCatch({
#     result <- associations(id = id, pval = 0.001)
#     if (!is.null(result) && nrow(result) > 0) {
#       outfile <- file.path(outdir, paste0(id, "_p0.001.tsv"))
#       write.table(result, outfile, sep = "\t", row.names = FALSE, quote = FALSE)
#       cat(sprintf("  Saved %d variants to: %s\n", nrow(result), outfile))
#     }
#   }, error = function(e) {
#     cat(sprintf("  Error: %s\n", e$message))
#   })
# }

cat("\nScript completed.\n")
cat(sprintf("Output directory: %s\n", outdir))
cat("\nRecommendation: Download VCF files manually using wget/curl.\n")
