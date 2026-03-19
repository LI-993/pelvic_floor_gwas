# Test LAVA - find working loci
library(LAVA)

BASE_DIR <- "D:/Nproject/gwas/pelvic_floor_gwas"
REF_PREFIX <- file.path(BASE_DIR, "reference/lava/g1000_eur")
INPUT_INFO <- file.path(BASE_DIR, "data/lava/input.info.txt")
SAMPLE_OVERLAP <- file.path(BASE_DIR, "data/lava/sample.overlap.txt")
LOCUS_FILE <- file.path(BASE_DIR, "LAVA-main/support_data/blocks_s2500_m25_f1_w200.GRCh37_hg19.locfile")

cat("Loading input data...\n")
input <- process.input(
  input.info.file = INPUT_INFO,
  sample.overlap.file = SAMPLE_OVERLAP,
  ref.prefix = REF_PREFIX,
  phenos = c("POP", "FemaleProlapse")
)

cat("Reading loci...\n")
loci <- read.loci(LOCUS_FILE)
cat("Total loci:", nrow(loci), "\n")

# Try 100 random loci to find working ones
set.seed(42)
test_indices <- sample(1:nrow(loci), 100)

univ_results <- list()
bivar_results <- list()

cat("\nTesting 100 random loci...\n")
success_count <- 0
for (i in test_indices) {
  locus <- tryCatch(
    process.locus(loci[i,], input),
    error = function(e) NULL
  )

  if (!is.null(locus)) {
    success_count <- success_count + 1
    cat("Locus", locus$id, "(", success_count, "): chr", locus$chr, ", SNPs:", locus$n.snps, "\n")

    # Run tests
    loc_out <- tryCatch(
      run.univ.bivar(locus, univ.thresh = 0.05),
      error = function(e) NULL
    )

    if (!is.null(loc_out)) {
      loc_info <- data.frame(
        locus = locus$id,
        chr = locus$chr,
        start = locus$start,
        stop = locus$stop,
        n.snps = locus$n.snps
      )
      if (!is.null(loc_out$univ)) {
        univ_results[[length(univ_results) + 1]] <- cbind(loc_info, loc_out$univ)
      }
      if (!is.null(loc_out$bivar)) {
        bivar_results[[length(bivar_results) + 1]] <- cbind(loc_info, loc_out$bivar)
      }
    }

    if (success_count >= 10) break  # Stop after 10 successful loci
  }
}

cat("\nSuccessfully processed", success_count, "loci out of 100 tested\n")

if (length(univ_results) > 0) {
  univ_df <- do.call(rbind, univ_results)
  cat("\nUnivariate results:\n")
  print(univ_df)
}

if (length(bivar_results) > 0) {
  bivar_df <- do.call(rbind, bivar_results)
  cat("\nBivariate results:\n")
  print(bivar_df)
}
