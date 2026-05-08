# Test LAVA with minimal setup
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

cat("SNPs in reference:", nrow(input$ref$bim), "\n")
cat("Reading loci...\n")
loci <- read.loci(LOCUS_FILE)
cat("Total loci:", nrow(loci), "\n")

# Test multiple loci
test_loci <- c(50, 100, 200, 300, 500)
for (i in test_loci) {
  cat("\nTesting locus", i, "...\n")
  locus <- tryCatch(
    process.locus(loci[i,], input),
    error = function(e) NULL
  )

  if (!is.null(locus)) {
    cat("Locus", locus$id, ": chr", locus$chr, ":", locus$start, "-", locus$stop, "\n")
    cat("SNPs:", locus$n.snps, ", PCs:", locus$K, "\n")

    cat("\nUnivariate test:\n")
    univ <- run.univ(locus)
    print(univ)

    cat("\nBivariate test:\n")
    bivar <- run.bivar(locus)
    print(bivar)
    break  # Stop after first successful locus
  } else {
    cat("Locus could not be processed\n")
  }
}
