# Install LAVA and dependencies
# Run this script once to set up the environment

cat("Installing LAVA dependencies...\n")

# Install required packages
required_packages <- c("devtools", "data.table", "Matrix", "parallel")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat(paste("Installing", pkg, "...\n"))
    install.packages(pkg, repos = "https://cloud.r-project.org/")
  }
}

# Install LAVA from GitHub
cat("\nInstalling LAVA from GitHub...\n")
if (!require("LAVA", quietly = TRUE)) {
  devtools::install_github("josefin-werme/LAVA")
}

# Verify installation
cat("\nVerifying installation...\n")
library(LAVA)
cat(paste("LAVA version:", packageVersion("LAVA"), "\n"))
cat("Installation complete!\n")
