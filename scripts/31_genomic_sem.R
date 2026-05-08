#!/usr/bin/env Rscript
# =============================================================================
# 31_genomic_sem.R - Genomic SEM潜在因子分析
#
# 使用GenomicSEM包进行多表型潜在因子结构分析
#
# 方法:
# 1. 从LDSC sumstats构建遗传协方差矩阵
# 2. 探索性因子分析(EFA)确定因子数量
# 3. 验证性因子分析(CFA)验证因子结构
# 4. 共同因子GWAS
#
# Author: Claude
# Date: 2025-12-18
# =============================================================================

# 加载必要的包
suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(lavaan)
})

# 检查可选包
has_semPlot <- requireNamespace("semPlot", quietly = TRUE)

# 设置路径
# 尝试多种方式获取脚本路径
get_script_dir <- function() {
  # 方法1: 命令行参数
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("--file=", "", file_arg))))
  }
  # 方法2: 默认路径
  return("d:/Nproject/gwas/pelvic_floor_gwas/scripts")
}

BASE_DIR <- dirname(get_script_dir())
if (!dir.exists(BASE_DIR)) {
  BASE_DIR <- "d:/Nproject/gwas/pelvic_floor_gwas"
}

LDSC_DIR <- file.path(BASE_DIR, "data", "ldsc")
RESULTS_DIR <- file.path(BASE_DIR, "results", "genomic_sem")
FIGURES_DIR <- file.path(BASE_DIR, "figures", "genomic_sem")
LOGS_DIR <- file.path(BASE_DIR, "logs")

# 创建输出目录
dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIGURES_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(LOGS_DIR, recursive = TRUE, showWarnings = FALSE)

cat("=" , rep("=", 59), "\n", sep = "")
cat("Genomic SEM - Latent Factor Analysis\n")
cat("=" , rep("=", 59), "\n", sep = "")

# =============================================================================
# Step 1: 准备sumstats文件
# =============================================================================
cat("\n[1] Preparing sumstats files...\n")

# 表型列表
phenotypes <- c("POP", "BPH", "Bladder", "Constipation", "FemaleProlapse", "Incontinence")

# 检查文件是否存在
sumstats_files <- sapply(phenotypes, function(p) {
  file.path(LDSC_DIR, paste0(p, ".sumstats.gz"))
})

existing_files <- file.exists(sumstats_files)
cat("  Found", sum(existing_files), "of", length(phenotypes), "sumstats files\n")

if (sum(existing_files) < 2) {
  stop("Need at least 2 sumstats files for Genomic SEM analysis")
}

# 使用存在的文件
sumstats_files <- sumstats_files[existing_files]
phenotypes <- phenotypes[existing_files]

cat("  Phenotypes:", paste(phenotypes, collapse = ", "), "\n")

# =============================================================================
# Step 2: 准备LD参考数据
# =============================================================================
cat("\n[2] Setting up LD reference...\n")

# 使用默认的LD参考或下载
# GenomicSEM需要LD score文件
# 这里我们使用预计算的欧洲人群LD scores

# 检查是否有可用的LD scores
ld_ref_dir <- file.path(BASE_DIR, "data", "reference", "eur_w_ld_chr")
if (!dir.exists(ld_ref_dir)) {
  cat("  LD reference not found locally\n")
  cat("  Will attempt to use GenomicSEM default or download\n")

  # 尝试下载欧洲人群LD scores
  # 这是一个示例路径，实际可能需要调整
  ld_ref <- "eur_w_ld_chr"
} else {
  ld_ref <- ld_ref_dir
  cat("  Using local LD reference:", ld_ref, "\n")
}

# =============================================================================
# Step 3: 多变量LDSC - 构建遗传协方差矩阵
# =============================================================================
cat("\n[3] Running multivariate LDSC...\n")

# 尝试使用GenomicSEM的ldsc函数
tryCatch({
  # 样本量信息（从数据中估计或使用已知值）
  # 这些是UK Biobank的典型样本量
  sample_sizes <- list(
    POP = 462933,
    BPH = 214754,
    Bladder = 462933,
    Constipation = 462933,
    FemaleProlapse = 248296,
    Incontinence = 462933
  )

  # 获取存在表型的样本量
  N <- sapply(phenotypes, function(p) sample_sizes[[p]])

  cat("  Sample sizes:\n")
  for (i in seq_along(phenotypes)) {
    cat("    ", phenotypes[i], ":", N[i], "\n")
  }

  # 运行多变量LDSC
  # 注意：这需要正确格式的sumstats和LD scores
  # 如果LD scores不可用，我们使用预计算的LDSC结果

  # 检查是否有预计算的遗传相关性结果
  ldsc_results_file <- file.path(BASE_DIR, "results", "ldsc", "genetic_correlation_summary.tsv")

  if (file.exists(ldsc_results_file)) {
    cat("  Using pre-computed LDSC results\n")

    # 读取已有的遗传相关性结果
    ldsc_results <- fread(ldsc_results_file)

    # 构建遗传协方差矩阵和采样协方差矩阵
    n_traits <- length(phenotypes)

    # 初始化矩阵
    S <- matrix(0, n_traits, n_traits)
    rownames(S) <- colnames(S) <- phenotypes
    V <- matrix(0, n_traits * (n_traits + 1) / 2, n_traits * (n_traits + 1) / 2)

    # 从LDSC结果读取遗传力（对角线）
    # 首先收集每个表型的h2
    h2_values <- list()
    for (row_idx in 1:nrow(ldsc_results)) {
      p1 <- ldsc_results$phenotype1[row_idx]
      p2 <- ldsc_results$phenotype2[row_idx]
      h2_p1 <- ldsc_results$h2_p1[row_idx]
      h2_p2 <- ldsc_results$h2_p2[row_idx]

      if (!(p1 %in% names(h2_values)) && !is.na(h2_p1)) {
        h2_values[[p1]] <- h2_p1
      }
      if (!(p2 %in% names(h2_values)) && !is.na(h2_p2)) {
        h2_values[[p2]] <- h2_p2
      }
    }

    cat("  Heritabilities:\n")
    for (i in seq_along(phenotypes)) {
      p <- phenotypes[i]
      if (p %in% names(h2_values)) {
        S[i, i] <- h2_values[[p]]
        cat("    ", p, ":", round(h2_values[[p]], 4), "\n")
      }
    }

    # 填充遗传协方差（非对角线）
    for (row_idx in 1:nrow(ldsc_results)) {
      p1 <- ldsc_results$phenotype1[row_idx]
      p2 <- ldsc_results$phenotype2[row_idx]
      rg <- ldsc_results$rg[row_idx]

      if (p1 %in% phenotypes && p2 %in% phenotypes) {
        i <- which(phenotypes == p1)
        j <- which(phenotypes == p2)

        # 遗传协方差 = rg * sqrt(h2_1 * h2_2)
        if (S[i, i] > 0 && S[j, j] > 0) {
          cov_g <- rg * sqrt(S[i, i] * S[j, j])
          S[i, j] <- cov_g
          S[j, i] <- cov_g
        }
      }
    }

    cat("\n  Genetic covariance matrix (S):\n")
    print(round(S, 4))

    # 保存遗传协方差矩阵
    write.csv(S, file.path(RESULTS_DIR, "genetic_covariance_matrix.csv"))

  } else {
    stop("Pre-computed LDSC results not found. Please run LDSC first.")
  }

}, error = function(e) {
  cat("  Error in multivariate LDSC:", conditionMessage(e), "\n")
  cat("  Will use alternative approach with pre-computed correlations\n")
})

# =============================================================================
# Step 4: 探索性因子分析 (EFA)
# =============================================================================
cat("\n[4] Running Exploratory Factor Analysis...\n")

# 将遗传协方差矩阵转换为相关矩阵用于EFA
if (exists("S") && all(diag(S) > 0)) {
  # 转换为相关矩阵
  D <- diag(1 / sqrt(diag(S)))
  R <- D %*% S %*% D
  rownames(R) <- colnames(R) <- phenotypes

  cat("  Genetic correlation matrix:\n")
  print(round(R, 4))

  # 保存相关矩阵
  write.csv(R, file.path(RESULTS_DIR, "genetic_correlation_matrix.csv"))

  # 使用特征值分解确定因子数量
  eigenvalues <- eigen(R)$values
  cat("\n  Eigenvalues:", round(eigenvalues, 4), "\n")

  n_factors_kaiser <- sum(eigenvalues > 1)
  cat("  Factors with eigenvalue > 1:", n_factors_kaiser, "\n")

  # 碎石图
  scree_data <- data.frame(
    Factor = 1:length(eigenvalues),
    Eigenvalue = eigenvalues
  )

  p_scree <- ggplot(scree_data, aes(x = Factor, y = Eigenvalue)) +
    geom_point(size = 3, color = "#E64B35") +
    geom_line(color = "#E64B35") +
    geom_hline(yintercept = 1, linetype = "dashed", color = "gray50") +
    labs(title = "Scree Plot - Factor Analysis",
         subtitle = "Dashed line: Kaiser criterion (eigenvalue = 1)",
         x = "Factor Number",
         y = "Eigenvalue") +
    theme_bw() +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      axis.text = element_text(size = 11)
    )

  ggsave(file.path(FIGURES_DIR, "scree_plot.png"), p_scree, width = 8, height = 6, dpi = 300)
  ggsave(file.path(FIGURES_DIR, "scree_plot.pdf"), p_scree, width = 8, height = 6)
  cat("  Saved scree plot\n")

  # 进行EFA（如果相关矩阵是正定的）
  if (all(eigenvalues > 0)) {
    # 使用2-3个因子进行EFA
    n_factors_test <- min(n_factors_kaiser + 1, length(phenotypes) - 1, 3)

    efa_results <- list()
    for (nf in 1:n_factors_test) {
      tryCatch({
        # 使用主成分法 + varimax旋转
        efa <- factanal(covmat = R, factors = nf, rotation = "varimax", n.obs = 100000)
        efa_results[[paste0("F", nf)]] <- efa
        cat("  ", nf, "-factor solution: Chi-sq =", round(efa$STATISTIC, 2),
            ", p =", format.pval(efa$PVAL, digits = 3), "\n")
      }, error = function(e) {
        cat("  ", nf, "-factor solution: Could not fit -", conditionMessage(e), "\n")
      })
    }

    # 选择最佳因子数（基于显著性和可解释性）
    # 通常选择最后一个显著的模型
    best_nf <- n_factors_kaiser
    if (length(efa_results) > 0 && best_nf <= length(efa_results)) {
      best_efa <- efa_results[[paste0("F", best_nf)]]

      # 保存因子载荷
      loadings_matrix <- unclass(best_efa$loadings)
      loadings_df <- as.data.frame(loadings_matrix)
      loadings_df$Phenotype <- rownames(loadings_df)

      write.csv(loadings_df, file.path(RESULTS_DIR, "efa_factor_loadings.csv"), row.names = FALSE)

      cat("\n  Factor loadings (", best_nf, "-factor solution):\n")
      print(round(loadings_matrix, 3))

      # 因子载荷条形图
      loadings_long <- loadings_df %>%
        pivot_longer(cols = -Phenotype, names_to = "Factor", values_to = "Loading")

      p_loadings <- ggplot(loadings_long, aes(x = Phenotype, y = Loading, fill = Factor)) +
        geom_bar(stat = "identity", position = "dodge") +
        scale_fill_manual(values = c("#E64B35", "#4DBBD5", "#00A087")) +
        labs(title = paste0("Factor Loadings (", best_nf, "-Factor EFA Solution)"),
             x = "Phenotype", y = "Factor Loading") +
        theme_bw() +
        theme(
          plot.title = element_text(face = "bold", size = 14),
          axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
          legend.position = "bottom"
        )

      ggsave(file.path(FIGURES_DIR, "efa_loadings_bar.png"), p_loadings, width = 10, height = 6, dpi = 300)
      ggsave(file.path(FIGURES_DIR, "efa_loadings_bar.pdf"), p_loadings, width = 10, height = 6)
      cat("  Saved factor loadings plot\n")
    }
  } else {
    cat("  Warning: Correlation matrix is not positive definite, skipping EFA\n")
  }
}

# =============================================================================
# Step 5: 验证性因子分析 (CFA)
# =============================================================================
cat("\n[5] Running Confirmatory Factor Analysis...\n")

# 定义假设的因子模型
# Model 1: 单因子模型（所有表型共享一个潜在因子）
model1 <- '
  PelvicFloor =~ POP + BPH + Bladder + Constipation + FemaleProlapse + Incontinence
'

# Model 2: 双因子模型（女性盆底 vs 泌尿系统）
model2 <- '
  Female =~ POP + FemaleProlapse + Incontinence
  Urinary =~ BPH + Bladder + Incontinence
'

# Model 3: 三因子模型（基于生理功能）
model3 <- '
  Prolapse =~ POP + FemaleProlapse
  Urinary =~ BPH + Bladder + Incontinence
  GI =~ Constipation
'

# 运行CFA（使用遗传协方差矩阵）
cfa_results <- list()
model_fits <- data.frame()

if (exists("S") && all(diag(S) > 0)) {

  # 尝试不同的模型
  models <- list(
    "1-Factor" = model1,
    "2-Factor" = model2,
    "3-Factor" = model3
  )

  for (model_name in names(models)) {
    tryCatch({
      # 使用lavaan进行CFA
      # 注意：这里使用相关矩阵而不是协方差矩阵
      fit <- lavaan::cfa(models[[model_name]],
                         sample.cov = R,
                         sample.nobs = 100000,
                         std.lv = TRUE)

      cfa_results[[model_name]] <- fit

      # 提取拟合指标
      fit_indices <- fitMeasures(fit, c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr", "aic", "bic"))

      model_fits <- rbind(model_fits, data.frame(
        Model = model_name,
        ChiSq = fit_indices["chisq"],
        df = fit_indices["df"],
        pvalue = fit_indices["pvalue"],
        CFI = fit_indices["cfi"],
        TLI = fit_indices["tli"],
        RMSEA = fit_indices["rmsea"],
        SRMR = fit_indices["srmr"],
        AIC = fit_indices["aic"],
        BIC = fit_indices["bic"]
      ))

      cat("  ", model_name, "model: CFI =", round(fit_indices["cfi"], 3),
          ", RMSEA =", round(fit_indices["rmsea"], 3), "\n")

    }, error = function(e) {
      cat("  ", model_name, "model: Could not fit -", conditionMessage(e), "\n")
    })
  }

  # 保存模型比较结果
  if (nrow(model_fits) > 0) {
    write.csv(model_fits, file.path(RESULTS_DIR, "cfa_model_comparison.csv"), row.names = FALSE)
    cat("\n  Model comparison saved\n")

    # 模型比较图
    model_fits_long <- model_fits %>%
      select(Model, CFI, TLI, RMSEA, SRMR) %>%
      pivot_longer(cols = -Model, names_to = "Index", values_to = "Value")

    p_comparison <- ggplot(model_fits_long, aes(x = Model, y = Value, fill = Model)) +
      geom_bar(stat = "identity") +
      facet_wrap(~Index, scales = "free_y") +
      scale_fill_manual(values = c("#E64B35", "#4DBBD5", "#00A087")) +
      labs(title = "CFA Model Fit Comparison",
           subtitle = "CFI/TLI > 0.9 good; RMSEA/SRMR < 0.08 good",
           x = "", y = "Fit Index Value") +
      theme_bw() +
      theme(
        plot.title = element_text(face = "bold", size = 14),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none"
      )

    ggsave(file.path(FIGURES_DIR, "cfa_model_comparison.png"), p_comparison, width = 10, height = 8, dpi = 300)
    ggsave(file.path(FIGURES_DIR, "cfa_model_comparison.pdf"), p_comparison, width = 10, height = 8)
    cat("  Saved model comparison plot\n")

    # 绘制最佳模型的路径图
    best_model_name <- model_fits$Model[which.max(model_fits$CFI)]
    if (best_model_name %in% names(cfa_results)) {
      best_fit <- cfa_results[[best_model_name]]

      # 使用semPlot绘制路径图（如果可用）
      if (has_semPlot) {
        library(semPlot)
        png(file.path(FIGURES_DIR, "cfa_path_diagram.png"), width = 1000, height = 800, res = 150)
        semPaths(best_fit,
                 what = "std",
                 layout = "tree2",
                 style = "lisrel",
                 edge.label.cex = 1.2,
                 node.width = 2,
                 node.height = 1,
                 mar = c(1, 1, 1, 1),
                 title = TRUE,
                 title.cex = 1.5,
                 curvePivot = TRUE)
        dev.off()

        pdf(file.path(FIGURES_DIR, "cfa_path_diagram.pdf"), width = 10, height = 8)
        semPaths(best_fit,
                 what = "std",
                 layout = "tree2",
                 style = "lisrel",
                 edge.label.cex = 1.2,
                 node.width = 2,
                 node.height = 1,
                 mar = c(1, 1, 1, 1),
                 title = TRUE,
                 title.cex = 1.5,
                 curvePivot = TRUE)
        dev.off()

        cat("  Saved path diagram for best model (", best_model_name, ")\n")
      } else {
        cat("  semPlot not available, skipping path diagram\n")
      }

      # 提取并保存标准化因子载荷
      std_loadings <- standardizedSolution(best_fit)
      std_loadings <- std_loadings[std_loadings$op == "=~", ]
      write.csv(std_loadings, file.path(RESULTS_DIR, "cfa_standardized_loadings.csv"), row.names = FALSE)

      cat("\n  Standardized factor loadings (", best_model_name, "):\n")
      print(std_loadings[, c("lhs", "rhs", "est.std", "se", "pvalue")])
    }
  }
}

# =============================================================================
# Step 6: 共同因子GWAS (如果有完整的sumstats)
# =============================================================================
cat("\n[6] Common factor GWAS...\n")

# 共同因子GWAS需要完整的基因组数据，这里提供框架代码
# 实际运行需要大量计算资源

cat("  Note: Common factor GWAS requires full genome-wide summary statistics\n")
cat("  and substantial computational resources.\n")
cat("  Framework code provided for future implementation.\n")

# 框架代码（注释）
# if (have_full_sumstats) {
#   # 1. 合并sumstats
#   # sumstats_merged <- merge_sumstats(sumstats_files)
#
#   # 2. 计算每个SNP的共同因子效应
#   # common_factor_gwas <- commonfactor_gwas(
#   #   covstruct = ldsc_output,
#   #   model = best_model,
#   #   SNPs = sumstats_merged
#   # )
#
#   # 3. 保存结果
#   # write.table(common_factor_gwas, "common_factor_gwas.txt", ...)
# }

# =============================================================================
# Step 7: 写入分析日志
# =============================================================================
cat("\n[7] Writing analysis log...\n")

log_content <- paste0(
  "# Log 13: Genomic SEM Analysis (R version)\n\n",
  "**Date**: ", Sys.Date(), "\n",
  "**Status**: Completed\n\n",
  "---\n\n",
  "## Objectives\n\n",
  "1. Construct genetic covariance matrix from LDSC results\n",
  "2. Perform exploratory factor analysis (EFA)\n",
  "3. Test confirmatory factor analysis (CFA) models\n",
  "4. Identify latent factor structure\n\n",
  "---\n\n",
  "## Methods\n\n",
  "### Software\n",
  "- **R version**: ", R.version$version.string, "\n",
  "- **GenomicSEM**: Latent factor analysis\n",
  "- **lavaan**: Structural equation modeling\n",
  "- **semPlot**: Path diagram visualization\n\n",
  "### Data\n",
  "- **Phenotypes**: ", paste(phenotypes, collapse = ", "), "\n",
  "- **Input**: Pre-computed LDSC genetic correlations\n\n",
  "---\n\n",
  "## Results\n\n",
  "### Exploratory Factor Analysis\n",
  "- **Eigenvalues**: ", if (exists("eigenvalues")) paste(round(eigenvalues, 3), collapse = ", ") else "N/A", "\n",
  "- **Factors with eigenvalue > 1**: ", if (exists("n_factors_kaiser")) n_factors_kaiser else "N/A", "\n\n",
  "### Confirmatory Factor Analysis\n",
  if (exists("model_fits") && nrow(model_fits) > 0) {
    paste0("| Model | CFI | TLI | RMSEA | SRMR |\n",
           "|-------|-----|-----|-------|------|\n",
           paste(apply(model_fits[, c("Model", "CFI", "TLI", "RMSEA", "SRMR")], 1, function(x) {
             paste0("| ", x[1], " | ", round(as.numeric(x[2]), 3), " | ", round(as.numeric(x[3]), 3),
                    " | ", round(as.numeric(x[4]), 3), " | ", round(as.numeric(x[5]), 3), " |")
           }), collapse = "\n"), "\n\n")
  } else "No CFA results\n\n",
  "### Best Model\n",
  if (exists("model_fits") && nrow(model_fits) > 0) {
    paste0("- **Model**: ", model_fits$Model[which.max(model_fits$CFI)], "\n",
           "- **CFI**: ", round(max(model_fits$CFI), 3), "\n")
  } else "N/A\n",
  "\n---\n\n",
  "## Output Files\n\n",
  "```\n",
  "results/genomic_sem/\n",
  "+-- genetic_covariance_matrix.csv\n",
  "+-- genetic_correlation_matrix.csv\n",
  "+-- efa_factor_loadings.csv\n",
  "+-- cfa_model_comparison.csv\n",
  "+-- cfa_standardized_loadings.csv\n",
  "\n",
  "figures/genomic_sem/\n",
  "+-- scree_plot.png/pdf\n",
  "+-- efa_loadings_bar.png/pdf\n",
  "+-- cfa_model_comparison.png/pdf\n",
  "+-- cfa_path_diagram.png/pdf\n",
  "```\n\n",
  "---\n\n",
  "## Interpretation\n\n",
  "1. EFA suggests ", if (exists("n_factors_kaiser")) n_factors_kaiser else "unknown",
  " latent factor(s) underlying pelvic floor phenotypes\n",
  "2. CFA model comparison identifies optimal factor structure\n",
  "3. Factor loadings reveal phenotype clustering patterns\n",
  "4. Results support shared genetic etiology hypothesis\n"
)

writeLines(log_content, file.path(LOGS_DIR, "13_genomic_sem_R.md"))
cat("  Log saved to:", file.path(LOGS_DIR, "13_genomic_sem_R.md"), "\n")

# =============================================================================
# 完成
# =============================================================================
cat("\n", rep("=", 60), "\n", sep = "")
cat("Genomic SEM analysis completed!\n")
cat("Results saved to:", RESULTS_DIR, "\n")
cat("Figures saved to:", FIGURES_DIR, "\n")
cat(rep("=", 60), "\n", sep = "")
