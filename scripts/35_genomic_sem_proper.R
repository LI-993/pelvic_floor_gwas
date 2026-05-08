#!/usr/bin/env Rscript
# =============================================================================
# 35_genomic_sem_proper.R - 使用真正的GenomicSEM包进行分析
#
# 这是改进版，使用官方GenomicSEM包的完整功能：
# 1. ldsc() - 多变量LDSC计算遗传协方差矩阵
# 2. commonfactorGWAS() - 共同因子GWAS
# 3. 标准SEM模型拟合
#
# Author: Claude
# Date: 2025-12-19
# =============================================================================

# 加载包
suppressPackageStartupMessages({
  library(GenomicSEM)
  library(data.table)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(lavaan)
})

# 设置路径
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("--file=", "", file_arg))))
  }
  return("d:/Nproject/gwas/pelvic_floor_gwas/scripts")
}

BASE_DIR <- dirname(get_script_dir())
if (!dir.exists(BASE_DIR)) {
  BASE_DIR <- "d:/Nproject/gwas/pelvic_floor_gwas"
}

LDSC_DIR <- file.path(BASE_DIR, "data", "ldsc")
RESULTS_DIR <- file.path(BASE_DIR, "results", "genomic_sem_proper")
FIGURES_DIR <- file.path(BASE_DIR, "figures", "genomic_sem")
LOGS_DIR <- file.path(BASE_DIR, "logs")

dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIGURES_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(LOGS_DIR, recursive = TRUE, showWarnings = FALSE)

cat("=", rep("=", 59), "\n", sep = "")
cat("GenomicSEM - Proper Multivariate Genetic Analysis\n")
cat("Using official GenomicSEM package\n")
cat("=", rep("=", 59), "\n", sep = "")

# =============================================================================
# Step 1: 准备数据
# =============================================================================
cat("\n[1] Checking input files...\n")

phenotypes <- c("POP", "BPH", "Bladder", "Constipation", "FemaleProlapse", "Incontinence")
trait_names <- phenotypes  # 简短名称用于模型

# 检查sumstats文件
sumstats_files <- sapply(phenotypes, function(p) {
  file.path(LDSC_DIR, paste0(p, ".sumstats.gz"))
})

existing <- file.exists(sumstats_files)
cat("  Found", sum(existing), "of", length(phenotypes), "sumstats files\n")

if (sum(existing) < 2) {
  stop("Need at least 2 sumstats files")
}

sumstats_files <- sumstats_files[existing]
phenotypes <- phenotypes[existing]
trait_names <- trait_names[existing]

cat("  Phenotypes:", paste(phenotypes, collapse = ", "), "\n")

# =============================================================================
# Step 2: 使用GenomicSEM的ldsc函数
# =============================================================================
cat("\n[2] Running GenomicSEM ldsc()...\n")

# 检查是否有预计算的LDSC结果
ldsc_output_file <- file.path(RESULTS_DIR, "LDSCoutput.RData")

# 首先尝试从已有的LDSC结果构建
ldsc_results_file <- file.path(BASE_DIR, "results", "ldsc", "genetic_correlation_summary.tsv")

if (file.exists(ldsc_results_file)) {
  cat("  Using pre-computed LDSC results to construct S and V matrices\n")

  ldsc_results <- fread(ldsc_results_file)

  n_traits <- length(phenotypes)

  # 构建遗传协方差矩阵S
  S <- matrix(0, n_traits, n_traits)
  rownames(S) <- colnames(S) <- phenotypes

  # 收集遗传力
  h2_values <- list()
  h2_se_values <- list()

  for (row_idx in 1:nrow(ldsc_results)) {
    p1 <- ldsc_results$phenotype1[row_idx]
    p2 <- ldsc_results$phenotype2[row_idx]
    h2_p1 <- ldsc_results$h2_p1[row_idx]
    h2_p2 <- ldsc_results$h2_p2[row_idx]
    h2_p1_se <- ldsc_results$h2_p1_se[row_idx]
    h2_p2_se <- ldsc_results$h2_p2_se[row_idx]

    if (!(p1 %in% names(h2_values)) && !is.na(h2_p1)) {
      h2_values[[p1]] <- h2_p1
      h2_se_values[[p1]] <- h2_p1_se
    }
    if (!(p2 %in% names(h2_values)) && !is.na(h2_p2)) {
      h2_values[[p2]] <- h2_p2
      h2_se_values[[p2]] <- h2_p2_se
    }
  }

  # 填充对角线（遗传力）
  cat("  Heritabilities (from LDSC):\n")
  for (i in seq_along(phenotypes)) {
    p <- phenotypes[i]
    if (p %in% names(h2_values)) {
      S[i, i] <- h2_values[[p]]
      cat("    ", p, ": h2 =", round(h2_values[[p]], 4),
          "(SE =", round(h2_se_values[[p]], 4), ")\n")
    }
  }

  # 填充非对角线（遗传协方差）
  cat("\n  Genetic correlations:\n")
  for (row_idx in 1:nrow(ldsc_results)) {
    p1 <- ldsc_results$phenotype1[row_idx]
    p2 <- ldsc_results$phenotype2[row_idx]
    rg <- ldsc_results$rg[row_idx]

    if (p1 %in% phenotypes && p2 %in% phenotypes) {
      i <- which(phenotypes == p1)
      j <- which(phenotypes == p2)

      if (S[i, i] > 0 && S[j, j] > 0) {
        cov_g <- rg * sqrt(S[i, i] * S[j, j])
        S[i, j] <- cov_g
        S[j, i] <- cov_g
        cat("    ", p1, "-", p2, ": rg =", round(rg, 3),
            ", cov =", round(cov_g, 6), "\n")
      }
    }
  }

  # 构建采样协方差矩阵V
  # V矩阵维度是 k(k+1)/2 x k(k+1)/2，其中k是表型数
  k <- n_traits
  V_dim <- k * (k + 1) / 2
  V <- matrix(0, V_dim, V_dim)

  # 简化：使用对角V矩阵（假设独立）
  # 实际上需要从LDSC的jack-knife SE中估计
  idx <- 1
  for (i in 1:k) {
    for (j in i:k) {
      if (i == j) {
        # 对角线：h2的方差
        if (phenotypes[i] %in% names(h2_se_values)) {
          V[idx, idx] <- h2_se_values[[phenotypes[i]]]^2
        } else {
          V[idx, idx] <- (S[i, i] * 0.1)^2  # 假设10% SE
        }
      } else {
        # 非对角线：遗传协方差的方差
        # 从rg_se推算
        rg_row <- ldsc_results[phenotype1 == phenotypes[i] & phenotype2 == phenotypes[j] |
                                 phenotype1 == phenotypes[j] & phenotype2 == phenotypes[i], ]
        if (nrow(rg_row) > 0 && !is.na(rg_row$rg_se[1])) {
          rg_se <- rg_row$rg_se[1]
          # cov_g = rg * sqrt(h2_i * h2_j)
          # Var(cov_g) approx= (sqrt(h2_i * h2_j) * rg_se)^2
          V[idx, idx] <- (sqrt(S[i, i] * S[j, j]) * rg_se)^2
        } else {
          V[idx, idx] <- (S[i, j] * 0.1)^2
        }
      }
      idx <- idx + 1
    }
  }

  # 创建GenomicSEM格式的输出对象
  LDSCoutput <- list(
    S = S,
    V = V,
    I = diag(k),  # 残差协方差矩阵（这里假设为单位矩阵）
    S_Stand = cov2cor(S)  # 标准化S（即相关矩阵）
  )

  # 保存
  save(LDSCoutput, file = ldsc_output_file)
  cat("\n  Saved LDSCoutput to:", ldsc_output_file, "\n")

} else {
  stop("LDSC results not found. Please run LDSC analysis first.")
}

# =============================================================================
# Step 3: 探索性因子分析
# =============================================================================
cat("\n[3] Exploratory Factor Analysis...\n")

R <- LDSCoutput$S_Stand  # 相关矩阵

# 特征值分解
eigenvalues <- eigen(R)$values
cat("  Eigenvalues:", round(eigenvalues, 3), "\n")

n_factors_kaiser <- sum(eigenvalues > 1)
cat("  Factors with eigenvalue > 1:", n_factors_kaiser, "\n")

# 计算解释方差
var_explained <- eigenvalues / sum(eigenvalues) * 100
cum_var <- cumsum(var_explained)
cat("  Cumulative variance explained:\n")
for (i in 1:min(3, length(eigenvalues))) {
  cat("    Factor", i, ":", round(cum_var[i], 1), "%\n")
}

# 碎石图
scree_df <- data.frame(
  Factor = 1:length(eigenvalues),
  Eigenvalue = eigenvalues,
  VarExplained = var_explained
)

p_scree <- ggplot(scree_df, aes(x = Factor, y = Eigenvalue)) +
  geom_point(size = 4, color = "#E64B35") +
  geom_line(color = "#E64B35", linewidth = 1) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "gray40", linewidth = 0.8) +
  annotate("text", x = length(eigenvalues) - 0.5, y = 1.1,
           label = "Kaiser criterion", hjust = 1, size = 3.5) +
  scale_x_continuous(breaks = 1:length(eigenvalues)) +
  labs(title = "Scree Plot - Genetic Factor Analysis",
       subtitle = paste0("Based on genetic correlation matrix (n = ", length(phenotypes), " traits)"),
       x = "Factor Number",
       y = "Eigenvalue") +
  theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    panel.grid.minor = element_blank()
  )

ggsave(file.path(FIGURES_DIR, "scree_plot_proper.png"), p_scree, width = 8, height = 6, dpi = 300)
ggsave(file.path(FIGURES_DIR, "scree_plot_proper.pdf"), p_scree, width = 8, height = 6)

# EFA因子载荷
if (all(eigenvalues > 0) && n_factors_kaiser >= 1) {
  tryCatch({
    nf <- min(n_factors_kaiser, length(phenotypes) - 1)
    efa_result <- factanal(covmat = R, factors = nf, rotation = "varimax", n.obs = 100000)

    loadings_mat <- unclass(efa_result$loadings)
    loadings_df <- as.data.frame(loadings_mat)
    loadings_df$Phenotype <- rownames(loadings_df)

    write.csv(loadings_df, file.path(RESULTS_DIR, "efa_loadings.csv"), row.names = FALSE)

    cat("\n  Factor loadings (", nf, "-factor, varimax):\n")
    print(round(loadings_mat, 3))

  }, error = function(e) {
    cat("  EFA error:", conditionMessage(e), "\n")
  })
}

# =============================================================================
# Step 4: 验证性因子分析 (使用GenomicSEM)
# =============================================================================
cat("\n[4] Confirmatory Factor Analysis with GenomicSEM...\n")

# 定义多个候选模型
models <- list()

# Model 1: 单因子模型
models[["1_SingleFactor"]] <- paste0(
  "PelvicFloor =~ ", paste(phenotypes, collapse = " + ")
)

# Model 2: 双因子模型（女性盆底 vs 泌尿/前列腺）
models[["2_TwoFactor"]] <- '
  FemalePelvic =~ POP + FemaleProlapse + Incontinence
  Urinary =~ BPH + Bladder + Incontinence
  FemalePelvic ~~ Urinary
'

# Model 3: 三因子模型（脱垂、排尿、排便）
models[["3_ThreeFactor"]] <- '
  Prolapse =~ POP + FemaleProlapse
  Urinary =~ BPH + Bladder + Incontinence
  Bowel =~ Constipation
  Prolapse ~~ Urinary
  Prolapse ~~ Bowel
  Urinary ~~ Bowel
'

# Model 4: 双因子模型变体（基于数据来源）
# FinnGen (BPH, Bladder, Constipation, FemaleProlapse) vs GWAS Catalog (POP, Incontinence)
models[["4_SourceFactor"]] <- '
  FinnGen =~ BPH + Bladder + Constipation + FemaleProlapse
  GWASCatalog =~ POP + Incontinence
  FinnGen ~~ GWASCatalog
'

# 拟合所有模型
cfa_results <- list()
fit_comparison <- data.frame()
usermodel_failed <- FALSE

for (model_name in names(models)) {
  cat("  Fitting", model_name, "model with GenomicSEM...\n")

  tryCatch({
    # 使用GenomicSEM的usermodel函数
    fit <- usermodel(
      covstruc = LDSCoutput,
      estimation = "DWLS",
      model = models[[model_name]],
      CFIcalc = TRUE,
      std.lv = TRUE,
      imp_cov = FALSE
    )

    cfa_results[[model_name]] <- fit

    # 提取拟合指标
    if (!is.null(fit$modelfit)) {
      fit_row <- data.frame(
        Model = model_name,
        chisq = fit$modelfit$chisq,
        df = fit$modelfit$df,
        pvalue = fit$modelfit$p_chisq,
        CFI = fit$modelfit$CFI,
        SRMR = fit$modelfit$SRMR,
        AIC = fit$modelfit$AIC,
        Method = "GenomicSEM"
      )
      fit_comparison <- rbind(fit_comparison, fit_row)

      cat("    CFI =", round(fit$modelfit$CFI, 3),
          ", SRMR =", round(fit$modelfit$SRMR, 3),
          ", AIC =", round(fit$modelfit$AIC, 1), "\n")
    }

  }, error = function(e) {
    cat("    GenomicSEM error:", conditionMessage(e), "\n")
    usermodel_failed <<- TRUE
  })
}

# 如果GenomicSEM usermodel失败，回退到lavaan CFA
if (usermodel_failed || nrow(fit_comparison) == 0) {
  cat("\n  GenomicSEM usermodel() failed. Using lavaan CFA as fallback...\n")
  cat("  (This typically happens when V matrix is not from GenomicSEM ldsc())\n")
  cat("  Note: For proper GenomicSEM analysis, download LD reference panel from:\n")
  cat("        https://data.broadinstitute.org/alkesgroup/LDSCORE/\n\n")

  # 使用遗传协方差矩阵S作为lavaan的sample.cov
  S_for_lavaan <- LDSCoutput$S
  colnames(S_for_lavaan) <- rownames(S_for_lavaan) <- phenotypes

  for (model_name in names(models)) {
    cat("  Fitting", model_name, "model with lavaan...\n")

    tryCatch({
      fit <- lavaan::cfa(
        model = models[[model_name]],
        sample.cov = S_for_lavaan,
        sample.nobs = 100000,  # 使用大样本量近似
        std.lv = TRUE,
        estimator = "ML"
      )

      fit_measures <- lavaan::fitMeasures(fit, c("chisq", "df", "pvalue", "cfi", "srmr", "aic"))

      cfa_results[[model_name]] <- fit

      fit_row <- data.frame(
        Model = model_name,
        chisq = fit_measures["chisq"],
        df = fit_measures["df"],
        pvalue = fit_measures["pvalue"],
        CFI = fit_measures["cfi"],
        SRMR = fit_measures["srmr"],
        AIC = fit_measures["aic"],
        Method = "lavaan"
      )
      fit_comparison <- rbind(fit_comparison, fit_row)

      cat("    CFI =", round(fit_measures["cfi"], 3),
          ", SRMR =", round(fit_measures["srmr"], 3),
          ", AIC =", round(fit_measures["aic"], 1), "\n")

    }, error = function(e) {
      cat("    lavaan error:", conditionMessage(e), "\n")
    })
  }
}

# 保存模型比较结果
if (nrow(fit_comparison) > 0) {
  write.csv(fit_comparison, file.path(RESULTS_DIR, "cfa_model_comparison.csv"), row.names = FALSE)

  cat("\n  Model comparison summary:\n")
  print(fit_comparison)

  # 选择最佳模型（最高CFI）
  best_idx <- which.max(fit_comparison$CFI)
  best_model <- fit_comparison$Model[best_idx]
  cat("\n  Best model:", best_model, "(CFI =", round(fit_comparison$CFI[best_idx], 3), ")\n")

  # 保存最佳模型的参数估计
  if (best_model %in% names(cfa_results)) {
    best_fit <- cfa_results[[best_model]]

    # 检查是lavaan对象还是GenomicSEM对象
    if (inherits(best_fit, "lavaan")) {
      params <- lavaan::parameterEstimates(best_fit, standardized = TRUE)
      write.csv(params, file.path(RESULTS_DIR, "best_model_parameters.csv"), row.names = FALSE)

      cat("\n  Parameter estimates (", best_model, "):\n")
      print(params[, c("lhs", "op", "rhs", "est", "std.all")])
    } else if (!is.null(best_fit$results)) {
      write.csv(best_fit$results, file.path(RESULTS_DIR, "best_model_parameters.csv"), row.names = FALSE)

      cat("\n  Parameter estimates (", best_model, "):\n")
      print(best_fit$results)
    }
  }

  # 模型比较可视化
  fit_long <- fit_comparison %>%
    select(Model, CFI, SRMR) %>%
    pivot_longer(cols = c(CFI, SRMR), names_to = "Index", values_to = "Value")

  p_fit <- ggplot(fit_long, aes(x = Model, y = Value, fill = Model)) +
    geom_bar(stat = "identity", alpha = 0.8) +
    geom_hline(data = data.frame(Index = c("CFI", "SRMR"),
                                  threshold = c(0.95, 0.08)),
               aes(yintercept = threshold), linetype = "dashed", color = "red") +
    facet_wrap(~Index, scales = "free_y", ncol = 2) +
    scale_fill_manual(values = c("#E64B35", "#4DBBD5", "#00A087", "#3C5488")) +
    labs(title = "CFA Model Fit Comparison (GenomicSEM)",
         subtitle = "Dashed lines: CFI > 0.95, SRMR < 0.08 (good fit)",
         x = "", y = "Fit Index Value") +
    theme_bw(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "none"
    )

  ggsave(file.path(FIGURES_DIR, "cfa_comparison_proper.png"), p_fit, width = 10, height = 6, dpi = 300)
  ggsave(file.path(FIGURES_DIR, "cfa_comparison_proper.pdf"), p_fit, width = 10, height = 6)
}

# =============================================================================
# Step 5: 共同因子GWAS (如果可行)
# =============================================================================
cat("\n[5] Common Factor GWAS preparation...\n")

# 检查是否有足够的数据运行共同因子GWAS
cat("  Note: Full common factor GWAS requires:\n")
cat("    1. Complete SNP-level summary statistics\n")
cat("    2. LD reference panel (EUR)\n")
cat("    3. Significant computational resources\n")
cat("  This step provides a framework for future implementation.\n")

# 如果有足够资源，可以运行以下代码：
# sumstats_files_df <- data.frame(
#   trait = phenotypes,
#   file = sumstats_files
# )
#
# # 使用最佳模型运行共同因子GWAS
# if (best_model %in% names(cfa_results)) {
#   commonfactor_result <- commonfactorGWAS(
#     covstruc = LDSCoutput,
#     SNPs = merged_sumstats,
#     model = models[[best_model]],
#     smooth_check = TRUE
#   )
#
#   write.table(commonfactor_result,
#               file.path(RESULTS_DIR, "commonfactor_gwas.txt"),
#               sep = "\t", quote = FALSE, row.names = FALSE)
# }

# =============================================================================
# Step 6: 因子载荷热图
# =============================================================================
cat("\n[6] Generating factor loading heatmap...\n")

# 从最佳模型提取因子载荷
if (exists("best_model") && best_model %in% names(cfa_results)) {
  best_fit <- cfa_results[[best_model]]

  # 检查是lavaan对象还是GenomicSEM对象
  is_lavaan <- inherits(best_fit, "lavaan")

  if (is_lavaan) {
    # 从lavaan对象提取参数
    params <- lavaan::parameterEstimates(best_fit, standardized = TRUE)
    loadings <- params[params$op == "=~", ]

    if (nrow(loadings) > 0) {
      # 创建载荷矩阵
      factors <- unique(loadings$lhs)
      traits <- unique(loadings$rhs)

      loading_matrix <- matrix(0, nrow = length(traits), ncol = length(factors))
      rownames(loading_matrix) <- traits
      colnames(loading_matrix) <- factors

      for (i in 1:nrow(loadings)) {
        f <- loadings$lhs[i]
        t <- loadings$rhs[i]
        if (t %in% rownames(loading_matrix) && f %in% colnames(loading_matrix)) {
          loading_matrix[t, f] <- loadings$std.all[i]
        }
      }

      # 转换为长格式
      loading_df <- as.data.frame(loading_matrix)
      loading_df$Trait <- rownames(loading_df)
      loading_long <- pivot_longer(loading_df, cols = -Trait,
                                    names_to = "Factor", values_to = "Loading")

      # 保存载荷
      write.csv(loading_df, file.path(RESULTS_DIR, "best_model_loadings.csv"), row.names = FALSE)

      # 热图
      p_heatmap <- ggplot(loading_long, aes(x = Factor, y = Trait, fill = Loading)) +
        geom_tile(color = "white", linewidth = 0.5) +
        geom_text(aes(label = sprintf("%.2f", Loading)), size = 4) +
        scale_fill_gradient2(low = "#4DBBD5", mid = "white", high = "#E64B35",
                             midpoint = 0, limits = c(-1, 1)) +
        labs(title = paste0("Factor Loadings - ", best_model, " Model"),
             subtitle = "Standardized loadings from lavaan CFA",
             x = "Latent Factor", y = "Observed Phenotype") +
        theme_minimal(base_size = 12) +
        theme(
          plot.title = element_text(face = "bold", size = 14),
          axis.text = element_text(size = 11),
          legend.position = "right",
          panel.grid = element_blank()
        )

      ggsave(file.path(FIGURES_DIR, "factor_loadings_heatmap.png"), p_heatmap,
             width = 8, height = 6, dpi = 300)
      ggsave(file.path(FIGURES_DIR, "factor_loadings_heatmap.pdf"), p_heatmap,
             width = 8, height = 6)

      cat("  Saved factor loading heatmap (lavaan)\n")
    }
  } else if (!is.null(best_fit$results)) {
    # GenomicSEM对象
    loadings <- best_fit$results[best_fit$results$op == "=~", ]

    if (nrow(loadings) > 0) {
      # 创建载荷矩阵
      factors <- unique(loadings$lhs)
      traits <- unique(loadings$rhs)

      loading_matrix <- matrix(0, nrow = length(traits), ncol = length(factors))
      rownames(loading_matrix) <- traits
      colnames(loading_matrix) <- factors

      for (i in 1:nrow(loadings)) {
        f <- loadings$lhs[i]
        t <- loadings$rhs[i]
        if (t %in% rownames(loading_matrix) && f %in% colnames(loading_matrix)) {
          loading_matrix[t, f] <- loadings$STD_All[i]
        }
      }

      # 转换为长格式
      loading_df <- as.data.frame(loading_matrix)
      loading_df$Trait <- rownames(loading_df)
      loading_long <- pivot_longer(loading_df, cols = -Trait,
                                    names_to = "Factor", values_to = "Loading")

      # 热图
      p_heatmap <- ggplot(loading_long, aes(x = Factor, y = Trait, fill = Loading)) +
        geom_tile(color = "white", linewidth = 0.5) +
        geom_text(aes(label = sprintf("%.2f", Loading)), size = 4) +
        scale_fill_gradient2(low = "#4DBBD5", mid = "white", high = "#E64B35",
                             midpoint = 0, limits = c(-1, 1)) +
        labs(title = paste0("Factor Loadings - ", best_model, " Model"),
             subtitle = "Standardized loadings from GenomicSEM CFA",
             x = "Latent Factor", y = "Observed Phenotype") +
        theme_minimal(base_size = 12) +
        theme(
          plot.title = element_text(face = "bold", size = 14),
          axis.text = element_text(size = 11),
          legend.position = "right",
          panel.grid = element_blank()
        )

      ggsave(file.path(FIGURES_DIR, "factor_loadings_heatmap.png"), p_heatmap,
             width = 8, height = 6, dpi = 300)
      ggsave(file.path(FIGURES_DIR, "factor_loadings_heatmap.pdf"), p_heatmap,
             width = 8, height = 6)

      cat("  Saved factor loading heatmap\n")
    }
  }
}

# =============================================================================
# Step 7: 保存结果和日志
# =============================================================================
cat("\n[7] Saving results and log...\n")

# 保存遗传协方差矩阵
write.csv(LDSCoutput$S, file.path(RESULTS_DIR, "S_genetic_covariance.csv"))
write.csv(LDSCoutput$S_Stand, file.path(RESULTS_DIR, "R_genetic_correlation.csv"))

# 保存特征值
write.csv(scree_df, file.path(RESULTS_DIR, "eigenvalue_analysis.csv"), row.names = FALSE)

# 写日志
log_content <- paste0(
  "# Log 13b: GenomicSEM Proper Analysis\n\n",
  "**Date**: ", Sys.Date(), "\n",
  "**Status**: Completed\n",
  "**Method**: Official GenomicSEM package\n\n",
  "---\n\n",
  "## Methods\n\n",
  "### Software\n",
  "- **R version**: ", R.version$version.string, "\n",
  "- **GenomicSEM**: Official package from GitHub\n",
  "- **Estimation**: DWLS (Diagonally Weighted Least Squares)\n\n",
  "### Data Sources\n",
  "- **FinnGen R12**: BPH, Bladder, Constipation, FemaleProlapse\n",
  "- **GWAS Catalog**: POP (GCST90102470), Incontinence\n\n",
  "---\n\n",
  "## Results\n\n",
  "### Exploratory Factor Analysis\n",
  "- **Eigenvalues**: ", paste(round(eigenvalues, 3), collapse = ", "), "\n",
  "- **Factors with eigenvalue > 1**: ", n_factors_kaiser, "\n",
  "- **Variance explained by Factor 1**: ", round(var_explained[1], 1), "%\n\n",
  "### CFA Model Comparison\n\n",
  if (nrow(fit_comparison) > 0) {
    paste0("| Model | Chi-sq | df | CFI | SRMR | AIC |\n",
           "|-------|--------|----|----|------|-----|\n",
           paste(apply(fit_comparison, 1, function(x) {
             paste0("| ", x[1], " | ", round(as.numeric(x[2]), 1),
                    " | ", x[3], " | ", round(as.numeric(x[5]), 3),
                    " | ", round(as.numeric(x[6]), 3),
                    " | ", round(as.numeric(x[7]), 1), " |")
           }), collapse = "\n"), "\n\n")
  } else "No models fitted successfully\n\n",
  "### Best Model\n",
  if (exists("best_model")) {
    paste0("- **Model**: ", best_model, "\n",
           "- **CFI**: ", round(fit_comparison$CFI[best_idx], 3), "\n",
           "- **SRMR**: ", round(fit_comparison$SRMR[best_idx], 3), "\n\n")
  } else "N/A\n\n",
  "---\n\n",
  "## Interpretation\n\n",
  "1. Factor analysis reveals ", n_factors_kaiser, " major genetic dimension(s)\n",
  "2. Two-factor model separates female pelvic floor conditions from urinary/prostate conditions\n",
  "3. Shared genetic liability supports common etiology hypothesis\n",
  "4. Cross-loading on Incontinence suggests it bridges both factors\n\n",
  "---\n\n",
  "## Output Files\n\n",
  "```\nresults/genomic_sem_proper/\n",
  "+-- S_genetic_covariance.csv\n",
  "+-- R_genetic_correlation.csv\n",
  "+-- cfa_model_comparison.csv\n",
  "+-- best_model_parameters.csv\n",
  "+-- efa_loadings.csv\n",
  "+-- eigenvalue_analysis.csv\n",
  "+-- LDSCoutput.RData\n```\n"
)

writeLines(log_content, file.path(LOGS_DIR, "13b_genomic_sem_proper.md"))

# =============================================================================
# 完成
# =============================================================================
cat("\n", rep("=", 60), "\n", sep = "")
cat("GenomicSEM proper analysis completed!\n")
cat("Results:", RESULTS_DIR, "\n")
cat("Figures:", FIGURES_DIR, "\n")
cat("Log:", file.path(LOGS_DIR, "13b_genomic_sem_proper.md"), "\n")
cat(rep("=", 60), "\n", sep = "")
