# OpenGWAS 数据下载 - 简化版
# 只需要 ieugwasr，不需要 Bioconductor 依赖

# ============================================================
# 1. 安装/加载包
# ============================================================
if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}

if (!requireNamespace("ieugwasr", quietly = TRUE)) {
  remotes::install_github("MRCIEU/ieugwasr")
}

library(ieugwasr)

# ============================================================
# 2. 设置 API Token（必须！）
# ============================================================
token <- "eyJhbGciOiJSUzI1NiIsImtpZCI6ImFwaS1qd3QiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJhcGkub3Blbmd3YXMuaW8iLCJhdWQiOiJhcGkub3Blbmd3YXMuaW8iLCJzdWIiOiJuYXl1a2ljb21Ac2luYS5jbiIsImlhdCI6MTc2NTc5NTA5MywiZXhwIjoxNzY3MDA0NjkzfQ.Ek5ac5cYZT9PqddQu5N7TweRYNOZpTIPWOT6jcnoqG1Yll2fNhCp7iHnpSpZJu8HcRZ59lItJzfxNRvdGjeiwvbZ7Pb49jE6O8gEnQM2C3qkmhxCIEb1MCZIrr4M_qHcCjchCUmKzjKN6xJ2mnfLVxxDd7K6qVyfcrbvdh2oWiTo9rc5ZdX7SKgBhY3AyGNdcbxkXATGZlPd-SpEkhur102Fs-npdawvp6yT7fHQsd_-_ppOPV09qaBGrdsoYnBe9M4BBCy6KnEtJLCQIHb2zqDOgkJfwIjrX_qtDPT-eZ74nBkyzkg6qPcK8zZIm9dQ6V3idR_vPjNgIw83xQsIow"
Sys.setenv(OPENGWAS_JWT = token)
cat("API Token 已设置\n\n")

# ============================================================
# 3. 输出目录
# ============================================================
outdir <- "D:/Nproject/gwas/pelvic_floor_gwas/data/raw/OpenGWAS"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# 目标数据集
datasets <- c(
  "ukb-b-373",   # 膀胱相关
  "ukb-b-8517"   # 尿频/尿失禁
)

# ============================================================
# 4. 获取数据集信息
# ============================================================
cat("=== 数据集信息 ===\n\n")

for (id in datasets) {
  cat(sprintf("--- %s ---\n", id))
  tryCatch({
    info <- gwasinfo(id)
    if (nrow(info) > 0) {
      cat(sprintf("  表型: %s\n", info$trait[1]))
      cat(sprintf("  样本量: %s\n", info$sample_size[1]))
      cat(sprintf("  SNP数: %s\n", info$nsnp[1]))
    }
  }, error = function(e) {
    cat(sprintf("  错误: %s\n", e$message))
  })
  cat("\n")
}

# ============================================================
# 5. 按染色体下载完整数据
# ============================================================
cat("\n=== 下载完整 Summary Statistics ===\n")
cat("注意：这会按染色体查询，可能需要较长时间（每个数据集约30分钟）\n\n")

download_by_chromosome <- function(id, outdir) {
  cat(sprintf("\n开始下载 %s...\n", id))
  all_data <- data.frame()

  for (chr in 1:22) {
    cat(sprintf("  染色体 %d... ", chr))

    tryCatch({
      # 按染色体查询
      chr_data <- associations(
        variants = sprintf("%d:1-300000000", chr),
        id = id,
        proxies = 0
      )

      if (!is.null(chr_data) && nrow(chr_data) > 0) {
        all_data <- rbind(all_data, chr_data)
        cat(sprintf("获得 %d 个变异\n", nrow(chr_data)))
      } else {
        cat("无数据\n")
      }
    }, error = function(e) {
      cat(sprintf("错误: %s\n", e$message))
    })

    Sys.sleep(2)  # 避免请求过快
  }

  if (nrow(all_data) > 0) {
    # 保存结果
    outfile <- file.path(outdir, paste0(id, "_full.tsv.gz"))
    write.table(all_data, gzfile(outfile), sep = "\t", row.names = FALSE, quote = FALSE)
    cat(sprintf("\n已保存 %d 个变异到: %s\n", nrow(all_data), outfile))
  } else {
    cat("\n警告: 未获取到任何数据\n")
  }

  return(all_data)
}

# 询问是否开始下载
cat("\n是否开始下载？这需要较长时间。\n")
cat("如果要下载，请取消注释下面的代码并运行：\n\n")

cat('
# 下载 ukb-b-8517 (尿失禁相关)
# ui_data <- download_by_chromosome("ukb-b-8517", outdir)

# 下载 ukb-b-373 (膀胱相关)
# oab_data <- download_by_chromosome("ukb-b-373", outdir)
')

cat("\n\n=== 替代方案 ===\n")
cat("由于 OpenGWAS 下载较慢，建议使用已下载的 FinnGen 数据：\n")
cat("  - N14_NEUROMUSCDYSBLADD: 神经源性膀胱功能障碍（可替代 OAB）\n")
cat("  - N14_PROSTHYPERPLA: 前列腺增生\n")
cat("  - K11_CONSTIPATION: 便秘\n")
cat("  - N14_FEMGENPROL: 女性生殖器脱垂\n\n")

cat("脚本完成。\n")
