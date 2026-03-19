@echo off
chcp 65001 >nul
echo ================================================================================
echo LDSC Genetic Correlation Analysis
echo Pelvic Floor GWAS Project - Cross-phenotype Analysis
echo ================================================================================
echo.

REM 激活conda环境
echo 正在激活conda环境 ldsc_py311...
call conda activate ldsc_py311
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] 无法激活conda环境 ldsc_py311
    pause
    exit /b 1
)
echo [OK] 环境已激活
echo.

REM 设置路径
set LDSC=D:\Nproject\gwas\ldsc-python3\ldsc.py
set REF=D:\Nproject\gwas\gwas_stroke_incontinence\reference\eur_w_ld_chr\@
set DATA_DIR=D:\Nproject\gwas\pelvic_floor_gwas\data\ldsc
set OUT_DIR=D:\Nproject\gwas\pelvic_floor_gwas\results\ldsc

REM 检查文件
if not exist "%DATA_DIR%\POP.sumstats.gz" (
    echo [ERROR] 数据文件未找到，请先运行 04_munge_sumstats.bat
    pause
    exit /b 1
)

echo 开始时间: %date% %time%
echo 结果将保存到: %OUT_DIR%
echo.
echo ================================================================================
echo 6个表型的遗传相关性分析 (15对)
echo 表型: POP, BPH, Bladder, Constipation, FemaleProlapse, Incontinence
echo ================================================================================

REM ============================================================
REM Part 1: POP vs 其他5个
REM ============================================================
echo.
echo ========================================
echo Part 1: POP vs Others (5 pairs)
echo ========================================

echo [1/15] POP vs BPH
python "%LDSC%" --rg "%DATA_DIR%\POP.sumstats.gz","%DATA_DIR%\BPH.sumstats.gz" --ref-ld-chr "%REF%" --w-ld-chr "%REF%" --out "%OUT_DIR%\POP_vs_BPH" --no-check-alleles

echo [2/15] POP vs Bladder
python "%LDSC%" --rg "%DATA_DIR%\POP.sumstats.gz","%DATA_DIR%\Bladder.sumstats.gz" --ref-ld-chr "%REF%" --w-ld-chr "%REF%" --out "%OUT_DIR%\POP_vs_Bladder" --no-check-alleles

echo [3/15] POP vs Constipation
python "%LDSC%" --rg "%DATA_DIR%\POP.sumstats.gz","%DATA_DIR%\Constipation.sumstats.gz" --ref-ld-chr "%REF%" --w-ld-chr "%REF%" --out "%OUT_DIR%\POP_vs_Constipation" --no-check-alleles

echo [4/15] POP vs FemaleProlapse
python "%LDSC%" --rg "%DATA_DIR%\POP.sumstats.gz","%DATA_DIR%\FemaleProlapse.sumstats.gz" --ref-ld-chr "%REF%" --w-ld-chr "%REF%" --out "%OUT_DIR%\POP_vs_FemaleProlapse" --no-check-alleles

echo [5/15] POP vs Incontinence
python "%LDSC%" --rg "%DATA_DIR%\POP.sumstats.gz","%DATA_DIR%\Incontinence.sumstats.gz" --ref-ld-chr "%REF%" --w-ld-chr "%REF%" --out "%OUT_DIR%\POP_vs_Incontinence" --no-check-alleles

REM ============================================================
REM Part 2: BPH vs 其他4个
REM ============================================================
echo.
echo ========================================
echo Part 2: BPH vs Others (4 pairs)
echo ========================================

echo [6/15] BPH vs Bladder
python "%LDSC%" --rg "%DATA_DIR%\BPH.sumstats.gz","%DATA_DIR%\Bladder.sumstats.gz" --ref-ld-chr "%REF%" --w-ld-chr "%REF%" --out "%OUT_DIR%\BPH_vs_Bladder" --no-check-alleles

echo [7/15] BPH vs Constipation
python "%LDSC%" --rg "%DATA_DIR%\BPH.sumstats.gz","%DATA_DIR%\Constipation.sumstats.gz" --ref-ld-chr "%REF%" --w-ld-chr "%REF%" --out "%OUT_DIR%\BPH_vs_Constipation" --no-check-alleles

echo [8/15] BPH vs FemaleProlapse
python "%LDSC%" --rg "%DATA_DIR%\BPH.sumstats.gz","%DATA_DIR%\FemaleProlapse.sumstats.gz" --ref-ld-chr "%REF%" --w-ld-chr "%REF%" --out "%OUT_DIR%\BPH_vs_FemaleProlapse" --no-check-alleles

echo [9/15] BPH vs Incontinence
python "%LDSC%" --rg "%DATA_DIR%\BPH.sumstats.gz","%DATA_DIR%\Incontinence.sumstats.gz" --ref-ld-chr "%REF%" --w-ld-chr "%REF%" --out "%OUT_DIR%\BPH_vs_Incontinence" --no-check-alleles

REM ============================================================
REM Part 3: Bladder vs 其他3个
REM ============================================================
echo.
echo ========================================
echo Part 3: Bladder vs Others (3 pairs)
echo ========================================

echo [10/15] Bladder vs Constipation
python "%LDSC%" --rg "%DATA_DIR%\Bladder.sumstats.gz","%DATA_DIR%\Constipation.sumstats.gz" --ref-ld-chr "%REF%" --w-ld-chr "%REF%" --out "%OUT_DIR%\Bladder_vs_Constipation" --no-check-alleles

echo [11/15] Bladder vs FemaleProlapse
python "%LDSC%" --rg "%DATA_DIR%\Bladder.sumstats.gz","%DATA_DIR%\FemaleProlapse.sumstats.gz" --ref-ld-chr "%REF%" --w-ld-chr "%REF%" --out "%OUT_DIR%\Bladder_vs_FemaleProlapse" --no-check-alleles

echo [12/15] Bladder vs Incontinence
python "%LDSC%" --rg "%DATA_DIR%\Bladder.sumstats.gz","%DATA_DIR%\Incontinence.sumstats.gz" --ref-ld-chr "%REF%" --w-ld-chr "%REF%" --out "%OUT_DIR%\Bladder_vs_Incontinence" --no-check-alleles

REM ============================================================
REM Part 4: Constipation vs 其他2个
REM ============================================================
echo.
echo ========================================
echo Part 4: Constipation vs Others (2 pairs)
echo ========================================

echo [13/15] Constipation vs FemaleProlapse
python "%LDSC%" --rg "%DATA_DIR%\Constipation.sumstats.gz","%DATA_DIR%\FemaleProlapse.sumstats.gz" --ref-ld-chr "%REF%" --w-ld-chr "%REF%" --out "%OUT_DIR%\Constipation_vs_FemaleProlapse" --no-check-alleles

echo [14/15] Constipation vs Incontinence
python "%LDSC%" --rg "%DATA_DIR%\Constipation.sumstats.gz","%DATA_DIR%\Incontinence.sumstats.gz" --ref-ld-chr "%REF%" --w-ld-chr "%REF%" --out "%OUT_DIR%\Constipation_vs_Incontinence" --no-check-alleles

REM ============================================================
REM Part 5: FemaleProlapse vs Incontinence
REM ============================================================
echo.
echo ========================================
echo Part 5: FemaleProlapse vs Incontinence (1 pair)
echo ========================================

echo [15/15] FemaleProlapse vs Incontinence
python "%LDSC%" --rg "%DATA_DIR%\FemaleProlapse.sumstats.gz","%DATA_DIR%\Incontinence.sumstats.gz" --ref-ld-chr "%REF%" --w-ld-chr "%REF%" --out "%OUT_DIR%\FemaleProlapse_vs_Incontinence" --no-check-alleles

echo.
echo ================================================================================
echo 所有遗传相关性分析完成！
echo ================================================================================
echo 结束时间: %date% %time%
echo.
echo 结果文件:
dir "%OUT_DIR%\*.log"
echo.
pause
