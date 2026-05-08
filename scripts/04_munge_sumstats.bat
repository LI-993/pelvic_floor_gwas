@echo off
chcp 65001 >nul
echo ================================================================================
echo LDSC Munge Summary Statistics
echo Pelvic Floor GWAS Project
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
set MUNGE=D:\Nproject\gwas\ldsc-python3\munge_sumstats.py
set DATA_IN=D:\Nproject\gwas\pelvic_floor_gwas\data\processed
set DATA_OUT=D:\Nproject\gwas\pelvic_floor_gwas\data\ldsc

REM 检查输入目录
if not exist "%DATA_IN%" (
    echo [ERROR] 输入目录不存在: %DATA_IN%
    pause
    exit /b 1
)

echo 开始时间: %date% %time%
echo 输入目录: %DATA_IN%
echo 输出目录: %DATA_OUT%
echo.
echo ================================================================================

REM 样本量设置 (用于h2计算)
REM POP: 28086 + 546291 = 574377
REM BPH: 41137 + 460000 = 501137
REM Bladder: 3550 + 500000 = 503550
REM Constipation: 51956 + 450000 = 501956
REM FemaleProlapse: 23074 + 480000 = 503074
REM Incontinence: 27714 + 402305 = 430019

echo.
echo [1/6] Munging POP...
python "%MUNGE%" ^
    --sumstats "%DATA_IN%\POP_GRCh38.tsv.gz" ^
    --snp SNP ^
    --a1 A1 ^
    --a2 A2 ^
    --p P ^
    --signed-sumstats BETA,0 ^
    --N 574377 ^
    --out "%DATA_OUT%\POP" ^
    --merge-alleles "D:\Nproject\gwas\gwas_stroke_incontinence\reference\eur_w_ld_chr\w_hm3.snplist"
echo.

echo [2/6] Munging BPH...
python "%MUNGE%" ^
    --sumstats "%DATA_IN%\BPH_GRCh38.tsv.gz" ^
    --snp SNP ^
    --a1 A1 ^
    --a2 A2 ^
    --p P ^
    --signed-sumstats BETA,0 ^
    --N 501137 ^
    --out "%DATA_OUT%\BPH" ^
    --merge-alleles "D:\Nproject\gwas\gwas_stroke_incontinence\reference\eur_w_ld_chr\w_hm3.snplist"
echo.

echo [3/6] Munging Bladder...
python "%MUNGE%" ^
    --sumstats "%DATA_IN%\Bladder_GRCh38.tsv.gz" ^
    --snp SNP ^
    --a1 A1 ^
    --a2 A2 ^
    --p P ^
    --signed-sumstats BETA,0 ^
    --N 503550 ^
    --out "%DATA_OUT%\Bladder" ^
    --merge-alleles "D:\Nproject\gwas\gwas_stroke_incontinence\reference\eur_w_ld_chr\w_hm3.snplist"
echo.

echo [4/6] Munging Constipation...
python "%MUNGE%" ^
    --sumstats "%DATA_IN%\Constipation_GRCh38.tsv.gz" ^
    --snp SNP ^
    --a1 A1 ^
    --a2 A2 ^
    --p P ^
    --signed-sumstats BETA,0 ^
    --N 501956 ^
    --out "%DATA_OUT%\Constipation" ^
    --merge-alleles "D:\Nproject\gwas\gwas_stroke_incontinence\reference\eur_w_ld_chr\w_hm3.snplist"
echo.

echo [5/6] Munging FemaleProlapse...
python "%MUNGE%" ^
    --sumstats "%DATA_IN%\FemaleProlapse_GRCh38.tsv.gz" ^
    --snp SNP ^
    --a1 A1 ^
    --a2 A2 ^
    --p P ^
    --signed-sumstats BETA,0 ^
    --N 503074 ^
    --out "%DATA_OUT%\FemaleProlapse" ^
    --merge-alleles "D:\Nproject\gwas\gwas_stroke_incontinence\reference\eur_w_ld_chr\w_hm3.snplist"
echo.

echo [6/6] Munging Incontinence...
python "%MUNGE%" ^
    --sumstats "%DATA_IN%\Incontinence_GRCh38.tsv.gz" ^
    --snp SNP ^
    --a1 A1 ^
    --a2 A2 ^
    --p P ^
    --signed-sumstats BETA,0 ^
    --N 430019 ^
    --out "%DATA_OUT%\Incontinence" ^
    --merge-alleles "D:\Nproject\gwas\gwas_stroke_incontinence\reference\eur_w_ld_chr\w_hm3.snplist"
echo.

echo ================================================================================
echo Munge完成！
echo ================================================================================
echo 结束时间: %date% %time%
echo.
echo 输出文件:
dir "%DATA_OUT%\*.sumstats.gz"
echo.
pause
