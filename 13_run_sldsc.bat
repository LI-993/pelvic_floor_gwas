@echo off
chcp 65001 >nul
echo ================================================================================
echo S-LDSC Partitioned Heritability Analysis
echo Pelvic Floor GWAS - Phase 4
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
set ANNOT=D:\Nproject\gwas\pelvic_floor_gwas\reference\ldsc_annotations
set BASELINE=%ANNOT%\baselineLD.
set WEIGHTS=%ANNOT%\1000G_Phase3_weights_hm3_no_MHC\weights.hm3_noMHC.
set FRQ=%ANNOT%\1000G_Phase3_frq\1000G.EUR.QC.
set DATA_DIR=D:\Nproject\gwas\pelvic_floor_gwas\data\ldsc
set OUT_DIR=D:\Nproject\gwas\pelvic_floor_gwas\results\sldsc

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

echo 开始时间: %date% %time%
echo 结果将保存到: %OUT_DIR%
echo.
echo ================================================================================
echo 使用 baselineLD v2.2 模型 (97 annotations)
echo ================================================================================

REM ============================================================
REM POP
REM ============================================================
echo.
echo [1/6] Running S-LDSC for POP...
python "%LDSC%" ^
    --h2 "%DATA_DIR%\POP.sumstats.gz" ^
    --ref-ld-chr "%BASELINE%" ^
    --w-ld-chr "%WEIGHTS%" ^
        --frqfile-chr "%FRQ%" ^
    --out "%OUT_DIR%\POP_baselineLD"

REM ============================================================
REM BPH
REM ============================================================
echo.
echo [2/6] Running S-LDSC for BPH...
python "%LDSC%" ^
    --h2 "%DATA_DIR%\BPH.sumstats.gz" ^
    --ref-ld-chr "%BASELINE%" ^
    --w-ld-chr "%WEIGHTS%" ^
        --frqfile-chr "%FRQ%" ^
    --out "%OUT_DIR%\BPH_baselineLD" ^
    
REM ============================================================
REM Bladder
REM ============================================================
echo.
echo [3/6] Running S-LDSC for Bladder...
python "%LDSC%" ^
    --h2 "%DATA_DIR%\Bladder.sumstats.gz" ^
    --ref-ld-chr "%BASELINE%" ^
    --w-ld-chr "%WEIGHTS%" ^
        --frqfile-chr "%FRQ%" ^
    --out "%OUT_DIR%\Bladder_baselineLD" ^
    
REM ============================================================
REM Constipation
REM ============================================================
echo.
echo [4/6] Running S-LDSC for Constipation...
python "%LDSC%" ^
    --h2 "%DATA_DIR%\Constipation.sumstats.gz" ^
    --ref-ld-chr "%BASELINE%" ^
    --w-ld-chr "%WEIGHTS%" ^
        --frqfile-chr "%FRQ%" ^
    --out "%OUT_DIR%\Constipation_baselineLD" ^
    
REM ============================================================
REM FemaleProlapse
REM ============================================================
echo.
echo [5/6] Running S-LDSC for FemaleProlapse...
python "%LDSC%" ^
    --h2 "%DATA_DIR%\FemaleProlapse.sumstats.gz" ^
    --ref-ld-chr "%BASELINE%" ^
    --w-ld-chr "%WEIGHTS%" ^
        --frqfile-chr "%FRQ%" ^
    --out "%OUT_DIR%\FemaleProlapse_baselineLD" ^
    
REM ============================================================
REM Incontinence
REM ============================================================
echo.
echo [6/6] Running S-LDSC for Incontinence...
python "%LDSC%" ^
    --h2 "%DATA_DIR%\Incontinence.sumstats.gz" ^
    --ref-ld-chr "%BASELINE%" ^
    --w-ld-chr "%WEIGHTS%" ^
        --frqfile-chr "%FRQ%" ^
    --out "%OUT_DIR%\Incontinence_baselineLD" ^
    
echo.
echo ================================================================================
echo S-LDSC分析完成！
echo ================================================================================
echo 结束时间: %date% %time%
echo.
echo 结果文件:
dir "%OUT_DIR%\*.results"
echo.
pause
