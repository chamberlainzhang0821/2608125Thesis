# Task 3 Sensitivity Analysis Summary

## What we tested
- Random-seed stability of 5-fold CV metrics (RMSE, R^2).
- Stability of 5 grouped effect sizes (Age/Partner/Industry/HomeState/HomeCountry).
- Bootstrap CIs for mean grouped effects.
- Sensitivity to ridge penalty λ and TopK category caps.

## Key takeaways
- **Fans_AvgFanShare**: CV $R^2$ = 0.078 ± 0.030 (across seeds).
- **Judges_AvgJudgeTotal**: CV $R^2$ = 0.249 ± 0.015 (across seeds).
- **Performance_WeeksSurvived**: CV $R^2$ = 0.165 ± 0.021 (across seeds).

## Outputs
- Figures: `/Users/jingyuanzhang/Program/Main Model/results/figures_task3_sensitivity`
- LaTeX tables: `/Users/jingyuanzhang/Program/Main Model/results/tables_task3_sensitivity.tex`