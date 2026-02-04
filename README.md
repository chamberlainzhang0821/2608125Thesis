README (English)

# 2608125Thesis — MCM/ICM 2026 Problem C (Team #2608125)

This repository contains our full solution for **MCM/ICM 2026 — Problem C**, including the final paper PDF, source files, code, and exported results.

## Highlights
- **Task 1:** Bayesian fan-vote inference (ABC) with uncertainty & identifiability diagnostics  
- **Task 2:** Counterfactual replay of elimination rules (Rank vs Percent vs Judge Save)  
- **Task 3:** Interpretable regression analysis (ridge + CV) for judges/fans/performance drivers  
- **Task 4:** FairVote: tunable elimination rule + robustness checks (grid sweep / bootstrap / LOSO)

## Repository Structure
> Update the paths below to match your actual folders.

- `main.pdf` — Final submitted paper (compiled PDF)
- `tex/` or `report/` — LaTeX source (if included)
- `main_model/` — Core modeling code and pipelines
- `results/` or `main_model/results/` — Generated figures and Excel reports
- `main_model/2026_MCM_Problem_C_Data.xlex` — Input data(originally .csv file, converted to .xlsx file for using)


## How to Reproduce Results
### Environment
Recommended:
- Python 3.10+ (or your used version)
- Key packages: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, `openpyxl`

Install:
```bash
pip install -r requirements.txt

2）Main program

python main_model/run_main.py
python main_model/run_task2_analysis_compare_rules.py
python main_model/run_task3_model_sweep.py
python main_model/run_task4_fairvote_backtest_simpleomega.py

3) Outputs

Key outputs are exported as Excel reports and figures, e.g.:
	•	contestant_benefit_analysis.xlsx
	•	fan_vote_estimates.xlsx
	•	method_comparison_summary.xlsx
	•	method_outcomes_by_week.xlsx
	•	task1_inference_log.xlsx
	•	weekly_uncertainty.xlsx
	•	task2_candidate_consistency_diff.xlsx
	•	task2_week_consistency_diff.xlsx
	•	task3_hyperparam_sweep_and_effects.xlsx
	•	fairvote_backtest_report.xlsx
	•	sensitivity_summary.xlsx
	•	task3_sensitivity_report.xlsx
	•	task4_sensitivity_report.xlsx

Figures are stored under:
	•	main_model/results/figures_task1/
	•	main_model/results/figures_task2/
	•	main_model/results/figures_task3/
	•	main_model/results/figures_task4/
	•	main_model/results/figures/
	•	main_model/results/figures_task3_sensitivity/
	•	main_model/results/figures_task4_sensitivity/
```

## Notes on Data

If the original dataset cannot be redistributed, this repo may exclude raw data.
In that case, please place the dataset under data/ following the expected file name used in the scripts.

## AI Use

0 AI use.

## Citation

If you reference this work, please cite:
	•	Team #2608125, MCM/ICM 2026 Problem C Solution, 2026.

## License

This project is licensed under the Apache License 2.0.

## Contact

For questions: open an issue or contact the repository owner. Alternatively, email zhangboting0821@outlook.com

---

## README（中文版）


## 2608125Thesis — MCM/ICM 2026 C题（队伍 #2608125）

本仓库包含我们对 **MCM/ICM 2026 Problem C** 的完整解答材料：最终论文 PDF、LaTeX 源文件、建模代码与结果导出表格/图像。

## 主要内容
- **Task 1：** 用 ABC 的贝叶斯方法推断每周粉丝投票强度，并给出不确定性与“可辨识度”诊断  
- **Task 2：** 固定粉丝支持度，比较不同淘汰规则（Rank / Percent / Judge Save）对结果的影响  
- **Task 3：** 使用可解释回归（岭回归 + 交叉验证）分析影响“评委/观众/存活周数”的因素  
- **Task 4：** 设计可调参的新规则 FairVote，并进行网格搜索、Bootstrap、LOSO 等稳健性检验

## 仓库结构
- `main.pdf` — 最终提交论文（PDF）
- `tex/` 或 `report/` — LaTeX 源文件（如有）
- `main_model/` — 核心建模代码与流程
- `results/` 或 `main_model/results/` — 生成的图表与 Excel 报告
- `main_model/2026_MCM_Problem_C_Data.xlex/` — 输入数据(原本为csv文件，已转为xlsx使用）

## 复现方法
### 环境：
建议：
- Python 3.10+（或与你实际使用一致的版本）
- 主要依赖：`numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, `openpyxl`

安装依赖：
```bash
pip install -r requirements.txt

2）运行程序

python main_model/run_main.py
python main_model/run_task2_analysis_compare_rules.py
python main_model/run_task3_model_sweep.py
python main_model/run_task4_fairvote_backtest_simpleomega.py

3）输出结果

关键输出通常会导出为 Excel + 图像，例如：
	•	contestant_benefit_analysis.xlsx
	•	fan_vote_estimates.xlsx
	•	method_comparison_summary.xlsx
	•	method_outcomes_by_week.xlsx
	•	task1_inference_log.xlsx
	•	weekly_uncertainty.xlsx
	•	task2_candidate_consistency_diff.xlsx
	•	task2_week_consistency_diff.xlsx
	•	task3_hyperparam_sweep_and_effects.xlsx
	•	fairvote_backtest_report.xlsx
	•	sensitivity_summary.xlsx
	•	task3_sensitivity_report.xlsx
	•	task4_sensitivity_report.xlsx

图表位于：
	•	main_model/results/figures_task1/
	•	main_model/results/figures_task2/
	•	main_model/results/figures_task3/
	•	main_model/results/figures_task4/
	•	main_model/results/figures/
	•	main_model/results/figures_task3_sensitivity/
	•	main_model/results/figures_task4_sensitivity/
```

##数据说明

原始数据已公开在https://www.contest.comap.com/undergraduate/contests/上，为2026_MCM_Problem_C_Data.csv

##AI 使用说明

0AI使用

##引用方式

如需引用本工作，可写：

Team #2608125, MCM/ICM 2026 Problem C Solution, 2026.

许可证

本项目采用 Apache License 2.0 开源许可证。

联系方式

如有问题，可提 issue 或联系仓库维护者。或邮件zhangboting0821@outlook.com

