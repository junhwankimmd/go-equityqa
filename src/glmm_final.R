# Mixed-effects logistic models reported in the manuscript (Results; Supplementary Table S2).
# Inputs: final adjudicated item x model tables (300 confirmatory provenance responses).
# Output: results/glmm_final_summaries.txt (coefficients, standard errors, Wald 95% CIs).
suppressMessages(library(lme4))
files <- c(strict_fabrication = "results/merged_strict_final_20260829.csv",
           citation_support_failure = "results/merged_citefail_final_20260829.csv")
sink("results/glmm_final_summaries.txt")
for (nm in names(files)) {
  d <- read.csv(files[[nm]])
  d$model <- factor(d$model, levels = c("claude", "gpt", "gemini"))
  m <- glmer(fabricated ~ log_count + model + (1 | item_id), data = d,
             family = binomial, control = glmerControl(optimizer = "bobyqa"))
  co <- summary(m)$coefficients
  ci <- confint(m, parm = "beta_", method = "Wald")
  cat("=====", nm, "(n =", nrow(d), "responses,", length(unique(d$item_id)), "items) =====\n")
  cat("Model: outcome ~ log10(coverage + 1) + model + (1 | item); reference model = Claude Opus 4.8\n")
  print(round(cbind(co, CI_low = ci[, 1], CI_high = ci[, 2]), 4))
  cat(sprintf("Random intercept SD (item): %.4f\n\n", attr(VarCorr(m)$item_id, "stddev")))
}
sink()
