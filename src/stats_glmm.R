# =====================================================================
# stats_glmm.R - GO-EquityQA: H4 mixed-effects logistic models (coverage)
# =====================================================================
# Coverage is represented by two variables:
#   - topic_density_label (hi/lo): coarse data density of the topic itself.
#     Used only for stratified descriptives; not entered as a fixed effect.
#   - topic_region_pubmed_count (integer): observed PubMed count for the
#     topic x region intersection, transformed to log10(x + 1) and used as
#     the continuous fixed effect for H4.
#   A high-density topic can still have low coverage for a specific region;
#   the continuous log count captures this, whereas the binary label does not.
#
# Models (primary endpoint: citation fabrication):
#   fabricate ~ log_count + model + (1 | item_id)   # H4 expects log_count < 0
#   correct   ~ log_count + model + (1 | item_id)   # H4 expects log_count > 0
#   The item_id random intercept absorbs item-level difficulty.
#
# Input: results/merged_coverage_*.csv (item x model long format, produced
#   by run_coverage.py)
#   Required columns: item_id, model, axis, topic_density_label, log_count,
#                     fabricated, refused, supported, acc_correct, reflects
#   (log_count = log10(topic_region_pubmed_count + 1))
#
# Usage: Rscript src/stats_glmm.R [merged_coverage.csv]
#   install.packages(c("lme4","emmeans"))
#
# Outcomes without variance (ceiling/floor) are skipped, since the model
# would not be identifiable.
# =====================================================================

suppressWarnings(suppressMessages({
  have_lme4 <- requireNamespace("lme4", quietly = TRUE)
}))

args <- commandArgs(trailingOnly = TRUE)
path <- if (length(args) >= 1) args[1] else {
  fs <- Sys.glob("results/merged_coverage_*.csv")
  if (length(fs) == 0) stop("No merged_coverage_*.csv found; run run_coverage.py first.")
  sort(fs, decreasing = TRUE)[1]
}
cat("[stats_glmm] input:", path, "\n")
df <- read.csv(path, stringsAsFactors = FALSE)

# Reference level is the alphabetically first model.
df$model <- factor(df$model)

# ---- Descriptives stratified by topic_density_label (auxiliary) ----
cat("\n===== Stratified descriptives (topic_density_label, auxiliary) =====\n")
cit <- subset(df, axis == "citation" & !is.na(fabricated))
if (nrow(cit) > 0) {
  agg <- aggregate(fabricated ~ model + topic_density_label, data = cit, FUN = mean)
  print(agg)
  cat("\nSpearman correlation, fabricated vs. continuous coverage (log_count):\n")
  ci_item <- aggregate(cbind(fabricated, log_count) ~ item_id, data = cit,
                       FUN = function(x) mean(x))
  print(suppressWarnings(cor.test(ci_item$log_count, ci_item$fabricated, method = "spearman")))
}

fit_glmm <- function(outcome, sub, label) {
  cat("\n===== GLMM:", label, " (", outcome, "~ log_count + model + (1|item_id) ) =====\n")
  sub <- sub[!is.na(sub[[outcome]]) & !is.na(sub$log_count), ]
  y <- sub[[outcome]]
  if (length(unique(y)) < 2) {
    cat("  [skipped] outcome has no variance (", paste(unique(y), collapse=","),
        ") (ceiling/floor effect).\n"); return(invisible(NULL))
  }
  if (!have_lme4) { cat("  [lme4 not installed] run install.packages('lme4').\n"); return(invisible(NULL)) }
  fml <- as.formula(paste(outcome, "~ log_count + model + (1 | item_id)"))
  m <- tryCatch(
    lme4::glmer(fml, data = sub, family = binomial,
                control = lme4::glmerControl(optimizer = "bobyqa",
                                             optCtrl = list(maxfun = 2e5))),
    error = function(e) { cat("  [convergence failure]", conditionMessage(e), "\n"); NULL })
  if (!is.null(m)) {
    print(summary(m)$coefficients)
    cat("\n  Interpretation of the log_count coefficient:",
        ifelse(outcome == "fabricated", "a negative sign supports H4 (lower coverage, more fabrication)",
                                        "a positive sign supports H4 (higher coverage, higher accuracy)"), "\n")
  }
}

# ---- H4 primary: citation fabrication ----
fit_glmm("fabricated", subset(df, axis == "citation"), "Citation fabrication (H2/H4, primary)")

# ---- H4 secondary: accuracy ----
fit_glmm("acc_correct", subset(df, axis == "accuracy"), "Accuracy (H1/H4, secondary)")

# ---- Erasure (H3) ----
fit_glmm("reflects", subset(df, axis == "erasure"), "Erasure reflection (H3)")

cat("\n[done] lme4 GLMM fits complete.\n")
cat("       Python alternative: src/coverage_proxy.py analyze_coverage() (BinomialBayesMixedGLM).\n")
