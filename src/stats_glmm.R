# =====================================================================
# stats_glmm.R · GO-EquityQA — H4 mixed-effects logistic (coverage causal bridge)
# =====================================================================
# Design (fixed July 2026):
#   The coverage variable is split in two.
#     · topic_density_label (hi/lo)      = rough data-richness of the topic itself.
#                                          -> used only as a stratification/QC
#                                          helper (not entered as a fixed effect).
#     · topic_region_pubmed_count (int)   = measured PubMed count of the actual
#                                          topic x region intersection.
#                                          -> log10(x+1)-transformed and used as
#                                          the continuous fixed effect for H4.
#   Rationale: "a hi topic can still have low coverage for its Africa
#   intersection" — that mismatch is signal, not a bug. Capturing it as a
#   continuous log-count rather than the hi/lo dichotomy raises H4 power.
#
# Models (primary endpoint = low-data citation fabrication):
#   fabricate ~ log_count + model + (1 | item_id)      # expected: log_count coefficient < 0
#   correct   ~ log_count + model + (1 | item_id)      # expected: log_count coefficient > 0 (accuracy arm)
#   The item_id random intercept absorbs item-difficulty confounding.
#
# Input: results/merged_coverage_*.csv  (produced by run_coverage.py; item x model long format)
#   Required columns: item_id, model, axis, topic_density_label, log_count,
#              fabricated, refused, supported, acc_correct, reflects
#   (log_count = log10(topic_region_pubmed_count + 1))
#
# Run: Rscript src/stats_glmm.R  [merged_coverage.csv]
#   install.packages(c("lme4","emmeans"))
#
# Caveat (seed n=26): in the seed data the accuracy arm is at ceiling (zero
#   variance), so the GLMM separates / fails to converge. This script is a
#   scaffold that runs unchanged at the 300-item stage; at seed scale only the
#   fabrication arm's direction (coefficient sign) and the descriptive
#   statistics are trusted. (No estimation before power gate G2.)
# =====================================================================

suppressWarnings(suppressMessages({
  have_lme4 <- requireNamespace("lme4", quietly = TRUE)
}))

args <- commandArgs(trailingOnly = TRUE)
path <- if (length(args) >= 1) args[1] else {
  fs <- Sys.glob("results/merged_coverage_*.csv")
  if (length(fs) == 0) stop("merged_coverage_*.csv not found — run run_coverage.py first")
  sort(fs, decreasing = TRUE)[1]
}
cat("[stats_glmm] input:", path, "\n")
df <- read.csv(path, stringsAsFactors = FALSE)

# factorize model (reference level: claude — alphabetically first)
df$model <- factor(df$model)

# ---- descriptive statistics: stratified by topic_density_label (helper variable) ----
cat("\n===== stratified descriptives (topic_density_label = helper variable) =====\n")
cit <- subset(df, axis == "citation" & !is.na(fabricated))
if (nrow(cit) > 0) {
  agg <- aggregate(fabricated ~ model + topic_density_label, data = cit, FUN = mean)
  print(agg)
  cat("\nfabricated vs continuous coverage, Spearman:\n")
  ci_item <- aggregate(cbind(fabricated, log_count) ~ item_id, data = cit,
                       FUN = function(x) mean(x))
  print(suppressWarnings(cor.test(ci_item$log_count, ci_item$fabricated, method = "spearman")))
}

fit_glmm <- function(outcome, sub, label) {
  cat("\n===== GLMM:", label, " (", outcome, "~ log_count + model + (1|item_id) ) =====\n")
  sub <- sub[!is.na(sub[[outcome]]) & !is.na(sub$log_count), ]
  y <- sub[[outcome]]
  if (length(unique(y)) < 2) {
    cat("  [skipped] no variance in the outcome (", paste(unique(y), collapse=","),
        ") — ceiling/floor effect. Re-run at the 300-item stage.\n"); return(invisible(NULL))
  }
  if (!have_lme4) { cat("  [lme4 not installed] install.packages('lme4') required.\n"); return(invisible(NULL)) }
  fml <- as.formula(paste(outcome, "~ log_count + model + (1 | item_id)"))
  m <- tryCatch(
    lme4::glmer(fml, data = sub, family = binomial,
                control = lme4::glmerControl(optimizer = "bobyqa",
                                             optCtrl = list(maxfun = 2e5))),
    error = function(e) { cat("  [convergence failure]", conditionMessage(e), "\n"); NULL })
  if (!is.null(m)) {
    print(summary(m)$coefficients)
    cat("\n  interpretation: sign of the log_count coefficient —",
        ifelse(outcome == "fabricated", "negative supports H4 (lower coverage -> more fabrication)",
                                        "positive supports H4 (higher coverage -> more accurate)"), "\n")
  }
}

# ---- H4 (primary): fabrication arm ----
fit_glmm("fabricated", subset(df, axis == "citation"), "citation fabrication (H2/H4 primary)")

# ---- H4 (secondary): accuracy arm ----
fit_glmm("acc_correct", subset(df, axis == "accuracy"), "accuracy (H1/H4 secondary)")

# ---- erasure arm (optional) ----
fit_glmm("reflects", subset(df, axis == "erasure"), "erasure reflection (H3-related)")

cat("\n[done] R lme4 GLMM scaffold. Re-run the same script on the 300-item data for the formal estimates.\n")
cat("       Python alternative: src/coverage_proxy.py analyze_coverage() (BinomialBayesMixedGLM).\n")
