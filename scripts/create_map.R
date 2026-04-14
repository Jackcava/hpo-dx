library(data.table)
library(ontologyIndex)

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------

repo_root <- "~/Desktop/hpo-dx"

hpo_terms_file <- file.path(repo_root, "assets", "ontology", "hpo_terms_original_set.csv")
hpo_obo_file <- file.path(repo_root, "assets", "ontology", "hp.obo")

ancestors_output_file <- file.path(repo_root, "assets", "ontology", "hpo_ancestors_map.tsv")
expanded_terms_output_file <- file.path(
  repo_root,
  "assets",
  "ontology",
  "hpo_terms_expanded_with_ancestors.csv"
)

# ------------------------------------------------------------------------------
# Load original HPO terms
# ------------------------------------------------------------------------------

hpo_terms_df <- fread(hpo_terms_file)

required_cols <- c("HPO_CODE", "hpo_name")
missing_cols <- setdiff(required_cols, colnames(hpo_terms_df))
if (length(missing_cols) > 0) {
  stop(
    paste0(
      "Missing required columns in hpo_terms_original_set.csv: ",
      paste(missing_cols, collapse = ", ")
    )
  )
}

hpo_terms_df <- unique(hpo_terms_df)
original_hpo_ids <- unique(na.omit(hpo_terms_df$HPO_CODE))

# ------------------------------------------------------------------------------
# Load HPO ontology
# ------------------------------------------------------------------------------

hpo <- get_ontology(hpo_obo_file, extract_tags = "everything")

# ------------------------------------------------------------------------------
# Build ancestors map
# ------------------------------------------------------------------------------

ancestors_list <- lapply(original_hpo_ids, function(hpo_id) {
  if (hpo_id %in% names(hpo$ancestors)) {
    ancestors <- hpo$ancestors[[hpo_id]]
    ancestors <- ancestors[grepl("^HP:", ancestors)]

    if (length(ancestors) == 0) {
      return(NULL)
    }

    data.table(
      hpo_id = hpo_id,
      ancestor_id = ancestors
    )
  } else {
    NULL
  }
})

ancestors_map_df <- rbindlist(ancestors_list, use.names = TRUE, fill = TRUE)

# Remove duplicates just in case
ancestors_map_df <- unique(ancestors_map_df)

# Remove self-matches if present
ancestors_map_df <- ancestors_map_df[hpo_id != ancestor_id]

# Sort for readability
setorder(ancestors_map_df, hpo_id, ancestor_id)

# ------------------------------------------------------------------------------
# Save ancestors map
# ------------------------------------------------------------------------------

fwrite(ancestors_map_df, ancestors_output_file, sep = "\t")

# ------------------------------------------------------------------------------
# Build expanded HPO catalog for the app UI
# ------------------------------------------------------------------------------

all_ancestor_ids <- sort(unique(ancestors_map_df$ancestor_id))
all_expanded_hpo_ids <- sort(unique(c(original_hpo_ids, all_ancestor_ids)))

# Retrieve HPO names from ontology
ontology_names_dt <- data.table(
  HPO_CODE = names(hpo$name),
  hpo_name = unname(hpo$name)
)

expanded_terms_df <- data.table(HPO_CODE = all_expanded_hpo_ids)
expanded_terms_df <- merge(
  expanded_terms_df,
  ontology_names_dt,
  by = "HPO_CODE",
  all.x = TRUE,
  sort = FALSE
)

# Add provenance flag to distinguish original terms from inferred ancestors
expanded_terms_df[, term_source := ifelse(
  HPO_CODE %in% original_hpo_ids,
  "original",
  "ancestor"
)]

# Optional: keep extra original metadata when available
extra_cols <- setdiff(colnames(hpo_terms_df), c("HPO_CODE", "hpo_name"))

if (length(extra_cols) > 0) {
  original_metadata_df <- unique(hpo_terms_df[, c("HPO_CODE", extra_cols), with = FALSE])

  expanded_terms_df <- merge(
    expanded_terms_df,
    original_metadata_df,
    by = "HPO_CODE",
    all.x = TRUE,
    sort = FALSE
  )
}

# Reorder columns to keep the file readable
priority_cols <- c("HPO_CODE", "hpo_name", "term_source")
remaining_cols <- setdiff(colnames(expanded_terms_df), priority_cols)
setcolorder(expanded_terms_df, c(priority_cols, remaining_cols))

# Sort nicely
setorder(expanded_terms_df, term_source, hpo_name, HPO_CODE)

# Save expanded catalog
fwrite(expanded_terms_df, expanded_terms_output_file)

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------

cat("Ancestors map saved to:\n", ancestors_output_file, "\n")
cat("Expanded HPO catalog saved to:\n", expanded_terms_output_file, "\n")
cat("Number of source HPO terms:", length(original_hpo_ids), "\n")
cat("Number of unique ancestors:", length(all_ancestor_ids), "\n")
cat("Number of hpo_id / ancestor_id pairs:", nrow(ancestors_map_df), "\n")
cat("Number of terms in expanded catalog:", nrow(expanded_terms_df), "\n")