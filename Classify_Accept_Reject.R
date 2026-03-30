# ============================================================
# Step 1 — Batch LLM-assisted screening pipeline
# Topic: road networks + biodiversity / ecological connectivity
# Provider: Groq
# Model: llama-3.3-70b-versatile
# Strategy: small batches, JSON object output with results array
# Output: adds screening columns to CSV/TSV
# ============================================================

library(httr)
library(jsonlite)

# ------------------------------------------------------------
# 1. API KEY
# ------------------------------------------------------------
api_key <- Sys.getenv("GROQ_API_KEY")
if (api_key == "") {
  stop("GROQ_API_KEY not found. Add it to your .Renviron file or environment.")
}

# ------------------------------------------------------------
# 2. MODEL SELECTION
# ------------------------------------------------------------
model_name <- "llama-3.3-70b-versatile"

# ------------------------------------------------------------
# 3. SETTINGS
# ------------------------------------------------------------
batch_size <- 6
max_retries <- 4
sleep_between_requests <- 2
request_timeout <- 90
max_abstract_chars <- 2500
max_keywords_chars <- 600

input_file <- "database_output_raw/scopus_export_prototype_1.csv"
output_file_csv <- "output/screened_step1_groq.csv"
output_file_tsv <- "output/screened_step1_groq.txt"
prompt_file <- "prompts/prototype2_roadnet_biodconnectivity.txt"

# ------------------------------------------------------------
# 4. LOAD THE SYSTEM PROMPT
# ------------------------------------------------------------
system_request <- readLines(prompt_file, warn = FALSE, encoding = "UTF-8")
full_system_request <- paste(system_request, collapse = "\n")

# ------------------------------------------------------------
# 5. HELPER: safe JSON parsing
# ------------------------------------------------------------
safe_parse_json <- function(text_output) {
  cleaned_text <- gsub("^```json\\s*|^```\\s*|\\s*```$", "", text_output)
  cleaned_text <- trimws(cleaned_text)
  
  tryCatch(
    fromJSON(cleaned_text, simplifyDataFrame = TRUE),
    error = function(e) NULL
  )
}

# ------------------------------------------------------------
# 6. HELPER: normalize one field
# ------------------------------------------------------------
normalize_field <- function(value, allowed = NULL, default = NA_character_) {
  if (is.null(value) || length(value) == 0 || is.na(value) || trimws(value) == "") {
    return(default)
  }
  
  value <- trimws(tolower(as.character(value)))
  
  if (!is.null(allowed) && !(value %in% allowed)) {
    return(default)
  }
  
  value
}

# ------------------------------------------------------------
# 7. HELPER: text truncation
# ------------------------------------------------------------
truncate_text <- function(x, max_chars) {
  x <- ifelse(is.na(x), "", x)
  ifelse(nchar(x) > max_chars, substr(x, 1, max_chars), x)
}

# ------------------------------------------------------------
# 8. HELPER: build empty fallback rows for a failed batch
# ------------------------------------------------------------
build_fallback_batch <- function(batch_df, raw_response = "", error_msg = "invalid_json") {
  data.frame(
    Code = batch_df$Code,
    ScreeningDecision = "error",
    ScreeningReason = error_msg,
    ScreeningConfidence = "low",
    NeedsManualCheck = "yes",
    RawModelOutput = raw_response,
    stringsAsFactors = FALSE
  )
}

# ------------------------------------------------------------
# 9. HELPER: standardize parsed batch output
# ------------------------------------------------------------
standardize_batch_output <- function(parsed_json, batch_df, raw_response = "") {
  if (is.null(parsed_json)) {
    return(build_fallback_batch(batch_df, raw_response, "parsed_json_null"))
  }
  
  # Expecting: { "results": [ ... ] }
  if (!is.list(parsed_json) || is.null(parsed_json$results)) {
    return(build_fallback_batch(batch_df, raw_response, "missing_results_array"))
  }
  
  results_df <- parsed_json$results
  
  if (!is.data.frame(results_df)) {
    results_df <- tryCatch(as.data.frame(results_df, stringsAsFactors = FALSE), error = function(e) NULL)
    if (is.null(results_df)) {
      return(build_fallback_batch(batch_df, raw_response, "results_not_dataframe"))
    }
  }
  
  required_cols <- c("code", "decision", "reason", "confidence", "needs_manual_check")
  missing_cols <- setdiff(required_cols, names(results_df))
  
  if (length(missing_cols) > 0) {
    return(build_fallback_batch(
      batch_df,
      raw_response,
      paste("missing_cols:", paste(missing_cols, collapse = ","))
    ))
  }
  
  results_df <- results_df[, required_cols, drop = FALSE]
  results_df$code <- as.character(results_df$code)
  
  merged <- merge(
    batch_df[, "Code", drop = FALSE],
    results_df,
    by.x = "Code",
    by.y = "code",
    all.x = TRUE,
    sort = FALSE
  )
  
  merged$ScreeningDecision <- vapply(
    merged$decision,
    normalize_field,
    character(1),
    allowed = c("accepted", "rejected"),
    default = "error"
  )
  
  merged$ScreeningReason <- ifelse(is.na(merged$reason), "", as.character(merged$reason))
  
  merged$ScreeningConfidence <- vapply(
    merged$confidence,
    normalize_field,
    character(1),
    allowed = c("high", "medium", "low"),
    default = "low"
  )
  
  merged$NeedsManualCheck <- vapply(
    merged$needs_manual_check,
    normalize_field,
    character(1),
    allowed = c("yes", "no"),
    default = "yes"
  )
  
  merged$RawModelOutput <- raw_response
  
  merged$ScreeningDecision[is.na(merged$ScreeningDecision)] <- "error"
  merged$ScreeningReason[is.na(merged$ScreeningReason)] <- "missing_row_in_json"
  merged$ScreeningConfidence[is.na(merged$ScreeningConfidence)] <- "low"
  merged$NeedsManualCheck[is.na(merged$NeedsManualCheck)] <- "yes"
  
  merged[, c(
    "Code", "ScreeningDecision", "ScreeningReason",
    "ScreeningConfidence", "NeedsManualCheck", "RawModelOutput"
  )]
}

# ------------------------------------------------------------
# 10. HELPER: build one batch prompt
# ------------------------------------------------------------
build_batch_prompt <- function(batch_df) {
  paper_blocks <- paste0(
    "PAPER CODE: ", batch_df$Code, "\n",
    "TITLE: ", batch_df$Title, "\n",
    "KEYWORDS: ", batch_df$Keywords, "\n",
    "ABSTRACT: ", batch_df$Abstract,
    collapse = "\n\n--------------------\n\n"
  )
  
  paste0(
    "Screen the following papers for relevance to the review topic.\n",
    "Return only one valid JSON object with a top-level key called results.\n\n",
    paper_blocks
  )
}

# ------------------------------------------------------------
# 11. MAIN FUNCTION: analyze one batch of papers with Groq
# ------------------------------------------------------------
analyze_batch_with_groq <- function(batch_df) {
  retries <- 0
  raw_response_text <- ""
  
  while (retries < max_retries) {
    user_prompt <- build_batch_prompt(batch_df)
    
    body <- list(
      model = model_name,
      temperature = 0,
      response_format = list(
        type = "json_object"
      ),
      messages = list(
        list(role = "system", content = full_system_request),
        list(role = "user", content = user_prompt)
      )
    )
    
    response <- tryCatch({
      POST(
        url = "https://api.groq.com/openai/v1/chat/completions",
        add_headers(
          Authorization = paste("Bearer", api_key),
          `Content-Type` = "application/json"
        ),
        body = toJSON(body, auto_unbox = TRUE, null = "null"),
        timeout(request_timeout)
      )
    }, error = function(e) {
      message("Request error: ", e$message)
      NULL
    })
    
    if (is.null(response)) {
      retries <- retries + 1
      Sys.sleep(5 * retries)
      next
    }
    
    parsed_response <- tryCatch(
      content(response, as = "parsed", type = "application/json"),
      error = function(e) NULL
    )
    
    if (status_code(response) == 429) {
      message("Rate limit hit. Waiting before retry...")
      retries <- retries + 1
      Sys.sleep(15 * retries)
      next
    }
    
    if (status_code(response) >= 400) {
      message("HTTP error ", status_code(response))
      if (!is.null(parsed_response$error$message)) {
        message("API message: ", parsed_response$error$message)
      }
      retries <- retries + 1
      Sys.sleep(5 * retries)
      next
    }
    
    raw_response_text <- NULL
    if (!is.null(parsed_response$choices) &&
        length(parsed_response$choices) > 0 &&
        !is.null(parsed_response$choices[[1]]$message$content)) {
      raw_response_text <- parsed_response$choices[[1]]$message$content
    }
    
    if (is.null(raw_response_text) || raw_response_text == "") {
      message("Empty response text.")
      retries <- retries + 1
      Sys.sleep(5 * retries)
      next
    }
    
    message("Raw response received for batch starting with ", batch_df$Code[1])
    
    parsed_json <- safe_parse_json(raw_response_text)
    
    if (is.null(parsed_json)) {
      message("Invalid JSON for batch starting with ", batch_df$Code[1])
      retries <- retries + 1
      Sys.sleep(5 * retries)
      next
    }
    
    Sys.sleep(sleep_between_requests)
    return(standardize_batch_output(parsed_json, batch_df, raw_response_text))
  }
  
  build_fallback_batch(batch_df, raw_response_text, "max_retries_reached")
}

# ------------------------------------------------------------
# 12. READ INPUT DATASET
# ------------------------------------------------------------
data <- read.csv(input_file, stringsAsFactors = FALSE)
print(names(data))

# ------------------------------------------------------------
# 13. OPTIONAL COLUMN RENAME IF NEEDED
# ------------------------------------------------------------
# names(data)[names(data) == "Document Title"] <- "Title"
# names(data)[names(data) == "Author Keywords"] <- "Keywords"
# names(data)[names(data) == "Abstract Note"] <- "Abstract"

# Ensure required columns exist
required_input_cols <- c("Title", "Abstract")
missing_input_cols <- setdiff(required_input_cols, names(data))
if (length(missing_input_cols) > 0) {
  stop("Missing required input columns: ", paste(missing_input_cols, collapse = ", "))
}

# If Keywords missing, create empty column
if (!"Keywords" %in% names(data)) {
  data$Keywords <- ""
}

# Keep only rows with title and abstract
# If you prefer to screen also papers without abstract, remove this filter.
data <- data[!is.na(data$Abstract) & data$Abstract != "", ]
data <- data[!is.na(data$Title) & data$Title != "", ]

# Create Code if missing
if (!"Code" %in% names(data)) {
  data$Code <- paste0("P", seq_len(nrow(data)))
}

# Truncate long fields to stabilize batching
# Do not overwrite originals if you want to preserve them separately.
data$Abstract <- truncate_text(data$Abstract, max_abstract_chars)
data$Keywords <- truncate_text(data$Keywords, max_keywords_chars)

# ------------------------------------------------------------
# 14. INITIALIZE OUTPUT COLUMNS
# ------------------------------------------------------------
data$ScreeningDecision <- NA_character_
data$ScreeningReason <- NA_character_
data$ScreeningConfidence <- NA_character_
data$NeedsManualCheck <- NA_character_
data$RawModelOutput <- NA_character_

# ------------------------------------------------------------
# 15. PROCESS IN BATCHES
# ------------------------------------------------------------
n <- nrow(data)
batch_starts <- seq(1, n, by = batch_size)

for (start_idx in batch_starts) {
  end_idx <- min(start_idx + batch_size - 1, n)
  batch_df <- data[start_idx:end_idx, c("Code", "Title", "Keywords", "Abstract"), drop = FALSE]
  
  message("Processing batch: rows ", start_idx, " to ", end_idx)
  
  batch_results <- analyze_batch_with_groq(batch_df)
  
  match_idx <- match(batch_results$Code, data$Code)
  
  data$ScreeningDecision[match_idx] <- batch_results$ScreeningDecision
  data$ScreeningReason[match_idx] <- batch_results$ScreeningReason
  data$ScreeningConfidence[match_idx] <- batch_results$ScreeningConfidence
  data$NeedsManualCheck[match_idx] <- batch_results$NeedsManualCheck
  data$RawModelOutput[match_idx] <- batch_results$RawModelOutput
  
  write.csv(data, output_file_csv, row.names = FALSE)
  write.table(data, output_file_tsv, sep = "\t", row.names = FALSE, quote = FALSE)
  
  percentage_completion <- round((end_idx / n) * 100, 2)
  message(percentage_completion, "% completed")
}

# ------------------------------------------------------------
# 16. FINAL OUTPUT
# ------------------------------------------------------------
message("Step 1 screening completed successfully.")