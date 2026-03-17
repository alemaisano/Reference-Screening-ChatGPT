# ============================================================
# Batch LLM-assisted screening and coding pipeline
# Topic: road networks + biodiversity / ecological connectivity
# Model: Gemini 2.5 Flash Lite
# Strategy: 10 papers per request, JSON array output
# ============================================================

library(httr)
library(jsonlite)

# ------------------------------------------------------------
# 1. API KEY
# ------------------------------------------------------------
api_key <- Sys.getenv("GOOGLE_API_KEY")
if (api_key == "") {
  stop("GOOGLE_API_KEY not found. Please add it to your .Renviron file.")
}

# ------------------------------------------------------------
# 2. MODEL SELECTION
# ------------------------------------------------------------
# Flash-lite is usually a better choice for high-throughput screening
model_name <- "gemini-2.5-flash-lite"

# ------------------------------------------------------------
# 3. SETTINGS
# ------------------------------------------------------------
batch_size <- 10
max_retries <- 3
sleep_between_requests <- 5   # helps with RPM limits
output_file_csv <- "output/screened_and_coded_output_batch.csv"
output_file_tsv <- "output/screened_and_coded_output_batch.txt"

# ------------------------------------------------------------
# 4. LOAD THE SYSTEM PROMPT
# ------------------------------------------------------------
system_request <- readLines(
  "prompts/prototype_roadnet_biodconnectivity.txt",
  warn = FALSE
)
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
normalize_field <- function(value, allowed = NULL, default = "unclear") {
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
# 7. HELPER: build empty fallback rows for a failed batch
# ------------------------------------------------------------
build_fallback_batch <- function(batch_df, raw_response = "", error_msg = "invalid_json") {
  data.frame(
    Code = batch_df$Code,
    Decision = "error",
    Reason = error_msg,
    IntegrationType = "unclear",
    SpatialScale = "unclear",
    MethodType = "unclear",
    InteractionDirection = "unclear",
    ConnectivityFocus = "unclear",
    Notes = "",
    RawModelOutput = raw_response,
    stringsAsFactors = FALSE
  )
}

# ------------------------------------------------------------
# 8. HELPER: standardize parsed batch output
# ------------------------------------------------------------
standardize_batch_output <- function(parsed_json, batch_df, raw_response = "") {
  # We expect a JSON array / dataframe with one row per paper.
  # If structure is wrong, return fallback rows.
  if (is.null(parsed_json)) {
    return(build_fallback_batch(batch_df, raw_response, "parsed_json_null"))
  }
  
  if (!is.data.frame(parsed_json)) {
    # Try coercion if parsed_json is a list of objects
    parsed_json <- tryCatch(as.data.frame(parsed_json, stringsAsFactors = FALSE), error = function(e) NULL)
    if (is.null(parsed_json)) {
      return(build_fallback_batch(batch_df, raw_response, "parsed_json_not_dataframe"))
    }
  }
  
  required_cols <- c(
    "code", "decision", "reason", "integration_type",
    "spatial_scale", "method_type", "interaction_direction",
    "connectivity_focus", "notes"
  )
  
  missing_cols <- setdiff(required_cols, names(parsed_json))
  if (length(missing_cols) > 0) {
    return(build_fallback_batch(
      batch_df,
      raw_response,
      paste("missing_cols:", paste(missing_cols, collapse = ","))
    ))
  }
  
  # Keep only relevant columns
  parsed_json <- parsed_json[, required_cols, drop = FALSE]
  
  # Normalize code
  parsed_json$code <- as.character(parsed_json$code)
  
  # Match parsed rows back to the original batch order
  merged <- merge(
    batch_df[, "Code", drop = FALSE],
    parsed_json,
    by.x = "Code",
    by.y = "code",
    all.x = TRUE,
    sort = FALSE
  )
  
  # Normalize fields
  merged$Decision <- vapply(
    merged$decision,
    normalize_field,
    character(1),
    allowed = c("accepted", "rejected", "unsure"),
    default = "unsure"
  )
  
  merged$Reason <- ifelse(is.na(merged$reason), "", as.character(merged$reason))
  
  merged$IntegrationType <- vapply(
    merged$integration_type,
    normalize_field,
    character(1),
    allowed = c("conceptual", "analytical", "sequential", "soft_coupling", "strong_coupling", "unclear"),
    default = "unclear"
  )
  
  merged$SpatialScale <- vapply(
    merged$spatial_scale,
    normalize_field,
    character(1),
    allowed = c("local", "regional", "supraregional", "multiple", "unclear"),
    default = "unclear"
  )
  
  merged$MethodType <- vapply(
    merged$method_type,
    normalize_field,
    character(1),
    allowed = c("empirical", "modeling", "review", "conceptual", "mixed", "unclear"),
    default = "unclear"
  )
  
  merged$InteractionDirection <- vapply(
    merged$interaction_direction,
    normalize_field,
    character(1),
    allowed = c("transport_to_ecosystem", "ecosystem_to_transport", "bidirectional", "unclear"),
    default = "unclear"
  )
  
  merged$ConnectivityFocus <- vapply(
    merged$connectivity_focus,
    normalize_field,
    character(1),
    allowed = c("yes", "no", "partial", "unclear"),
    default = "unclear"
  )
  
  merged$Notes <- ifelse(is.na(merged$notes), "", as.character(merged$notes))
  merged$RawModelOutput <- raw_response
  
  # Fill any missing rows with fallback values
  merged$Decision[is.na(merged$Decision)] <- "error"
  merged$Reason[is.na(merged$Reason)] <- "missing_row_in_json"
  merged$IntegrationType[is.na(merged$IntegrationType)] <- "unclear"
  merged$SpatialScale[is.na(merged$SpatialScale)] <- "unclear"
  merged$MethodType[is.na(merged$MethodType)] <- "unclear"
  merged$InteractionDirection[is.na(merged$InteractionDirection)] <- "unclear"
  merged$ConnectivityFocus[is.na(merged$ConnectivityFocus)] <- "unclear"
  merged$Notes[is.na(merged$Notes)] <- ""
  
  merged[, c(
    "Code", "Decision", "Reason", "IntegrationType",
    "SpatialScale", "MethodType", "InteractionDirection",
    "ConnectivityFocus", "Notes", "RawModelOutput"
  )]
}

# ------------------------------------------------------------
# 9. HELPER: build one batch prompt
# ------------------------------------------------------------
build_batch_prompt <- function(batch_df) {
  # Build a compact but structured prompt for multiple papers
  paper_blocks <- paste0(
    "PAPER CODE: ", batch_df$Code, "\n",
    "TITLE: ", batch_df$Title, "\n",
    "ABSTRACT: ", batch_df$Abstract,
    collapse = "\n\n--------------------\n\n"
  )
  
  paste0(
    "Classify the following papers.\n",
    "Return only a valid JSON array.\n\n",
    paper_blocks
  )
}

# ------------------------------------------------------------
# 10. MAIN FUNCTION: analyze one batch of papers
# ------------------------------------------------------------
analyze_batch_with_gemini <- function(batch_df) {
  retries <- 0
  raw_response_text <- ""
  
  while (retries < max_retries) {
    user_prompt <- build_batch_prompt(batch_df)
    
    # Strong schema constraint to reduce invalid JSON
    body <- list(
      system_instruction = list(
        parts = list(
          list(text = full_system_request)
        )
      ),
      contents = list(
        list(
          role = "user",
          parts = list(
            list(text = user_prompt)
          )
        )
      ),
      generationConfig = list(
        temperature = 0,
        maxOutputTokens = 2500,
        responseMimeType = "application/json",
        responseSchema = list(
          type = "ARRAY",
          items = list(
            type = "OBJECT",
            properties = list(
              code = list(type = "STRING"),
              decision = list(
                type = "STRING",
                enum = list("accepted", "rejected", "unsure")
              ),
              reason = list(type = "STRING"),
              integration_type = list(
                type = "STRING",
                enum = list(
                  "conceptual", "analytical", "sequential",
                  "soft_coupling", "strong_coupling", "unclear"
                )
              ),
              spatial_scale = list(
                type = "STRING",
                enum = list("local", "regional", "supraregional", "multiple", "unclear")
              ),
              method_type = list(
                type = "STRING",
                enum = list("empirical", "modeling", "review", "conceptual", "mixed", "unclear")
              ),
              interaction_direction = list(
                type = "STRING",
                enum = list("transport_to_ecosystem", "ecosystem_to_transport", "bidirectional", "unclear")
              ),
              connectivity_focus = list(
                type = "STRING",
                enum = list("yes", "no", "partial", "unclear")
              ),
              notes = list(type = "STRING")
            ),
            required = list(
              "code", "decision", "reason", "integration_type",
              "spatial_scale", "method_type", "interaction_direction",
              "connectivity_focus", "notes"
            )
          )
        )
      )
    )
    
    response <- tryCatch({
      POST(
        url = paste0(
          "https://generativelanguage.googleapis.com/v1beta/models/",
          model_name,
          ":generateContent"
        ),
        add_headers(
          "x-goog-api-key" = api_key,
          "Content-Type" = "application/json"
        ),
        body = toJSON(body, auto_unbox = TRUE, null = "null"),
        timeout(60)
      )
    }, error = function(e) {
      message("Request error: ", e$message)
      NULL
    })
    
    if (is.null(response)) {
      retries <- retries + 1
      Sys.sleep(10 * retries)
      next
    }
    
    parsed_response <- tryCatch(
      content(response, as = "parsed", type = "application/json"),
      error = function(e) NULL
    )
    
    # Handle 429 / rate limits explicitly
    if (status_code(response) == 429) {
      message("Rate limit hit. Waiting before retry...")
      retries <- retries + 1
      Sys.sleep(30 * retries)
      next
    }
    
    if (status_code(response) >= 400) {
      message("HTTP error ", status_code(response))
      if (!is.null(parsed_response$error$message)) {
        message("API message: ", parsed_response$error$message)
      }
      retries <- retries + 1
      Sys.sleep(10 * retries)
      next
    }
    
    # Gemini text is usually here
    raw_response_text <- NULL
    if (!is.null(parsed_response$candidates) &&
        length(parsed_response$candidates) > 0 &&
        !is.null(parsed_response$candidates[[1]]$content$parts) &&
        length(parsed_response$candidates[[1]]$content$parts) > 0 &&
        !is.null(parsed_response$candidates[[1]]$content$parts[[1]]$text)) {
      raw_response_text <- parsed_response$candidates[[1]]$content$parts[[1]]$text
    }
    
    if (is.null(raw_response_text) || raw_response_text == "") {
      message("Empty response text.")
      retries <- retries + 1
      Sys.sleep(10 * retries)
      next
    }
    
    # Print a compact debug message only
    message("Raw response received for batch starting with ", batch_df$Code[1])
    
    parsed_json <- safe_parse_json(raw_response_text)
    
    if (is.null(parsed_json)) {
      message("Invalid JSON for batch starting with ", batch_df$Code[1])
      retries <- retries + 1
      Sys.sleep(10 * retries)
      next
    }
    
    # Success
    Sys.sleep(sleep_between_requests)
    return(standardize_batch_output(parsed_json, batch_df, raw_response_text))
  }
  
  # If all retries failed
  build_fallback_batch(batch_df, raw_response_text, "max_retries_reached")
}

# ------------------------------------------------------------
# 11. READ INPUT DATASET
# ------------------------------------------------------------
data <- read.csv(
  "database_output_raw/scopus_export_prototype_1.csv",
  stringsAsFactors = FALSE
)

# Optional: inspect names if needed
print(names(data))

# If necessary, rename columns here
# names(data)[names(data) == "Document Title"] <- "Title"
# names(data)[names(data) == "Year"] <- "Year"

# Keep only rows with title and abstract
data <- data[!is.na(data$Abstract) & data$Abstract != "", ]
data <- data[!is.na(data$Title) & data$Title != "", ]

# Create Code if missing
if (!"Code" %in% names(data)) {
  data$Code <- paste0("P", seq_len(nrow(data)))
}

# Optional: shorten very long abstracts to reduce tokens
truncate_text <- function(x, max_chars = 2500) {
  ifelse(nchar(x) > max_chars, substr(x, 1, max_chars), x)
}
data$Abstract <- truncate_text(data$Abstract, max_chars = 2500)

# Initialize output columns now, so you can inspect progress while running
data$Decision <- NA_character_
data$Reason <- NA_character_
data$IntegrationType <- NA_character_
data$SpatialScale <- NA_character_
data$MethodType <- NA_character_
data$InteractionDirection <- NA_character_
data$ConnectivityFocus <- NA_character_
data$Notes <- NA_character_
data$RawModelOutput <- NA_character_

# ------------------------------------------------------------
# 12. PROCESS IN BATCHES OF 10
# ------------------------------------------------------------
n <- nrow(data)
batch_starts <- seq(1, n, by = batch_size)

for (start_idx in batch_starts) {
  end_idx <- min(start_idx + batch_size - 1, n)
  batch_df <- data[start_idx:end_idx, c("Code", "Title", "Abstract"), drop = FALSE]
  
  message("Processing batch: rows ", start_idx, " to ", end_idx)
  
  batch_results <- analyze_batch_with_gemini(batch_df)
  
  # Write results back into the original dataframe
  match_idx <- match(batch_results$Code, data$Code)
  
  data$Decision[match_idx] <- batch_results$Decision
  data$Reason[match_idx] <- batch_results$Reason
  data$IntegrationType[match_idx] <- batch_results$IntegrationType
  data$SpatialScale[match_idx] <- batch_results$SpatialScale
  data$MethodType[match_idx] <- batch_results$MethodType
  data$InteractionDirection[match_idx] <- batch_results$InteractionDirection
  data$ConnectivityFocus[match_idx] <- batch_results$ConnectivityFocus
  data$Notes[match_idx] <- batch_results$Notes
  data$RawModelOutput[match_idx] <- batch_results$RawModelOutput
  
  # Save partial output after every batch
  write.csv(data, output_file_csv, row.names = FALSE)
  write.table(data, output_file_tsv, sep = "\t", row.names = FALSE, quote = FALSE)
  
  percentage_completion <- round((end_idx / n) * 100, 2)
  message(percentage_completion, "% completed")
}

# ------------------------------------------------------------
# 13. FINAL OUTPUT
# ------------------------------------------------------------
message("Batch screening and coding completed successfully.")