# ============================================================
# Step 1 — LLM-assisted screening pipeline
# Resume from P744 onward without overwriting previous rows
# ============================================================

library(httr)
library(jsonlite)

# ------------------------------------------------------------
# 1. API KEY
# ------------------------------------------------------------
api_key <- Sys.getenv("GROQ_API_KEY")
if (api_key == "") stop("GROQ_API_KEY not found.")

# ------------------------------------------------------------
# 2. SETTINGS
# ------------------------------------------------------------
model_name          <- "meta-llama/llama-4-scout-17b-16e-instruct"
batch_size          <- 2
max_retries         <- 4
request_timeout     <- 60
max_abstract_chars  <- 2500
max_keywords_chars  <- 600
sleep_between_calls <- 10

input_file          <- "database_output_raw/scopus_export_prototype_1.csv"
resume_file_csv     <- "output/screened_step1.1_groq2.csv"   # existing partial output
output_file_csv     <- "output/screened_step1.1_groq2_resumed.csv"
output_file_tsv     <- "output/screened_step1.1_groq2_resumed.txt"
prompt_file         <- "prompts/prototype2_roadnet_biodconnectivity.txt"

resume_from_code    <- "P744"

# ------------------------------------------------------------
# 3. SYSTEM PROMPT
# ------------------------------------------------------------
system_request <- paste(
  readLines(prompt_file, warn = FALSE, encoding = "UTF-8"),
  collapse = "\n"
)

# ------------------------------------------------------------
# 4. HELPERS
# ------------------------------------------------------------
truncate_text <- function(x, max_chars) {
  x <- ifelse(is.na(x), "", x)
  ifelse(nchar(x) > max_chars, substr(x, 1, max_chars), x)
}

normalize_field <- function(value, allowed = NULL, default = NA_character_) {
  if (is.null(value) || length(value) == 0 || is.na(value) || trimws(value) == "") {
    return(default)
  }
  value <- trimws(tolower(as.character(value)))
  if (!is.null(allowed) && !(value %in% allowed)) return(default)
  value
}

build_fallback_batch <- function(batch_df, raw_response = "", error_msg = "error") {
  data.frame(
    Code                = batch_df$Code,
    ScreeningDecision   = "error",
    ScreeningReason     = error_msg,
    ScreeningConfidence = "low",
    RawModelOutput      = raw_response,
    stringsAsFactors    = FALSE
  )
}

# ------------------------------------------------------------
# 5. DYNAMIC JSON SCHEMA
# ------------------------------------------------------------
build_response_schema <- function(batch_df) {
  list(
    type = "json_schema",
    json_schema = list(
      name   = "screening_results",
      strict = TRUE,
      schema = list(
        type                 = "object",
        required             = list("results"),
        additionalProperties = FALSE,
        properties           = list(
          results = list(
            type     = "array",
            minItems = nrow(batch_df),
            maxItems = nrow(batch_df),
            items    = list(
              type                 = "object",
              required             = list("code", "decision", "reason", "confidence"),
              additionalProperties = FALSE,
              properties           = list(
                code = list(
                  type = "string",
                  enum = as.list(batch_df$Code)
                ),
                decision = list(
                  type = "string",
                  enum = list("accepted", "rejected")
                ),
                reason = list(type = "string"),
                confidence = list(
                  type = "string",
                  enum = list("high", "medium", "low")
                )
              )
            )
          )
        )
      )
    )
  )
}

# ------------------------------------------------------------
# 6. BUILD USER PROMPT
# ------------------------------------------------------------
build_batch_prompt <- function(batch_df) {
  paper_blocks <- paste0(
    "PAPER CODE: ", batch_df$Code, "\n",
    "TITLE: ",      batch_df$Title, "\n",
    "KEYWORDS: ",   batch_df$Keywords, "\n",
    "ABSTRACT: ",   batch_df$Abstract,
    collapse = "\n\n--------------------\n\n"
  )
  
  paste0(
    "Screen the following ", nrow(batch_df), " papers.\n",
    "Return exactly one result per paper using the paper code as provided.\n\n",
    paper_blocks
  )
}

# ------------------------------------------------------------
# 7. STANDARDIZE PARSED OUTPUT
# ------------------------------------------------------------
standardize_batch_output <- function(parsed_json, batch_df, raw_response = "") {
  if (is.null(parsed_json)) {
    return(build_fallback_batch(batch_df, raw_response, "parsed_json_null"))
  }
  
  if (!is.list(parsed_json) || is.null(parsed_json$results)) {
    return(build_fallback_batch(batch_df, raw_response, "missing_results_array"))
  }
  
  results_df <- parsed_json$results
  
  if (!is.data.frame(results_df)) {
    results_df <- tryCatch(
      as.data.frame(results_df, stringsAsFactors = FALSE),
      error = function(e) NULL
    )
    if (is.null(results_df)) {
      return(build_fallback_batch(batch_df, raw_response, "results_not_dataframe"))
    }
  }
  
  required_cols <- c("code", "decision", "reason", "confidence")
  missing_cols  <- setdiff(required_cols, names(results_df))
  if (length(missing_cols) > 0) {
    return(build_fallback_batch(
      batch_df, raw_response,
      paste("missing_cols:", paste(missing_cols, collapse = ","))
    ))
  }
  
  results_df <- results_df[, required_cols, drop = FALSE]
  results_df$code <- as.character(results_df$code)
  
  merged <- merge(
    batch_df[, "Code", drop = FALSE],
    results_df,
    by.x  = "Code",
    by.y  = "code",
    all.x = TRUE,
    sort  = FALSE
  )
  
  merged$ScreeningDecision <- vapply(
    merged$decision, normalize_field, character(1),
    allowed = c("accepted", "rejected"), default = "error"
  )
  
  merged$ScreeningReason <- ifelse(is.na(merged$reason), "", as.character(merged$reason))
  
  merged$ScreeningConfidence <- vapply(
    merged$confidence, normalize_field, character(1),
    allowed = c("high", "medium", "low"), default = "low"
  )
  
  merged$RawModelOutput <- raw_response
  
  merged$ScreeningDecision[is.na(merged$ScreeningDecision)]     <- "error"
  merged$ScreeningReason[is.na(merged$ScreeningReason)]         <- "missing_row_in_json"
  merged$ScreeningConfidence[is.na(merged$ScreeningConfidence)] <- "low"
  
  merged[, c("Code", "ScreeningDecision", "ScreeningReason",
             "ScreeningConfidence", "RawModelOutput")]
}

# ------------------------------------------------------------
# 8. API CALL CON ATTESA DINAMICA (RATE LIMIT)
# ------------------------------------------------------------
analyze_batch <- function(batch_df) {
  retries <- 0
  raw_response_text <- ""
  
  while (retries < max_retries) {
    body <- list(
      model           = model_name,
      temperature     = 0,
      response_format = build_response_schema(batch_df),
      messages        = list(
        list(role = "system", content = system_request),
        list(role = "user",   content = build_batch_prompt(batch_df))
      )
    )
    
    response <- tryCatch({
      POST(
        url = "https://api.groq.com/openai/v1/chat/completions",
        add_headers(
          Authorization  = paste("Bearer", api_key),
          `Content-Type` = "application/json"
        ),
        body    = toJSON(body, auto_unbox = TRUE, null = "null"),
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
    
    # --- GESTIONE RATE LIMIT (HTTP 429) ---
    if (status_code(response) == 429) {
      # Prova a leggere l'header 'retry-after' (spesso è vuoto su Groq)
      wait_time <- as.numeric(headers(response)[["retry-after"]])
      
      # Se manca, estraiamo i secondi dal messaggio di errore JSON
      if (is.na(wait_time)) {
        err_content <- content(response, as = "parsed")
        err_msg <- err_content$error$message
        
        # Cerca pattern come "in 1.5s" o "in 15s" nel messaggio di Groq
        wait_match <- regexec("in ([0-9\\.]+)s", err_msg)
        wait_val <- regmatches(err_msg, wait_match)[[1]][2]
        
        if (!is.na(wait_val)) {
          wait_time <- as.numeric(wait_val)
        } else {
          wait_time <- 20 # Fallback prudente se non trova il tempo nel testo
        }
      }
      
      # Aggiungiamo un margine di sicurezza di 1 secondo
      wait_time <- wait_time + 1
      
      message("!!! Rate Limit !!! Groq richiede attesa di ", wait_time, " secondi...")
      Sys.sleep(wait_time)
      retries <- retries + 1
      next
    }
    
    # --- ALTRI ERRORI HTTP ---
    if (status_code(response) >= 400) {
      parsed_err <- content(response, as = "parsed")
      message("HTTP Error ", status_code(response), ": ", parsed_err$error$message)
      retries <- retries + 1
      Sys.sleep(5 * retries)
      next
    }
    
    # --- PARSING OK ---
    parsed_response <- content(response, as = "parsed")
    raw_response_text <- parsed_response$choices[[1]]$message$content
    
    parsed_json <- tryCatch(
      fromJSON(raw_response_text, simplifyDataFrame = TRUE),
      error = function(e) NULL
    )
    
    if (is.null(parsed_json)) {
      message("JSON non valido. Riprovo...")
      retries <- retries + 1
      Sys.sleep(2)
      next
    }
    
    message("Batch OK | Partenza da: ", batch_df$Code[1])
    Sys.sleep(sleep_between_calls)
    return(standardize_batch_output(parsed_json, batch_df, raw_response_text))
  }
  
  build_fallback_batch(batch_df, raw_response_text, "max_retries_reached_rate_limit")
}

# ------------------------------------------------------------
# 9. LOAD RAW DATA
# ------------------------------------------------------------
data <- read.csv(input_file, stringsAsFactors = FALSE)
message("Columns found: ", paste(names(data), collapse = ", "))

required_input_cols <- c("Title", "Abstract")
missing_input_cols  <- setdiff(required_input_cols, names(data))
if (length(missing_input_cols) > 0) {
  stop("Missing columns: ", paste(missing_input_cols, collapse = ", "))
}

if (!"Keywords" %in% names(data)) data$Keywords <- ""

data <- data[!is.na(data$Abstract) & data$Abstract != "", ]
data <- data[!is.na(data$Title)    & data$Title    != "", ]

if (!"Code" %in% names(data)) data$Code <- paste0("P", seq_len(nrow(data)))

data$Abstract <- truncate_text(data$Abstract, max_abstract_chars)
data$Keywords <- truncate_text(data$Keywords, max_keywords_chars)

# ------------------------------------------------------------
# 10. INITIALISE OUTPUT COLUMNS
# ------------------------------------------------------------
data$ScreeningDecision   <- NA_character_
data$ScreeningReason     <- NA_character_
data$ScreeningConfidence <- NA_character_
data$RawModelOutput      <- NA_character_

# ------------------------------------------------------------
# 11. LOAD PREVIOUS OUTPUT AND MERGE
# ------------------------------------------------------------
if (file.exists(resume_file_csv)) {
  old <- read.csv(resume_file_csv, stringsAsFactors = FALSE)
  
  keep_cols <- c("Code", "ScreeningDecision", "ScreeningReason",
                 "ScreeningConfidence", "RawModelOutput")
  keep_cols <- intersect(keep_cols, names(old))
  
  old <- old[, keep_cols, drop = FALSE]
  
  idx <- match(data$Code, old$Code)
  
  data$ScreeningDecision   <- old$ScreeningDecision[idx]
  data$ScreeningReason     <- old$ScreeningReason[idx]
  data$ScreeningConfidence <- old$ScreeningConfidence[idx]
  data$RawModelOutput      <- old$RawModelOutput[idx]
  
  message("Loaded previous output from: ", resume_file_csv)
} else {
  stop("Resume file not found: ", resume_file_csv)
}

# ------------------------------------------------------------
# 12. FIND START ROW
# ------------------------------------------------------------
start_resume_idx <- match(resume_from_code, data$Code)
if (is.na(start_resume_idx)) {
  stop("resume_from_code not found in data: ", resume_from_code)
}

message("Resuming from code ", resume_from_code, " at row ", start_resume_idx)

# ------------------------------------------------------------
# 13. MAIN LOOP - SOVRASCRITTURA FORZATA DA START_RESUME_IDX
# ------------------------------------------------------------
n <- nrow(data)
batch_starts <- seq(start_resume_idx, n, by = batch_size)

for (start_idx in batch_starts) {
  end_idx <- min(start_idx + batch_size - 1, n)
  batch_codes <- data$Code[start_idx:end_idx]
  
  # NOTA: Qui ho rimosso il controllo "if (all(!is.na(...))) next"
  # Questo assicura che da P744 in poi il codice esegua SEMPRE l'API
  
  batch_df <- data[start_idx:end_idx,
                   c("Code", "Title", "Keywords", "Abstract"),
                   drop = FALSE]
  
  message("--- Elaborazione Batch: righe ", start_idx, " – ", end_idx, 
          " | Codes: ", paste(batch_codes, collapse = ", "))
  
  batch_results <- analyze_batch(batch_df)
  
  # Mappatura precisa basata sul codice per sovrascrivere nel dataframe principale
  match_idx <- match(batch_results$Code, data$Code)
  
  data$ScreeningDecision[match_idx]   <- batch_results$ScreeningDecision
  data$ScreeningReason[match_idx]     <- batch_results$ScreeningReason
  data$ScreeningConfidence[match_idx] <- batch_results$ScreeningConfidence
  data$RawModelOutput[match_idx]      <- batch_results$RawModelOutput
  
  # Salvataggio immediato (sovrascrive il file di output a ogni batch)
  write.csv(data, output_file_csv, row.names = FALSE)
  write.table(data, output_file_tsv, sep = "\t", row.names = FALSE, quote = FALSE)
  
  progress <- round(end_idx / n * 100, 1)
  message(progress, "% completato. Decisioni: ", 
          paste(batch_results$ScreeningDecision, collapse = ", "))
}

message("Screening terminato con sovrascrittura da ", resume_from_code, " in poi.")