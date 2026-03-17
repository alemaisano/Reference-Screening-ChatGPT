# ============================================================
# LLM-assisted screening and coding pipeline for papers
# Topic: road networks + biodiversity / ecological connectivity
# Model: Google Gemini 2.5 Flash
# ============================================================

# Load required libraries:
# - httr: used to send HTTP requests to the Gemini API
# - jsonlite: used to convert R objects to JSON and parse JSON responses
library(httr)
library(jsonlite)

# ------------------------------------------------------------
# 1. API KEY
# ------------------------------------------------------------

# Read the Google API key from the environment variable.
# IMPORTANT:
# Do NOT hardcode your API key directly in the script.
# Instead, save it in your .Renviron file as:
# GOOGLE_API_KEY=your_key_here
api_key <- Sys.getenv("GOOGLE_API_KEY")

# Stop the script immediately if the API key is missing.
if (api_key == "") {
  stop("GOOGLE_API_KEY not found. Please add it to your .Renviron file.")
}

# ------------------------------------------------------------
# 2. MODEL SELECTION
# ------------------------------------------------------------

# Define the Gemini model to use.
# For your prototype, Gemini 2.5 Flash is a good balance of cost and speed.
model_name <- "gemini-2.5-flash"

# ------------------------------------------------------------
# 3. GLOBAL COUNTER
# ------------------------------------------------------------

# Global counter used to print progress while iterating through papers.
counter <- 0

# ------------------------------------------------------------
# 4. LOAD THE SYSTEM PROMPT
# ------------------------------------------------------------

# Read the external prompt file containing the review instructions.
# This prompt should define:
# - inclusion / exclusion criteria
# - coding categories
# - required JSON output format
system_request <- readLines(
  "prompts/prototype_roadnet_biodconnectivity.txt",
  warn = FALSE
)

# Collapse the prompt into one single string so it can be passed to the API.
full_system_request <- paste(system_request, collapse = "\n")

# ------------------------------------------------------------
# 5. HELPER FUNCTION:
#    Safely extract the text content from the Gemini response
# ------------------------------------------------------------

safe_extract_text <- function(parsed_response) {
  # Gemini responses are nested in a structure like:
  # candidates[[1]]$content$parts[[1]]$text
  #
  # This helper function safely checks whether the expected fields exist.
  # If the response structure is incomplete or unexpected, it returns NULL
  # instead of crashing the script.
  
  if (!is.null(parsed_response$candidates) &&
      length(parsed_response$candidates) > 0 &&
      !is.null(parsed_response$candidates[[1]]$content$parts) &&
      length(parsed_response$candidates[[1]]$content$parts) > 0 &&
      !is.null(parsed_response$candidates[[1]]$content$parts[[1]]$text)) {
    return(parsed_response$candidates[[1]]$content$parts[[1]]$text)
  }
  
  return(NULL)
}

# ------------------------------------------------------------
# 6. HELPER FUNCTION:
#    Safely parse the model output as JSON
# ------------------------------------------------------------

safe_parse_json <- function(text_output) {
  # Even when instructed to return only JSON, the model may sometimes wrap
  # the output inside markdown code fences, such as:
  #
  # ```json
  # {...}
  # ```
  #
  # This function removes those wrappers and then tries to parse the JSON.
  # If parsing fails, it returns NULL.
  
  cleaned_text <- gsub("^```json\\s*|^```\\s*|\\s*```$", "", text_output)
  cleaned_text <- trimws(cleaned_text)
  
  tryCatch(
    fromJSON(cleaned_text),
    error = function(e) NULL
  )
}

# ------------------------------------------------------------
# 7. HELPER FUNCTION:
#    Normalize and validate one field returned by the model
# ------------------------------------------------------------

normalize_field <- function(value, allowed = NULL, default = "unclear") {
  # This function standardizes values returned by the model:
  # - converts them to lowercase
  # - trims whitespace
  # - checks whether they belong to an allowed set
  #
  # If the value is missing, empty, or outside the allowed values,
  # it returns the specified default value.
  
  if (is.null(value) || length(value) == 0 || is.na(value) || trimws(value) == "") {
    return(default)
  }
  
  value <- trimws(tolower(as.character(value)))
  
  if (!is.null(allowed) && !(value %in% allowed)) {
    return(default)
  }
  
  return(value)
}

# ------------------------------------------------------------
# 8. MAIN FUNCTION:
#    Analyze one paper with Gemini
# ------------------------------------------------------------

analyze_text_with_gemini <- function(Code, Title, Abstract, total_n) {
  
  # Number of retries allowed in case of API issues or invalid output
  retries <- 0
  max_retries <- 3
  
  # Track whether we successfully received a valid answer
  valid_response_received <- FALSE
  
  # Default result structure.
  # This ensures that even in case of repeated failure,
  # the function always returns the same fields.
  result <- list(
    Decision = "error",
    Reason = "No valid response received",
    IntegrationType = "unclear",
    SpatialScale = "unclear",
    MethodType = "unclear",
    InteractionDirection = "unclear",
    ConnectivityFocus = "unclear",
    Notes = ""
  )
  
  # Retry loop
  while (!valid_response_received && retries < max_retries) {
    
    # --------------------------------------------------------
    # Build the user prompt for the current paper
    # --------------------------------------------------------
    #
    # We provide:
    # - paper code (for traceability)
    # - title
    # - abstract
    # - explicit instruction to return only JSON
    user_prompt <- paste0(
      "PAPER CODE: ", Code, "\n\n",
      "TITLE:\n", Title, "\n\n",
      "ABSTRACT:\n", Abstract, "\n\n",
      "Classify this paper and return only valid JSON."
    )
    
    # --------------------------------------------------------
    # Build the Gemini request body
    # --------------------------------------------------------
    #
    # system_instruction:
    #   contains the global screening/coding instructions
    #
    # contents:
    #   contains the specific paper currently being analyzed
    #
    # generationConfig:
    #   temperature = 0 for deterministic / stable responses
    #   maxOutputTokens limits output length
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
        maxOutputTokens = 800
      )
    )
    
    # --------------------------------------------------------
    # Send the POST request to the Gemini API
    # --------------------------------------------------------
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
        timeout(50)
      )
    }, error = function(e) {
      message("Request error for ", Code, ": ", e$message)
      NULL
    })
    
    # If the request itself failed, retry
    if (is.null(response)) {
      retries <- retries + 1
      Sys.sleep(2^retries)
      next
    }
    
    # --------------------------------------------------------
    # Parse the raw HTTP response into an R object
    # --------------------------------------------------------
    parsed_response <- tryCatch(
      content(response, as = "parsed", type = "application/json"),
      error = function(e) NULL
    )
    
    # --------------------------------------------------------
    # Handle HTTP-level errors
    # --------------------------------------------------------
    if (status_code(response) >= 400) {
      message("HTTP error ", status_code(response), " for ", Code)
      
      if (!is.null(parsed_response$error$message)) {
        message("API message: ", parsed_response$error$message)
      }
      
      retries <- retries + 1
      Sys.sleep(2^retries)
      next
    }
    
    # --------------------------------------------------------
    # Extract the textual content returned by the model
    # --------------------------------------------------------
    response_text <- safe_extract_text(parsed_response)
    
    if (is.null(response_text) || response_text == "") {
      message("Empty response text for ", Code)
      retries <- retries + 1
      Sys.sleep(2^retries)
      next
    }
    
    # Print model output for debugging / transparency
    print(response_text)
    
    # --------------------------------------------------------
    # Parse the model output as JSON
    # --------------------------------------------------------
    parsed_json <- safe_parse_json(response_text)
    
    if (is.null(parsed_json)) {
      message("Invalid JSON returned for ", Code)
      retries <- retries + 1
      Sys.sleep(2^retries)
      next
    }
    
    # --------------------------------------------------------
    # Extract and normalize each field returned by the model
    # --------------------------------------------------------
    
    # Screening decision
    result$Decision <- normalize_field(
      parsed_json$decision,
      allowed = c("accepted", "rejected", "unsure"),
      default = "unsure"
    )
    
    # Short reason for the decision
    result$Reason <- ifelse(
      is.null(parsed_json$reason),
      "",
      as.character(parsed_json$reason)
    )
    
    # Integration type:
    # conceptual / analytical / sequential / soft_coupling / strong_coupling / unclear
    result$IntegrationType <- normalize_field(
      parsed_json$integration_type,
      allowed = c(
        "conceptual",
        "analytical",
        "sequential",
        "soft_coupling",
        "strong_coupling",
        "unclear"
      ),
      default = "unclear"
    )
    
    # Spatial scale:
    # local / regional / supraregional / multiple / unclear
    result$SpatialScale <- normalize_field(
      parsed_json$spatial_scale,
      allowed = c("local", "regional", "supraregional", "multiple", "unclear"),
      default = "unclear"
    )
    
    # Method type:
    # empirical / modeling / review / conceptual / mixed / unclear
    result$MethodType <- normalize_field(
      parsed_json$method_type,
      allowed = c("empirical", "modeling", "review", "conceptual", "mixed", "unclear"),
      default = "unclear"
    )
    
    # Interaction direction:
    # transport_to_ecosystem / ecosystem_to_transport / bidirectional / unclear
    result$InteractionDirection <- normalize_field(
      parsed_json$interaction_direction,
      allowed = c("transport_to_ecosystem", "ecosystem_to_transport", "bidirectional", "unclear"),
      default = "unclear"
    )
    
    # Whether ecological connectivity is central in the paper
    result$ConnectivityFocus <- normalize_field(
      parsed_json$connectivity_focus,
      allowed = c("yes", "no", "partial", "unclear"),
      default = "unclear"
    )
    
    # Additional notes
    result$Notes <- ifelse(
      is.null(parsed_json$notes),
      "",
      as.character(parsed_json$notes)
    )
    
    # Mark as valid and exit the retry loop
    valid_response_received <- TRUE
  }
  
  # ----------------------------------------------------------
  # Update progress counter and print completion status
  # ----------------------------------------------------------
  .GlobalEnv$counter <- .GlobalEnv$counter + 1
  percentage_completion <- (.GlobalEnv$counter / total_n) * 100
  
  message(
    round(percentage_completion, 2),
    "% completed - ",
    Code,
    " -> ",
    result$Decision
  )
  
  return(result)
}

# ------------------------------------------------------------
# 9. READ THE INPUT DATASET
# ------------------------------------------------------------

# Read the CSV file containing your paper dataset.
#
# IMPORTANT:
# If your file actually has a .csv extension, use:
# "database_output_raw/scopus_export_prototype.csv"
#
# If your file has no extension, keep the current path as written below.
data <- read.csv(
  "database_output_raw/scopus_export_prototype",
  stringsAsFactors = FALSE
)

# Print column names so you can inspect them if needed
print(names(data))

# ------------------------------------------------------------
# 10. OPTIONAL COLUMN RENAMING
# ------------------------------------------------------------
#
# Uncomment and adapt these lines if your CSV uses different column names.
# For example, if Scopus exports them differently.
#
# names(data)[names(data) == "Document Title"] <- "Title"
# names(data)[names(data) == "Abstract"] <- "Abstract"
# names(data)[names(data) == "Year"] <- "Year"

# ------------------------------------------------------------
# 11. CLEAN THE DATA
# ------------------------------------------------------------

# Remove rows with missing or empty abstracts
data <- data[!is.na(data$Abstract) & data$Abstract != "", ]

# Remove rows with missing or empty titles
data <- data[!is.na(data$Title) & data$Title != "", ]

# If the dataset does not contain a Code column, create one automatically
if (!"Code" %in% names(data)) {
  data$Code <- paste0("P", seq_len(nrow(data)))
}

# ------------------------------------------------------------
# 12. APPLY THE GEMINI ANALYSIS TO ALL PAPERS
# ------------------------------------------------------------

results <- mapply(
  FUN = analyze_text_with_gemini,
  Code = data$Code,
  Title = data$Title,
  Abstract = data$Abstract,
  MoreArgs = list(total_n = nrow(data)),
  SIMPLIFY = FALSE
)

# ------------------------------------------------------------
# 13. ADD THE MODEL OUTPUT AS NEW COLUMNS
# ------------------------------------------------------------

data$Decision <- sapply(results, `[[`, "Decision")
data$Reason <- sapply(results, `[[`, "Reason")
data$IntegrationType <- sapply(results, `[[`, "IntegrationType")
data$SpatialScale <- sapply(results, `[[`, "SpatialScale")
data$MethodType <- sapply(results, `[[`, "MethodType")
data$InteractionDirection <- sapply(results, `[[`, "InteractionDirection")
data$ConnectivityFocus <- sapply(results, `[[`, "ConnectivityFocus")
data$Notes <- sapply(results, `[[`, "Notes")

# ------------------------------------------------------------
# 14. DEFINE OUTPUT COLUMNS
# ------------------------------------------------------------

# Start with a core set of columns we definitely want in the output.
output_cols <- c(
  "Code",
  "Title",
  "Decision",
  "Reason",
  "IntegrationType",
  "SpatialScale",
  "MethodType",
  "InteractionDirection",
  "ConnectivityFocus",
  "Notes"
)

# If Year exists in the input dataset, include it in the output.
if ("Year" %in% names(data)) {
  output_cols <- append(output_cols, "Year", after = 2)
}

# ------------------------------------------------------------
# 15. WRITE THE OUTPUT FILE
# ------------------------------------------------------------

# Save the final screened and coded dataset as a tab-separated text file.
# Tab-separated output is often safer than CSV because abstracts may contain commas.
write.table(
  data[, output_cols],
  "output/screened_and_coded_output.txt",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

# Optional: also save a CSV version if you want
write.csv(
  data[, output_cols],
  "output/screened_and_coded_output.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# END OF SCRIPT
# ------------------------------------------------------------
message("Screening and coding completed successfully.")
