
# =============================================================================
# =============================================================================
# MASTER R PIPELINE -- ENHANCED LOG-HAR VERSION
# Signal or Redundancy? Technical Indicators and Volatility Forecasting
# =============================================================================
#
# Purpose
#   Run the ENTIRE R-side workflow in one execution:
#     1. audit and preprocess the NSE stock-level OHLCV panel;
#     2. select the development-only eligible stock sample;
#     3. create 5-, 10-, and 20-day forward Yang-Zhang variance targets;
#     4. construct primitive information, technical indicators, and liquidity;
#     5. create leakage-safe expanding validation/test registries;
#     6. estimate positivity-preserving real-time Log-HAR-X information-set forecasts;
#     7. estimate a GJR-GARCH(1,1)-Student-t conventional benchmark;
#     8. evaluate QLIKE/MSE, liquidity heterogeneity, and B-vs-C loss tests;
#     9. export one frozen Colab-ready panel for LightGBM/LSTM;
#    10. write all audit/results files and create ONE ZIP archive.
#
# IMPORTANT
#   LightGBM and LSTM are intentionally NOT fitted in R. They will be fitted
#   in Colab from the exact R-generated panel inside the ZIP archive.
#
# Locked design
#   - Sample eligibility decided using 2016-08-01 to 2023-12-29 only
#   - Main sample: development coverage >= 90%, missing volume <= 10%
#   - Unit: individual stock
#   - Primary target: forward 5-day Yang-Zhang daily variance estimate
#   - Robustness targets: forward 10- and 20-day Yang-Zhang daily variance
#   - Corporate actions reset rolling/technical features
#   - No price interpolation or OHLC repair
#   - "-" volume is missing; numeric 0 volume remains 0
#   - Test period: 2024-01-02 to 2026-07-31
#
# Required packages: data.table, TTR, rugarch, zip
# =============================================================================

required_pkgs <- c("data.table", "TTR", "rugarch", "zip")
missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs)) {
  stop(
    "Install required package(s) first: ",
    paste(missing_pkgs, collapse = ", "),
    "\nExample: install.packages(c(",
    paste(sprintf('"%s"', missing_pkgs), collapse = ", "), "))"
  )
}

library(data.table)
library(TTR)
library(rugarch)
library(zip)

# ----------------------------- CONFIG ----------------------------------------
# Use the same NSE data directory convention as the previous R forecasting code.
# You can override either path with environment variables without editing this file.
DATA_DIR <- Sys.getenv(
  "NSE_DATA_DIR",
  unset = "C:/..DATA From Office LapTop/PhD Research Paper/Data/Testing-Data Files/R-Data Import"
)

MASTER_FILENAME <- Sys.getenv(
  "NSE_MASTER_FILE",
  unset = "Final Master File_All variables 01082016-31072026.csv"
)

if (!dir.exists(DATA_DIR)) {
  stop(
    "DATA_DIR does not exist: ", DATA_DIR,
    "\nSet NSE_DATA_DIR or edit DATA_DIR."
  )
}

INPUT_FILE <- file.path(DATA_DIR, MASTER_FILENAME)

SAMPLE_START_DATE <- as.IDate("2016-08-01")
SAMPLE_END_DATE   <- as.IDate("2026-07-31")

PERIOD_TAG <- paste0(
  format(as.Date(SAMPLE_START_DATE), "%Y%m%d"),
  "_",
  format(as.Date(SAMPLE_END_DATE), "%Y%m%d")
)

OUT_DIR <- file.path(
  DATA_DIR,
  paste0("Signal_or_Redundancy_R_FULL_LOGHAR_FIXED_", PERIOD_TAG)
)

# Always start from a clean output directory so a rerun cannot mix stale and
# current results in the final ZIP archive.
if (dir.exists(OUT_DIR)) {
  unlink(OUT_DIR, recursive = TRUE, force = TRUE)
}
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

DEV_START  <- as.IDate("2016-08-01")
DEV_END    <- as.IDate("2023-12-29")
TEST_START <- as.IDate("2024-01-02")
TEST_END   <- as.IDate("2026-07-31")

MIN_DEV_COVERAGE <- 0.90
MAX_MISSING_VOL  <- 0.10
LIQ_WINDOW       <- 20L
MIN_LIQ_OBS      <- 16L       # at least 80% of a 20-observation rolling window
HORIZONS         <- c(5L, 10L, 20L)

PRIMARY_H        <- 5L

# ----------------------- FULL-RUN MODEL CONFIG -------------------------------
SEED <- 42L
set.seed(SEED)
RUN_STARTED_AT <- Sys.time()

# HAR-X: real-time refitting. Validation is annual (2020/21/22/23 folds);
# the locked test is refitted at the start of every calendar month.
HAR_TEST_REFIT_MONTHS <- 1L

# GJR-GARCH is a conventional benchmark only. Quarterly refitting materially
# reduces runtime while remaining genuinely recursive/real-time.
GJR_REFIT_MONTHS <- 3L
GJR_MIN_TRAIN_RETURNS <- 500L
GJR_DISTRIBUTION <- "std"       # standardized Student-t
GJR_RETURN_SCALE <- 100         # estimate on percentage log returns

# Forecast-evaluation constants.
VAR_SCALE <- 10000              # raw log-return variance -> percent-return^2
FORECAST_EPS <- 1e-12
TARGET_EPS   <- 1e-12

# Log-HAR retransformation:
#   log(target * VAR_SCALE + LOGHAR_EPS_SCALED) is modeled by full-rank OLS.
#   Forecasts are retransformed with Duan's training-only smearing estimator.
LOGHAR_EPS_SCALED <- 1e-8
LOGHAR_QR_TOL <- 1e-10
LOGHAR_MIN_STOCK_SMEAR_N <- 100L
LOGHAR_MAX_ABS_RESID_FOR_EXP <- 50

BOOTSTRAP_REPS <- 5000L
BOOTSTRAP_SEED <- 42L

# Primary RQ information sets.
HAR_A_VARS <- c(
  "yz_hist_5", "yz_hist_10", "yz_hist_20"
)

HAR_B_EXTRA <- c(
  "ret_cc", "abs_ret_cc", "ret_on", "log_hl_range",
  "log_volume", "log_turnover20", "zero_ret20",
  "log_amihud20", "log_mcap"
)

TREND_VARS <- c("sma_gap20", "nmacd_12_26", "adx14")
MOMENTUM_VARS <- c("rsi14", "roc10", "stoch_k14")
RANGE_TI_VARS <- c("natr14", "bbw20")

RUN_FAMILY_ABLATIONS <- TRUE

# ZIP configuration.
ZIP_BASENAME <- paste0("Signal_or_Redundancy_R_FULL_LOGHAR_FIXED_", PERIOD_TAG, ".zip")
ZIP_FILE <- file.path(DATA_DIR, ZIP_BASENAME)

# -------------------------- HELPER FUNCTIONS ---------------------------------
num_clean <- function(x) {
  suppressWarnings(as.numeric(gsub(",", "", trimws(as.character(x)), fixed = TRUE)))
}

safe_log_ratio <- function(a, b) {
  out <- rep(NA_real_, length(a))
  ok <- is.finite(a) & is.finite(b) & a > 0 & b > 0
  out[ok] <- log(a[ok] / b[ok])
  out
}

roll_mean_min <- function(x, n, min_obs = ceiling(0.8 * n)) {
  if (length(x) < n) return(rep(NA_real_, length(x)))
  z <- frollmean(x, n = n, align = "right", na.rm = TRUE)
  cnt <- frollsum(as.integer(!is.na(x)), n = n, align = "right")
  z[cnt < min_obs] <- NA_real_
  z[!is.finite(z)] <- NA_real_
  z
}

roll_sd_safe <- function(x, n) {
  if (length(x) < n) return(rep(NA_real_, length(x)))
  frollapply(x, n = n, FUN = stats::sd, align = "right")
}

ema_safe <- function(x, n) {
  if (length(x) < n) return(rep(NA_real_, length(x)))
  as.numeric(TTR::EMA(x, n = n))
}

rsi_safe <- function(x, n = 14L) {
  if (length(x) <= n) return(rep(NA_real_, length(x)))
  as.numeric(TTR::RSI(x, n = n))
}

atr_safe <- function(H, L, C, n = 14L) {
  N <- length(C)
  if (N <= n) return(rep(NA_real_, N))
  ans <- tryCatch(
    TTR::ATR(cbind(high = H, low = L, close = C), n = n),
    error = function(e) NULL
  )
  if (is.null(ans)) return(rep(NA_real_, N))
  as.numeric(ans[, "atr"])
}

adx_safe <- function(H, L, C, n = 14L) {
  N <- length(C)
  if (N < (2L * n)) return(rep(NA_real_, N))
  ans <- tryCatch(
    TTR::ADX(cbind(high = H, low = L, close = C), n = n),
    error = function(e) NULL
  )
  if (is.null(ans)) return(rep(NA_real_, N))
  as.numeric(ans[, "ADX"])
}

stoch_k_safe <- function(H, L, C, n = 14L) {
  N <- length(C)
  if (N < n) return(rep(NA_real_, N))
  ans <- tryCatch(
    TTR::stoch(
      cbind(high = H, low = L, close = C),
      nFastK = n, nFastD = 3L, nSlowD = 3L
    ),
    error = function(e) NULL
  )
  if (is.null(ans)) return(rep(NA_real_, N))
  as.numeric(ans[, "fastK"])
}

# Yang-Zhang from already aligned matrices (each row is one forecast origin).
# Output is an estimate of average DAILY variance over the n-day window.
yz_from_mats <- function(o_mat, c_mat, rs_mat) {
  n <- ncol(o_mat)
  stopifnot(n >= 2L, ncol(c_mat) == n, ncol(rs_mat) == n)
  
  ok <- rowSums(!is.finite(o_mat)) == 0L &
    rowSums(!is.finite(c_mat)) == 0L &
    rowSums(!is.finite(rs_mat)) == 0L
  
  out <- rep(NA_real_, nrow(o_mat))
  if (!any(ok)) return(out)
  
  om <- rowMeans(o_mat[ok, , drop = FALSE])
  cm <- rowMeans(c_mat[ok, , drop = FALSE])
  
  vo <- (rowSums(o_mat[ok, , drop = FALSE]^2) - n * om^2) / (n - 1)
  vc <- (rowSums(c_mat[ok, , drop = FALSE]^2) - n * cm^2) / (n - 1)
  vr <- rowMeans(rs_mat[ok, , drop = FALSE])
  
  # Protect against tiny negative values from floating-point cancellation.
  vo <- pmax(vo, 0)
  vc <- pmax(vc, 0)
  vr <- pmax(vr, 0)
  
  k <- 0.34 / (1.34 + (n + 1) / (n - 1))
  out[ok] <- vo + k * vc + (1 - k) * vr
  out
}

yz_trailing <- function(o, c, rs, n) {
  if (length(o) < n) return(rep(NA_real_, length(o)))
  o_mat  <- do.call(cbind, lapply(0:(n - 1L), function(k) shift(o,  k, type = "lag")))
  c_mat  <- do.call(cbind, lapply(0:(n - 1L), function(k) shift(c,  k, type = "lag")))
  rs_mat <- do.call(cbind, lapply(0:(n - 1L), function(k) shift(rs, k, type = "lag")))
  yz_from_mats(o_mat, c_mat, rs_mat)
}

yz_forward <- function(o, c, rs, n) {
  if (length(o) <= n) return(rep(NA_real_, length(o)))
  o_mat  <- do.call(cbind, lapply(1:n, function(k) shift(o,  k, type = "lead")))
  c_mat  <- do.call(cbind, lapply(1:n, function(k) shift(c,  k, type = "lead")))
  rs_mat <- do.call(cbind, lapply(1:n, function(k) shift(rs, k, type = "lead")))
  yz_from_mats(o_mat, c_mat, rs_mat)
}

future_any <- function(x, n) {
  if (length(x) <= n) return(rep(NA, length(x)))
  m <- do.call(cbind, lapply(1:n, function(k) shift(x, k, type = "lead")))
  out <- rowSums(m == TRUE, na.rm = TRUE) > 0
  incomplete <- rowSums(is.na(m)) > 0
  out[incomplete] <- NA
  out
}

# ------------------------------ LOAD -----------------------------------------
cat("\n================ PATH CONFIGURATION ================\n")
cat("DATA_DIR   :", normalizePath(DATA_DIR, winslash = "/", mustWork = TRUE), "\n")
cat("INPUT_FILE :", normalizePath(INPUT_FILE, winslash = "/", mustWork = FALSE), "\n")
cat("OUTPUT_DIR :", normalizePath(OUT_DIR, winslash = "/", mustWork = FALSE), "\n")
cat("====================================================\n\n")

if (!file.exists(INPUT_FILE)) {
  stop(
    "Master CSV not found: ", INPUT_FILE,
    "\nCheck MASTER_FILENAME or NSE_MASTER_FILE."
  )
}

DT <- fread(
  INPUT_FILE,
  na.strings = c("", "NA", "N/A", "-", "NULL"),
  strip.white = TRUE,
  showProgress = TRUE
)

expected <- c(
  "Date", "Open", "High", "Low", "Close", "Volume", "Category", "Stock",
  "Bonus issue", "Issued Shares", "Market Capitalization (KES)"
)
missing_cols <- setdiff(expected, names(DT))
if (length(missing_cols)) {
  stop("Missing expected column(s): ", paste(missing_cols, collapse = ", "))
}

setnames(
  DT,
  old = c("Bonus issue", "Issued Shares", "Market Capitalization (KES)"),
  new = c("BonusIssue", "IssuedShares", "MarketCap")
)

# Parse DD/MM/YYYY explicitly.
DT[, Date := as.IDate(Date, format = "%d/%m/%Y")]

for (v in c("Open", "High", "Low", "Close", "Volume", "IssuedShares", "MarketCap")) {
  set(DT, j = v, value = num_clean(DT[[v]]))
}

DT[, Stock := trimws(as.character(Stock))]
DT[, Category := trimws(as.character(Category))]
DT[, BonusIssue := trimws(as.character(BonusIssue))]
DT[BonusIssue %chin% c("", "NA", "NaN", "NULL"), BonusIssue := NA_character_]

if (anyNA(DT$Date)) stop("Unparsed dates remain.")
if (anyDuplicated(DT, by = c("Stock", "Date"))) {
  dup <- DT[duplicated(DT, by = c("Stock", "Date")) |
              duplicated(DT, by = c("Stock", "Date"), fromLast = TRUE)]
  fwrite(dup, file.path(OUT_DIR, "ERROR_duplicate_stock_dates.csv"))
  stop("Duplicate Stock-Date rows detected. See ERROR_duplicate_stock_dates.csv")
}

setorder(DT, Stock, Date)

# ----------------------- BASIC QUALITY FLAGS ---------------------------------
DT[, observed_row := 1L]
DT[, ohlc_valid :=
     is.finite(Open) & is.finite(High) & is.finite(Low) & is.finite(Close) &
     Open > 0 & High > 0 & Low > 0 & Close > 0 &
     High >= pmax(Open, Close) &
     Low  <= pmin(Open, Close)]

# ------------------- ELIGIBILITY: DEVELOPMENT ONLY ---------------------------
market_calendar <- sort(unique(DT$Date))
dev_calendar <- market_calendar[market_calendar >= DEV_START & market_calendar <= DEV_END]
n_dev_dates <- length(dev_calendar)

dev_stats <- DT[
  Date >= DEV_START & Date <= DEV_END,
  .(
    n_dev_dates_observed = uniqueN(Date),
    n_dev_rows = .N,
    missing_volume_rate = mean(is.na(Volume))
  ),
  by = .(Stock)
]
dev_stats[, dev_coverage := n_dev_dates_observed / n_dev_dates]
dev_stats[, eligible :=
            dev_coverage >= MIN_DEV_COVERAGE &
            missing_volume_rate <= MAX_MISSING_VOL]

# Attach stable sector label for audit.
sector_map <- DT[, .(Category = names(sort(table(Category), decreasing = TRUE))[1L]), by = Stock]
dev_stats <- merge(dev_stats, sector_map, by = "Stock", all.x = TRUE)

eligible_stocks <- dev_stats[eligible == TRUE, Stock]
if (!length(eligible_stocks)) stop("No stocks satisfy the locked eligibility rule.")

fwrite(
  dev_stats[order(-eligible, -dev_coverage, missing_volume_rate, Stock)],
  file.path(OUT_DIR, "eligible_stock_audit.csv")
)

# ------------------ OBSERVED ELIGIBLE STOCK DATA -----------------------------
OBS <- DT[Stock %chin% eligible_stocks]
setorder(OBS, Stock, Date)

# Corporate-action event: any explicit BonusIssue OR change in issued shares.
# The threshold only avoids floating-point equality noise.
OBS[, prev_shares := shift(IssuedShares), by = Stock]
OBS[, share_change_rel :=
      fifelse(
        is.finite(prev_shares) & prev_shares > 0 & is.finite(IssuedShares),
        abs(IssuedShares / prev_shares - 1),
        NA_real_
      )]
OBS[, corp_event :=
      (!is.na(BonusIssue)) |
      (!is.na(share_change_rel) & share_change_rel > 1e-8)]

# Reset all path-dependent features at each event.
OBS[, segment_id := cumsum(as.integer(corp_event)), by = Stock]

corp_actions <- OBS[
  corp_event == TRUE,
  .(Stock, Category, Date, prev_shares, IssuedShares, share_change_rel, BonusIssue)
]
fwrite(corp_actions, file.path(OUT_DIR, "corporate_action_audit.csv"))

# ------------------- OBSERVED-SESSION PRIMITIVE FEATURES ---------------------
OBS[, prev_close_segment := shift(Close), by = .(Stock, segment_id)]

OBS[, ret_cc := safe_log_ratio(Close, prev_close_segment), by = .(Stock, segment_id)]
OBS[, abs_ret_cc := abs(ret_cc)]
OBS[, ret_on := safe_log_ratio(Open,  prev_close_segment), by = .(Stock, segment_id)]
OBS[, ret_oc := safe_log_ratio(Close, Open)]
OBS[, log_hl_range := safe_log_ratio(High, Low)]

# Yang-Zhang components on observed sessions, reset at corporate actions.
OBS[, yz_o := ret_on]
OBS[, yz_c := ret_oc]
OBS[, yz_u := safe_log_ratio(High, Open)]
OBS[, yz_d := safe_log_ratio(Low,  Open)]
OBS[, yz_rs := yz_u * (yz_u - yz_c) + yz_d * (yz_d - yz_c)]

OBS[ohlc_valid == FALSE, c("ret_oc", "log_hl_range", "yz_c", "yz_u", "yz_d", "yz_rs") :=
      list(NA_real_, NA_real_, NA_real_, NA_real_, NA_real_, NA_real_)]

# If previous close is unavailable within the post-event segment, overnight/CC are unavailable.
OBS[!is.finite(prev_close_segment) | prev_close_segment <= 0,
    c("ret_cc", "ret_on", "yz_o") := list(NA_real_, NA_real_, NA_real_)]

# Historical multi-scale Yang-Zhang variance (observed stock sessions).
for (h in HORIZONS) {
  nm <- paste0("yz_hist_", h)
  OBS[, (nm) := yz_trailing(yz_o, yz_c, yz_rs, h),
      by = .(Stock, segment_id)]
}

# Trading activity / liquidity primitives.
OBS[, log_volume := fifelse(is.na(Volume), NA_real_, log1p(Volume))]
OBS[, turnover :=
      fifelse(
        is.finite(Volume) & is.finite(IssuedShares) & IssuedShares > 0,
        Volume / IssuedShares,
        NA_real_
      )]
OBS[, amihud_day :=
      fifelse(
        is.finite(ret_cc) & is.finite(Close) & Close > 0 &
          is.finite(Volume) & Volume > 0,
        abs(ret_cc) / (Close * Volume),
        NA_real_
      )]
OBS[, log_mcap := fifelse(is.finite(MarketCap) & MarketCap > 0, log(MarketCap), NA_real_)]

OBS[, turnover20 := roll_mean_min(turnover, LIQ_WINDOW, MIN_LIQ_OBS),
    by = .(Stock, segment_id)]
OBS[, zero_ret20 := roll_mean_min(as.numeric(ret_cc == 0), LIQ_WINDOW, MIN_LIQ_OBS),
    by = .(Stock, segment_id)]
OBS[, amihud20 := roll_mean_min(amihud_day, LIQ_WINDOW, MIN_LIQ_OBS),
    by = .(Stock, segment_id)]
OBS[, log_turnover20 := fifelse(is.finite(turnover20) & turnover20 >= 0,
                                log1p(turnover20), NA_real_)]
# Scale Amihud before logging only for numerical interpretability.
OBS[, log_amihud20 := fifelse(
  is.finite(amihud20) & amihud20 >= 0,
  log1p(amihud20 * 1e6),
  NA_real_
)]

# ----------------------- TECHNICAL INDICATORS --------------------------------
# All are calculated WITHIN post-corporate-action segments so pre-event prices
# never enter post-event moving averages / EMA / momentum / range indicators.

# 1) Price-to-SMA(20) gap
OBS[, sma20 := {
  if (.N < 20L) rep(NA_real_, .N) else frollmean(Close, 20L, align = "right")
}, by = .(Stock, segment_id)]
OBS[, sma_gap20 := Close / sma20 - 1]

# 2) Normalized MACD(12,26)
OBS[, ema12 := ema_safe(Close, 12L), by = .(Stock, segment_id)]
OBS[, ema26 := ema_safe(Close, 26L), by = .(Stock, segment_id)]
OBS[, nmacd_12_26 := (ema12 - ema26) / Close]

# 3) ADX(14)
OBS[, adx14 := adx_safe(High, Low, Close, 14L), by = .(Stock, segment_id)]

# 4) RSI(14)
OBS[, rsi14 := rsi_safe(Close, 14L), by = .(Stock, segment_id)]

# 5) ROC(10), simple percentage rate of change
OBS[, roc10 := Close / shift(Close, 10L) - 1, by = .(Stock, segment_id)]

# 6) Stochastic %K(14)
OBS[, stoch_k14 := stoch_k_safe(High, Low, Close, 14L),
    by = .(Stock, segment_id)]

# 7) Normalized ATR(14)
OBS[, atr14 := atr_safe(High, Low, Close, 14L), by = .(Stock, segment_id)]
OBS[, natr14 := atr14 / Close]

# 8) Bollinger Band Width(20), based on Close; Upper-Lower = 4 * rolling sd
OBS[, sd20 := roll_sd_safe(Close, 20L), by = .(Stock, segment_id)]
OBS[, bbw20 := (4 * sd20) / sma20]

TECH_VARS <- c(
  "sma_gap20", "nmacd_12_26", "adx14", "rsi14",
  "roc10", "stoch_k14", "natr14", "bbw20"
)

PRIMITIVE_VARS <- c(
  "yz_hist_5", "yz_hist_10", "yz_hist_20",
  "ret_cc", "abs_ret_cc", "ret_on", "ret_oc", "log_hl_range",
  "log_volume", "turnover", "turnover20", "log_turnover20",
  "zero_ret20", "amihud20", "log_amihud20", "log_mcap"
)

# Keep only model-facing observed-session columns before calendar expansion.
model_cols <- unique(c(
  "Stock", "Date", "Category", "observed_row", "ohlc_valid",
  "segment_id", "corp_event",
  "Open", "High", "Low", "Close", "Volume", "IssuedShares", "MarketCap",
  PRIMITIVE_VARS, TECH_VARS
))
OBS_MODEL <- OBS[, ..model_cols]

# -------------------------- FULL MARKET CALENDAR ------------------------------
# Critical: forward targets use the next h NSE market dates, not merely the next
# h observed records for an individual stock. This prevents illiquid stocks from
# silently converting 5 market days into a much longer calendar interval.

GRID <- CJ(Stock = eligible_stocks, Date = market_calendar, unique = TRUE)
GRID <- merge(GRID, OBS_MODEL, by = c("Stock", "Date"), all.x = TRUE, sort = FALSE)
setorder(GRID, Stock, Date)

GRID[is.na(observed_row), observed_row := 0L]
GRID[is.na(corp_event), corp_event := FALSE]

# Fill sector label ONLY as a static identifier; never fill prices/features.
stock_sector <- sector_map[Stock %chin% eligible_stocks]
GRID <- merge(GRID, stock_sector, by = "Stock", all.x = TRUE, suffixes = c("", "_static"))
GRID[is.na(Category), Category := Category_static]
GRID[, Category_static := NULL]
setorder(GRID, Stock, Date)

# Calendar-based observation coverage: useful liquidity/staleness robustness.
GRID[, obs_share20 := {
  if (.N < LIQ_WINDOW) rep(NA_real_, .N)
  else frollmean(observed_row, LIQ_WINDOW, align = "right")
}, by = Stock]

# Ex-ante cross-sectional liquidity state at each market date.
# This uses only contemporaneously observable rolling turnover and never targets.
GRID[, liq_state := {
  z <- log_turnover20
  out <- rep(NA_character_, .N)
  ok <- is.finite(z)
  if (sum(ok) >= 10L) {
    r <- frank(z[ok], ties.method = "average")
    n <- length(r)
    out[ok] <- fifelse(
      r <= n / 3, "Low",
      fifelse(r > 2 * n / 3, "High", "Medium")
    )
  }
  out
}, by = Date]

# Calendar-day OHLC validity/components for forward Yang-Zhang targets.
GRID[, cal_ohlc_valid :=
       observed_row == 1L &
       is.finite(Open) & is.finite(High) & is.finite(Low) & is.finite(Close) &
       Open > 0 & High > 0 & Low > 0 & Close > 0 &
       High >= pmax(Open, Close) &
       Low <= pmin(Open, Close)]

GRID[, prev_close_calendar := shift(Close), by = Stock]
GRID[, cal_o := safe_log_ratio(Open, prev_close_calendar), by = Stock]
GRID[, cal_c := safe_log_ratio(Close, Open)]
GRID[, cal_u := safe_log_ratio(High, Open)]
GRID[, cal_d := safe_log_ratio(Low, Open)]
GRID[, cal_rs := cal_u * (cal_u - cal_c) + cal_d * (cal_d - cal_c)]

GRID[cal_ohlc_valid == FALSE,
     c("cal_c", "cal_u", "cal_d", "cal_rs") :=
       list(NA_real_, NA_real_, NA_real_, NA_real_)]

GRID[!is.finite(prev_close_calendar) | prev_close_calendar <= 0,
     cal_o := NA_real_]

# ----------------------- FORWARD TARGETS 5/10/20 -----------------------------
for (h in HORIZONS) {
  target_nm <- paste0("target_yz_", h)
  end_nm    <- paste0("target_end_", h)
  event_nm  <- paste0("future_corp_event_", h)
  
  GRID[, (target_nm) := yz_forward(cal_o, cal_c, cal_rs, h), by = Stock]
  GRID[, (end_nm) := shift(Date, h, type = "lead"), by = Stock]
  GRID[, (event_nm) := future_any(corp_event, h), by = Stock]
  
  # A target crossing a capital-structure event is not used.
  GRID[get(event_nm) == TRUE, (target_nm) := NA_real_]
  
  # Forecast origin itself must be an actual, valid stock observation.
  GRID[observed_row != 1L | cal_ohlc_valid != TRUE, (target_nm) := NA_real_]
}

# --------------------------- SPLIT LABELS ------------------------------------
GRID[, calendar_year := as.integer(format(Date, "%Y"))]
GRID[, split_label := fifelse(
  Date >= DEV_START & Date <= as.IDate("2019-12-31"), "initial_train",
  fifelse(
    Date >= as.IDate("2020-01-01") & Date <= as.IDate("2020-12-31"), "validation_2020",
    fifelse(
      Date >= as.IDate("2021-01-01") & Date <= as.IDate("2021-12-31"), "validation_2021",
      fifelse(
        Date >= as.IDate("2022-01-01") & Date <= as.IDate("2022-12-31"), "validation_2022",
        fifelse(
          Date >= as.IDate("2023-01-01") & Date <= DEV_END, "validation_2023",
          fifelse(Date >= TEST_START & Date <= TEST_END, "locked_test", "outside")
        )
      )
    )
  )
)]

# Readiness flags. We do NOT impute here; any imputation/scaling is fold-specific.
GRID[, primitive_price_ready :=
       complete.cases(.SD),
     .SDcols = c(
       "yz_hist_5", "yz_hist_10", "yz_hist_20",
       "ret_cc", "ret_on", "ret_oc", "log_hl_range", "log_mcap"
     )]

GRID[, technical_ready := complete.cases(.SD), .SDcols = TECH_VARS]

# Primary target indicator.
GRID[, primary_target_available := is.finite(target_yz_5)]

# ----------------------- EXPANDING-FOLD REGISTRY ------------------------------
fold_registry <- rbindlist(list(
  data.table(
    fold = "fold_2020",
    train_start = DEV_START,
    train_end = as.IDate("2019-12-31"),
    validation_start = as.IDate("2020-01-01"),
    validation_end = as.IDate("2020-12-31")
  ),
  data.table(
    fold = "fold_2021",
    train_start = DEV_START,
    train_end = as.IDate("2020-12-31"),
    validation_start = as.IDate("2021-01-01"),
    validation_end = as.IDate("2021-12-31")
  ),
  data.table(
    fold = "fold_2022",
    train_start = DEV_START,
    train_end = as.IDate("2021-12-31"),
    validation_start = as.IDate("2022-01-01"),
    validation_end = as.IDate("2022-12-31")
  ),
  data.table(
    fold = "fold_2023",
    train_start = DEV_START,
    train_end = as.IDate("2022-12-31"),
    validation_start = as.IDate("2023-01-01"),
    validation_end = DEV_END
  )
))
fwrite(fold_registry, file.path(OUT_DIR, "fold_registry.csv"))

# Example purge rule for downstream modeling:
#   training origin Date <= train_end AND target_end_h <= train_end
#   validation origin Date within validation year AND target_end_h <= validation_end
# This prevents h-day outcomes from crossing fold boundaries.

# ----------------------------- AUDIT OUTPUTS ----------------------------------
audit_overall <- data.table(
  metric = c(
    "raw_rows",
    "raw_stocks",
    "raw_market_dates",
    "development_market_dates",
    "eligible_stocks",
    "eligible_sectors",
    "invalid_ohlc_raw",
    "missing_volume_raw",
    "numeric_zero_volume_raw",
    "corporate_action_events_eligible"
  ),
  value = c(
    nrow(DT),
    uniqueN(DT$Stock),
    uniqueN(DT$Date),
    n_dev_dates,
    length(eligible_stocks),
    uniqueN(GRID[observed_row == 1L, Category]),
    DT[ohlc_valid == FALSE, .N],
    DT[is.na(Volume), .N],
    DT[!is.na(Volume) & Volume == 0, .N],
    nrow(corp_actions)
  )
)

target_audit <- rbindlist(lapply(HORIZONS, function(h) {
  nm <- paste0("target_yz_", h)
  end_nm <- paste0("target_end_", h)
  data.table(
    horizon = h,
    development_origins_with_target =
      GRID[
        Date >= DEV_START & Date <= DEV_END &
          observed_row == 1L & is.finite(get(nm)),
        .N
      ],
    locked_test_origins_observed =
      GRID[Date >= TEST_START & Date <= TEST_END & observed_row == 1L, .N],
    locked_test_origins_with_target =
      GRID[
        Date >= TEST_START & Date <= TEST_END &
          observed_row == 1L & is.finite(get(nm)),
        .N
      ],
    locked_test_target_rate =
      GRID[
        Date >= TEST_START & Date <= TEST_END & observed_row == 1L,
        mean(is.finite(get(nm)))
      ]
  )
}))

fwrite(audit_overall, file.path(OUT_DIR, "preprocessing_audit_overall.csv"))
fwrite(target_audit, file.path(OUT_DIR, "preprocessing_audit_targets.csv"))

# Main analysis panel.
# Keep missing values: downstream imputation/scaling must be fitted within each fold only.
fwrite(
  GRID,
  file.path(OUT_DIR, "analysis_panel.csv.gz"),
  compress = "gzip",
  na = "NA"
)

# --------------------- PREPROCESSING CHECKPOINT -------------------------------
cat("\n================ PREPROCESSING COMPLETE ================\n")
print(audit_overall)
cat("\nEligible stock count:", length(eligible_stocks), "\n")
cat("\nTarget availability:\n")
print(target_audit)
cat("========================================================\n")


# =============================================================================
# PART II. R FORECASTING MODELS
# =============================================================================

# ---------------------------- MODEL MANIFEST ----------------------------------
HAR_B_VARS <- unique(c(HAR_A_VARS, HAR_B_EXTRA))
HAR_C_VARS <- unique(c(HAR_B_VARS, TECH_VARS))

HAR_SPECS <- list(
  A_volatility_memory = HAR_A_VARS,
  B_primitive = HAR_B_VARS,
  C_primitive_plus_TI = HAR_C_VARS
)

if (RUN_FAMILY_ABLATIONS) {
  HAR_SPECS <- c(
    HAR_SPECS,
    list(
      B_plus_trend = unique(c(HAR_B_VARS, TREND_VARS)),
      B_plus_momentum = unique(c(HAR_B_VARS, MOMENTUM_VARS)),
      B_plus_range_TI = unique(c(HAR_B_VARS, RANGE_TI_VARS))
    )
  )
}

feature_manifest <- rbindlist(list(
  data.table(variable = HAR_A_VARS, group = "volatility_memory"),
  data.table(variable = setdiff(HAR_B_VARS, HAR_A_VARS), group = "primitive_extra"),
  data.table(variable = TREND_VARS, group = "technical_trend"),
  data.table(variable = MOMENTUM_VARS, group = "technical_momentum"),
  data.table(variable = RANGE_TI_VARS, group = "technical_range"),
  data.table(variable = c("log_turnover20", "zero_ret20", "log_amihud20", "obs_share20"),
             group = "liquidity")
), fill = TRUE)
feature_manifest <- unique(feature_manifest)
fwrite(feature_manifest, file.path(OUT_DIR, "feature_manifest.csv"))

config_manifest <- data.table(
  parameter = c(
    "sample_start", "sample_end", "development_end", "locked_test_start",
    "locked_test_end", "eligible_stock_rule_min_coverage",
    "eligible_stock_rule_max_missing_volume", "horizons",
    "primary_horizon", "primary_loss", "secondary_loss", "har_link", "har_retransformation",
    "har_test_refit_months", "gjr_refit_months", "gjr_distribution",
    "bootstrap_reps", "seed"
  ),
  value = c(
    as.character(SAMPLE_START_DATE), as.character(SAMPLE_END_DATE),
    as.character(DEV_END), as.character(TEST_START), as.character(TEST_END),
    as.character(MIN_DEV_COVERAGE), as.character(MAX_MISSING_VOL),
    paste(HORIZONS, collapse = ","), as.character(PRIMARY_H),
    "QLIKE", "MSE", "log", "Duan training-only smearing", as.character(HAR_TEST_REFIT_MONTHS),
    as.character(GJR_REFIT_MONTHS), GJR_DISTRIBUTION,
    as.character(BOOTSTRAP_REPS), as.character(SEED)
  )
)
fwrite(config_manifest, file.path(OUT_DIR, "run_configuration.csv"))

# --------------------------- LOSS FUNCTIONS ----------------------------------
qlike_loss <- function(y, f, eps_y = TARGET_EPS, eps_f = FORECAST_EPS) {
  yy <- pmax(as.numeric(y), eps_y)
  ff <- pmax(as.numeric(f), eps_f)
  ratio <- yy / ff
  ratio - log(ratio) - 1
}

mse_loss <- function(y, f) {
  (as.numeric(y) - as.numeric(f))^2
}

safe_mean <- function(x) {
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  mean(x)
}

# Newey-West standard error of the sample mean using Bartlett weights.
nw_mean_test <- function(x, lag = 0L) {
  x <- as.numeric(x)
  x <- x[is.finite(x)]
  n <- length(x)
  if (n < 5L) {
    return(data.table(n = n, mean = NA_real_, se_hac = NA_real_,
                      t_hac = NA_real_, p_hac = NA_real_, lag = lag))
  }
  lag <- max(0L, min(as.integer(lag), n - 1L))
  xc <- x - mean(x)
  gamma0 <- sum(xc^2) / n
  lrv <- gamma0
  if (lag > 0L) {
    for (j in seq_len(lag)) {
      gj <- sum(xc[(j + 1L):n] * xc[1L:(n - j)]) / n
      wj <- 1 - j / (lag + 1)
      lrv <- lrv + 2 * wj * gj
    }
  }
  se <- sqrt(max(lrv, 0) / n)
  tt <- if (is.finite(se) && se > 0) mean(x) / se else NA_real_
  pp <- if (is.finite(tt)) 2 * stats::pnorm(-abs(tt)) else NA_real_
  data.table(n = n, mean = mean(x), se_hac = se, t_hac = tt, p_hac = pp, lag = lag)
}

# Circular moving-block bootstrap CI/p-value for a time-series mean.
# The confidence interval resamples the observed series. The p-value is based
# on a centered series so the bootstrap distribution represents H0: mean = 0.
mbb_mean_test <- function(x, block_length, B = BOOTSTRAP_REPS, seed = BOOTSTRAP_SEED) {
  x <- as.numeric(x)
  x <- x[is.finite(x)]
  n <- length(x)
  if (n < 10L) {
    return(data.table(
      boot_reps = B, block_length = block_length,
      ci_low = NA_real_, ci_high = NA_real_, p_boot = NA_real_
    ))
  }
  
  L <- max(2L, min(as.integer(block_length), n))
  obs_mean <- mean(x)
  x0 <- x - obs_mean
  
  set.seed(seed)
  means_uncentered <- numeric(B)
  means_null <- numeric(B)
  
  for (b in seq_len(B)) {
    draw_idx <- integer(0)
    while (length(draw_idx) < n) {
      s <- sample.int(n, 1L)
      idx <- ((s - 1L + 0:(L - 1L)) %% n) + 1L
      draw_idx <- c(draw_idx, idx)
    }
    draw_idx <- draw_idx[seq_len(n)]
    means_uncentered[b] <- mean(x[draw_idx])
    means_null[b] <- mean(x0[draw_idx])
  }
  
  ci <- as.numeric(stats::quantile(
    means_uncentered,
    probs = c(0.025, 0.975),
    na.rm = TRUE
  ))
  
  pboot <- mean(abs(means_null) >= abs(obs_mean))
  pboot <- min(1, max(0, pboot))
  
  data.table(
    boot_reps = B,
    block_length = L,
    ci_low = ci[1],
    ci_high = ci[2],
    p_boot = pboot
  )
}

# -------------------------- HAR-X PREPARATION --------------------------------
# HAR regression is estimated in percentage-return variance units to improve
# numerical conditioning. Forecasts are transformed back to raw variance.
har_prepare <- function(dt, target_col, features) {
  z <- copy(dt)
  
  # Positive response on the log-variance scale.
  z[, log_har_y := log(
    pmax(get(target_col) * VAR_SCALE, 0) + LOGHAR_EPS_SCALED
  )]
  
  # Heterogeneous volatility-memory terms enter in logs as well.
  # Primitive returns/ranges retain their natural signed/level interpretation.
  for (v in features) {
    if (!v %in% names(z)) next
    nv <- paste0("x__", v)
    
    if (grepl("^yz_hist_", v)) {
      z[, (nv) := log(
        pmax(get(v) * VAR_SCALE, 0) + LOGHAR_EPS_SCALED
      )]
    } else if (v %chin% c(
      "ret_cc", "abs_ret_cc", "ret_on", "ret_oc", "log_hl_range"
    )) {
      z[, (nv) := get(v) * 100]
    } else {
      z[, (nv) := get(v)]
    }
  }
  
  xvars <- paste0("x__", features)
  list(data = z, xvars = xvars)
}

# Full-rank Log-HAR-X:
#   - constructs the model matrix explicitly;
#   - retains only QR-independent columns;
#   - therefore avoids aliased-coefficient prediction warnings;
#   - uses training-only Duan smearing for positive retransformation;
#   - uses stock-specific smearing when sufficiently supported, otherwise the
#     global training smearing factor.
loghar_fit_predict <- function(train_dt, pred_dt, target_col, features,
                               horizon, spec_name, refit_label, stage) {
  prep_train <- har_prepare(train_dt, target_col, features)
  prep_pred  <- har_prepare(pred_dt,  target_col, features)
  
  tr <- prep_train$data
  pr <- prep_pred$data
  xvars <- prep_train$xvars
  
  required_tr <- c("log_har_y", "Stock", xvars)
  required_pr <- c("Stock", xvars)
  
  # Explicit column selection avoids data.table's ..name ambiguity and preserves
  # an auditable row mask for prediction.
  train_ok <- complete.cases(tr[, required_tr, with = FALSE])
  tr <- tr[train_ok]
  
  pr[, .row_id_internal := .I]
  pred_complete <- complete.cases(pr[, required_pr, with = FALSE])
  pr[, pred_ok := pred_complete]
  
  empty_prediction <- function(status_text, dropped_cols = NA_character_) {
    pred_out <- pr[, .(
      Stock, Date, Category, liq_state,
      target = get(target_col),
      forecast = NA_real_
    )]
    pred_out[, `:=`(
      horizon = horizon,
      model = "LOG_HAR_X",
      specification = spec_name,
      stage = stage,
      refit_label = refit_label
    )]
    
    fit_log <- data.table(
      model = "LOG_HAR_X",
      horizon = horizon,
      specification = spec_name,
      stage = stage,
      refit_label = refit_label,
      n_train = nrow(tr),
      n_stocks = uniqueN(tr$Stock),
      status = status_text,
      design_columns = NA_integer_,
      design_rank = NA_integer_,
      dropped_collinear_columns_n = NA_integer_,
      dropped_collinear_columns = dropped_cols,
      r_squared_log = NA_real_,
      adj_r_squared_log = NA_real_,
      sigma_log = NA_real_,
      global_smear = NA_real_,
      prediction_rows_total = nrow(pr),
      prediction_rows_complete = sum(pr$pred_ok, na.rm = TRUE),
      prediction_rows_missing_features = sum(!pr$pred_ok, na.rm = TRUE),
      min_forecast = NA_real_,
      nonpositive_forecasts = NA_integer_
    )
    list(pred = pred_out, fit_log = fit_log, coef = NULL)
  }
  
  if (nrow(tr) < 1000L || uniqueN(tr$Stock) < 10L) {
    return(empty_prediction("insufficient_training_data"))
  }
  
  stock_levels <- sort(unique(as.character(tr$Stock)))
  tr[, StockF := factor(Stock, levels = stock_levels)]
  pr[, StockF := factor(Stock, levels = stock_levels)]
  
  # Formula only generates a model matrix; fitting uses lm.fit() below.
  rhs <- paste(c(xvars, "StockF"), collapse = " + ")
  fm <- stats::as.formula(paste("~", rhs))
  
  Xtr <- tryCatch(
    stats::model.matrix(fm, data = tr),
    error = function(e) e
  )
  if (inherits(Xtr, "error")) {
    return(empty_prediction(
      paste0("model_matrix_error: ", conditionMessage(Xtr))
    ))
  }
  
  # Keep a linearly independent subset. This removes exact/near-exact aliases
  # before estimation rather than allowing predict.lm() to warn later.
  qx <- qr(Xtr, tol = LOGHAR_QR_TOL)
  if (qx$rank < 1L) return(empty_prediction("zero_design_rank"))
  
  keep_idx <- sort(qx$pivot[seq_len(qx$rank)])
  drop_idx <- setdiff(seq_len(ncol(Xtr)), keep_idx)
  
  Xtr_keep <- Xtr[, keep_idx, drop = FALSE]
  kept_names <- colnames(Xtr_keep)
  dropped_names <- if (length(drop_idx)) {
    colnames(Xtr)[drop_idx]
  } else {
    character(0)
  }
  
  fit <- tryCatch(
    stats::lm.fit(x = Xtr_keep, y = tr$log_har_y),
    error = function(e) e
  )
  if (inherits(fit, "error")) {
    return(empty_prediction(
      paste0("fit_error: ", conditionMessage(fit)),
      paste(dropped_names, collapse = "|")
    ))
  }
  
  beta <- as.numeric(fit$coefficients)
  names(beta) <- kept_names
  
  if (any(!is.finite(beta))) {
    return(empty_prediction(
      "nonfinite_coefficients",
      paste(dropped_names, collapse = "|")
    ))
  }
  
  # Training residuals and retransformation factors.
  fitted_log <- as.numeric(Xtr_keep %*% beta)
  resid_log <- tr$log_har_y - fitted_log
  
  # Guard exp() numerically without altering ordinary residuals.
  exp_resid <- exp(pmin(
    pmax(resid_log, -LOGHAR_MAX_ABS_RESID_FOR_EXP),
    LOGHAR_MAX_ABS_RESID_FOR_EXP
  ))
  
  global_smear <- mean(exp_resid[is.finite(exp_resid)])
  if (!is.finite(global_smear) || global_smear <= 0) global_smear <- 1
  
  smear_dt <- data.table(
    Stock = as.character(tr$Stock),
    exp_resid = exp_resid
  )[
    is.finite(exp_resid),
    .(
      smear_stock = mean(exp_resid),
      smear_n = .N
    ),
    by = Stock
  ]
  
  smear_dt[
    smear_n < LOGHAR_MIN_STOCK_SMEAR_N |
      !is.finite(smear_stock) |
      smear_stock <= 0,
    smear_stock := global_smear
  ]
  
  # Build the prediction matrix ONLY on rows with complete predictors.
  # model.matrix() otherwise applies na.omit internally and shortens the matrix,
  # which breaks row alignment with the original prediction table.
  pr[, pred_ok := pred_ok & !is.na(StockF)]
  valid_idx <- which(pr$pred_ok)
  
  pred_log <- rep(NA_real_, nrow(pr))
  
  if (length(valid_idx)) {
    pr_valid <- pr[valid_idx]
    
    Xpr <- tryCatch(
      stats::model.matrix(fm, data = pr_valid),
      error = function(e) e
    )
    
    if (inherits(Xpr, "error")) {
      return(empty_prediction(
        paste0("prediction_matrix_error: ", conditionMessage(Xpr)),
        paste(dropped_names, collapse = "|")
      ))
    }
    
    # Hard row-alignment audit: after pre-filtering complete rows,
    # model.matrix() must return exactly one row per valid prediction row.
    if (nrow(Xpr) != length(valid_idx)) {
      return(empty_prediction(
        paste0(
          "prediction_matrix_row_mismatch: expected_",
          length(valid_idx), "_got_", nrow(Xpr)
        ),
        paste(dropped_names, collapse = "|")
      ))
    }
    
    # model.matrix() can omit unused treatment columns. Add any training-kept
    # column absent in prediction as zero, then enforce identical column order.
    missing_pr_cols <- setdiff(kept_names, colnames(Xpr))
    if (length(missing_pr_cols)) {
      z <- matrix(
        0,
        nrow = nrow(Xpr),
        ncol = length(missing_pr_cols),
        dimnames = list(NULL, missing_pr_cols)
      )
      Xpr <- cbind(Xpr, z)
    }
    
    # Conversely, extra columns not used by the full-rank training design are
    # harmless; retain only the columns actually estimated.
    Xpr_keep <- Xpr[, kept_names, drop = FALSE]
    
    pred_log[valid_idx] <- as.numeric(Xpr_keep %*% beta)
  }
  
  # Training-only stock-specific Duan smearing; global fallback.
  pr[, smear_stock := global_smear]
  pr[smear_dt, on = .(Stock), smear_stock := i.smear_stock]
  pr[
    !is.finite(smear_stock) | smear_stock <= 0,
    smear_stock := global_smear
  ]
  
  p <- rep(NA_real_, nrow(pr))
  if (length(valid_idx)) {
    p_scaled <- exp(pred_log[valid_idx]) * pr$smear_stock[valid_idx]
    p[valid_idx] <- p_scaled / VAR_SCALE
  }
  
  # No clipping is used as a modeling device. Positivity follows from exp().
  # FORECAST_EPS is only a last-resort numerical guard against underflow.
  underflow_idx <- which(is.finite(p) & p <= 0)
  if (length(underflow_idx)) p[underflow_idx] <- FORECAST_EPS
  
  pred_out <- pr[, .(
    Stock, Date, Category, liq_state,
    target = get(target_col)
  )]
  pred_out[, forecast := p]
  pred_out[, `:=`(
    horizon = horizon,
    model = "LOG_HAR_X",
    specification = spec_name,
    stage = stage,
    refit_label = refit_label
  )]
  
  nobs_fit <- length(resid_log)
  rss <- sum(resid_log^2)
  tss <- sum((tr$log_har_y - mean(tr$log_har_y))^2)
  r2 <- if (is.finite(tss) && tss > 0) 1 - rss / tss else NA_real_
  k_eff <- length(beta)
  adjr2 <- if (
    is.finite(r2) && nobs_fit > k_eff + 1L
  ) {
    1 - (1 - r2) * (nobs_fit - 1) / (nobs_fit - k_eff)
  } else {
    NA_real_
  }
  sigma_log <- if (nobs_fit > k_eff) {
    sqrt(rss / (nobs_fit - k_eff))
  } else {
    NA_real_
  }
  
  finite_pred <- p[is.finite(p)]
  fit_log <- data.table(
    model = "LOG_HAR_X",
    horizon = horizon,
    specification = spec_name,
    stage = stage,
    refit_label = refit_label,
    n_train = nobs_fit,
    n_stocks = uniqueN(tr$Stock),
    status = "ok",
    design_columns = ncol(Xtr),
    design_rank = qx$rank,
    dropped_collinear_columns_n = length(dropped_names),
    dropped_collinear_columns = paste(dropped_names, collapse = "|"),
    r_squared_log = r2,
    adj_r_squared_log = adjr2,
    sigma_log = sigma_log,
    global_smear = global_smear,
    prediction_rows_total = nrow(pr),
    prediction_rows_complete = length(valid_idx),
    prediction_rows_missing_features = nrow(pr) - length(valid_idx),
    min_forecast = if (length(finite_pred)) min(finite_pred) else NA_real_,
    nonpositive_forecasts = sum(is.finite(p) & p <= 0)
  )
  
  cf <- data.table(
    term = kept_names,
    estimate = beta
  )
  cf[, `:=`(
    model = "LOG_HAR_X",
    horizon = horizon,
    specification = spec_name,
    stage = stage,
    refit_label = refit_label
  )]
  
  list(pred = pred_out, fit_log = fit_log, coef = cf)
}

# Validation periods are annual. Locked-test periods are monthly.
make_har_periods <- function() {
  val <- data.table(
    stage = "validation",
    refit_label = c("validation_2020", "validation_2021",
                    "validation_2022", "validation_2023"),
    period_start = as.IDate(c("2020-01-01", "2021-01-01",
                              "2022-01-01", "2023-01-01")),
    period_end = as.IDate(c("2020-12-31", "2021-12-31",
                            "2022-12-31", as.character(DEV_END)))
  )
  
  test_dates <- sort(unique(GRID[
    Date >= TEST_START & Date <= TEST_END, Date
  ]))
  ym <- unique(format(as.Date(test_dates), "%Y-%m"))
  starts <- as.IDate(paste0(ym, "-01"))
  ends <- shift(starts, type = "lead") - 1L
  ends[length(ends)] <- TEST_END
  
  tst <- data.table(
    stage = "locked_test",
    refit_label = paste0("test_", gsub("-", "", ym)),
    period_start = starts,
    period_end = as.IDate(ends)
  )
  rbind(val, tst)
}

HAR_PERIODS <- make_har_periods()
fwrite(HAR_PERIODS, file.path(OUT_DIR, "log_har_refit_registry.csv"))

cat("\n================ LOG-HAR-X FORECASTING STARTED ================\n")
har_predictions <- list()
har_fit_logs <- list()
har_coefs <- list()
har_counter <- 0L

for (h in HORIZONS) {
  target_col <- paste0("target_yz_", h)
  target_end_col <- paste0("target_end_", h)
  
  for (spec_name in names(HAR_SPECS)) {
    features <- HAR_SPECS[[spec_name]]
    
    for (rr in seq_len(nrow(HAR_PERIODS))) {
      period_start <- HAR_PERIODS$period_start[rr]
      period_end   <- HAR_PERIODS$period_end[rr]
      stage        <- HAR_PERIODS$stage[rr]
      refit_label  <- HAR_PERIODS$refit_label[rr]
      
      # Strict outcome-availability rule:
      # training target must have fully ended before the refit date.
      train_dt <- GRID[
        Date >= DEV_START &
          Date < period_start &
          get(target_end_col) < period_start &
          observed_row == 1L &
          is.finite(get(target_col))
      ]
      
      pred_dt <- GRID[
        Date >= period_start & Date <= period_end &
          observed_row == 1L &
          is.finite(get(target_col))
      ]
      
      if (!nrow(pred_dt)) next
      
      ans <- loghar_fit_predict(
        train_dt = train_dt,
        pred_dt = pred_dt,
        target_col = target_col,
        features = features,
        horizon = h,
        spec_name = spec_name,
        refit_label = refit_label,
        stage = stage
      )
      
      har_counter <- har_counter + 1L
      har_predictions[[har_counter]] <- ans$pred
      har_fit_logs[[har_counter]] <- ans$fit_log
      if (!is.null(ans$coef)) har_coefs[[har_counter]] <- ans$coef
      
      if (har_counter %% 25L == 0L) {
        cat("LOG-HAR-X completed fits:", har_counter, "\n")
      }
    }
  }
}

HAR_PRED <- rbindlist(har_predictions, fill = TRUE)
HAR_FIT_LOG <- rbindlist(har_fit_logs, fill = TRUE)
HAR_COEF <- if (length(har_coefs)) rbindlist(har_coefs, fill = TRUE) else data.table()

HAR_PRED[, qlike := qlike_loss(target, forecast)]
HAR_PRED[, mse := mse_loss(target, forecast)]

LOGHAR_POSITIVITY_AUDIT <- HAR_PRED[, .(
  n_forecasts = .N,
  n_finite = sum(is.finite(forecast)),
  n_nonpositive = sum(is.finite(forecast) & forecast <= 0),
  n_at_numerical_floor = sum(is.finite(forecast) & forecast <= FORECAST_EPS),
  min_forecast = if (any(is.finite(forecast))) min(forecast, na.rm = TRUE) else NA_real_,
  p001_forecast = if (any(is.finite(forecast))) {
    as.numeric(quantile(forecast, 0.001, na.rm = TRUE))
  } else NA_real_
), by = .(stage, horizon, specification)]

fwrite(
  LOGHAR_POSITIVITY_AUDIT,
  file.path(OUT_DIR, "log_har_positivity_audit.csv")
)

fwrite(HAR_PRED, file.path(OUT_DIR, "log_har_x_forecasts.csv.gz"),
       compress = "gzip", na = "NA")
fwrite(HAR_FIT_LOG, file.path(OUT_DIR, "log_har_x_fit_log.csv"))
if (nrow(HAR_COEF)) {
  fwrite(HAR_COEF, file.path(OUT_DIR, "log_har_x_coefficients_all_refits.csv.gz"),
         compress = "gzip")
}
cat("================ LOG-HAR-X FORECASTING COMPLETE ================\n")

# =============================================================================
# PART III. GJR-GARCH(1,1) CONVENTIONAL BENCHMARK
# =============================================================================

# Create quarterly test/validation refit periods.
make_gjr_periods <- function(refit_months = GJR_REFIT_MONTHS) {
  all_dates <- sort(unique(GRID[
    Date >= as.IDate("2020-01-01") & Date <= TEST_END, Date
  ]))
  first_month <- as.Date(format(min(as.Date(all_dates)), "%Y-%m-01"))
  last_month  <- as.Date(format(max(as.Date(all_dates)), "%Y-%m-01"))
  month_seq <- seq(first_month, last_month, by = paste(refit_months, "months"))
  
  starts <- as.IDate(month_seq)
  ends <- as.IDate(c(month_seq[-1] - 1, as.Date(TEST_END)))
  
  out <- data.table(
    period_start = starts,
    period_end = ends
  )
  out[, stage := fifelse(period_start >= TEST_START, "locked_test", "validation")]
  out[, refit_label := paste0(
    "gjr_", format(as.Date(period_start), "%Y%m"),
    "_", format(as.Date(period_end), "%Y%m")
  )]
  out
}

GJR_PERIODS <- make_gjr_periods()
fwrite(GJR_PERIODS, file.path(OUT_DIR, "gjr_refit_registry.csv"))

gjr_spec <- ugarchspec(
  variance.model = list(
    model = "gjrGARCH",
    garchOrder = c(1L, 1L)
  ),
  mean.model = list(
    armaOrder = c(0L, 0L),
    include.mean = TRUE
  ),
  distribution.model = GJR_DISTRIBUTION
)

# Fit once at a period boundary, then update the GJR state recursively through
# that refit period using returns actually observed by each forecast origin.
run_gjr_stock_period <- function(stock_name, period_start, period_end,
                                 refit_label, stage) {
  st <- GRID[Stock == stock_name]
  setorder(st, Date)
  
  train_ret <- st[
    Date < period_start & observed_row == 1L & is.finite(ret_cc),
    ret_cc * GJR_RETURN_SCALE
  ]
  
  base_log <- data.table(
    Stock = stock_name, refit_label = refit_label, stage = stage,
    period_start = period_start, period_end = period_end,
    n_train_returns = length(train_ret)
  )
  
  if (length(train_ret) < GJR_MIN_TRAIN_RETURNS) {
    base_log[, `:=`(
      status = "insufficient_training_returns",
      convergence = NA_integer_,
      omega = NA_real_, alpha1 = NA_real_, beta1 = NA_real_,
      gamma1 = NA_real_, mu = NA_real_, persistence = NA_real_
    )]
    return(list(pred = NULL, fit_log = base_log))
  }
  
  fit <- tryCatch(
    ugarchfit(
      spec = gjr_spec,
      data = train_ret,
      solver = "hybrid",
      solver.control = list(trace = 0)
    ),
    error = function(e) e
  )
  
  if (inherits(fit, "error")) {
    base_log[, `:=`(
      status = paste0("fit_error: ", conditionMessage(fit)),
      convergence = NA_integer_,
      omega = NA_real_, alpha1 = NA_real_, beta1 = NA_real_,
      gamma1 = NA_real_, mu = NA_real_, persistence = NA_real_
    )]
    return(list(pred = NULL, fit_log = base_log))
  }
  
  conv <- fit@fit$convergence
  cf <- coef(fit)
  getcf <- function(nm, default = NA_real_) {
    if (nm %in% names(cf)) as.numeric(cf[[nm]]) else default
  }
  
  omega <- getcf("omega")
  alpha <- getcf("alpha1")
  beta  <- getcf("beta1")
  gamma <- getcf("gamma1")
  mu    <- getcf("mu", 0)
  
  persistence <- alpha + beta + 0.5 * gamma
  
  sig <- as.numeric(sigma(fit))
  res <- as.numeric(residuals(fit))
  last_sig2 <- tail(sig[is.finite(sig)], 1)^2
  last_eps  <- tail(res[is.finite(res)], 1)
  
  good_fit <- conv == 0L &&
    all(is.finite(c(omega, alpha, beta, gamma, mu, persistence,
                    last_sig2, last_eps))) &&
    omega >= 0 && last_sig2 > 0
  
  base_log[, `:=`(
    status = if (good_fit) "ok" else "invalid_fit_parameters",
    convergence = conv,
    omega = omega, alpha1 = alpha, beta1 = beta,
    gamma1 = gamma, mu = mu, persistence = persistence
  )]
  
  if (!good_fit) return(list(pred = NULL, fit_log = base_log))
  
  walk <- st[
    Date >= period_start & Date <= period_end &
      observed_row == 1L,
    .(
      Stock, Date, Category, liq_state, ret_cc,
      target_yz_5, target_yz_10, target_yz_20
    )
  ]
  if (!nrow(walk)) return(list(pred = NULL, fit_log = base_log))
  
  out_list <- vector("list", nrow(walk))
  out_n <- 0L
  
  sig2_prev <- last_sig2
  eps_prev <- last_eps
  
  for (ii in seq_len(nrow(walk))) {
    r_pct <- walk$ret_cc[ii] * GJR_RETURN_SCALE
    
    # If return is unavailable (e.g., segment reset), do not fabricate a state.
    if (!is.finite(r_pct)) next
    
    sig2_now <- omega +
      alpha * eps_prev^2 +
      gamma * as.numeric(eps_prev < 0) * eps_prev^2 +
      beta * sig2_prev
    
    if (!is.finite(sig2_now) || sig2_now <= 0) next
    
    eps_now <- r_pct - mu
    
    # h-step expected GJR variance path from origin t.
    v1 <- omega +
      alpha * eps_now^2 +
      gamma * as.numeric(eps_now < 0) * eps_now^2 +
      beta * sig2_now
    
    if (!is.finite(v1) || v1 <= 0) {
      sig2_prev <- sig2_now
      eps_prev <- eps_now
      next
    }
    
    preds <- numeric(length(HORIZONS))
    names(preds) <- as.character(HORIZONS)
    
    maxh <- max(HORIZONS)
    vv <- numeric(maxh)
    vv[1L] <- v1
    if (maxh > 1L) {
      for (kk in 2:maxh) {
        vv[kk] <- omega + persistence * vv[kk - 1L]
      }
    }
    
    for (hh in HORIZONS) {
      # Convert percentage-return variance back to raw log-return variance.
      preds[as.character(hh)] <- mean(vv[seq_len(hh)]) / VAR_SCALE
    }
    
    out_n <- out_n + 1L
    out_list[[out_n]] <- rbindlist(lapply(HORIZONS, function(hh) {
      target_nm <- paste0("target_yz_", hh)
      data.table(
        Stock = walk$Stock[ii],
        Date = walk$Date[ii],
        Category = walk$Category[ii],
        liq_state = walk$liq_state[ii],
        target = walk[[target_nm]][ii],
        forecast = preds[as.character(hh)],
        horizon = hh,
        model = "GJR_GARCH",
        specification = "gjrGARCH_1_1_std",
        stage = stage,
        refit_label = refit_label
      )
    }))
    
    sig2_prev <- sig2_now
    eps_prev <- eps_now
  }
  
  pred <- if (out_n) rbindlist(out_list[seq_len(out_n)], fill = TRUE) else NULL
  list(pred = pred, fit_log = base_log)
}

cat("\n================ GJR-GARCH FORECASTING STARTED ================\n")
gjr_preds <- list()
gjr_logs <- list()
gjr_n <- 0L

for (rr in seq_len(nrow(GJR_PERIODS))) {
  ps <- GJR_PERIODS$period_start[rr]
  pe <- GJR_PERIODS$period_end[rr]
  lab <- GJR_PERIODS$refit_label[rr]
  stg <- GJR_PERIODS$stage[rr]
  
  for (ss in eligible_stocks) {
    ans <- run_gjr_stock_period(ss, ps, pe, lab, stg)
    gjr_n <- gjr_n + 1L
    gjr_logs[[gjr_n]] <- ans$fit_log
    if (!is.null(ans$pred)) gjr_preds[[gjr_n]] <- ans$pred
    
    if (gjr_n %% 50L == 0L) {
      cat("GJR stock-refits completed:", gjr_n, "\n")
    }
  }
}

GJR_PRED <- if (length(gjr_preds)) rbindlist(gjr_preds, fill = TRUE) else data.table()
GJR_FIT_LOG <- rbindlist(gjr_logs, fill = TRUE)

if (nrow(GJR_PRED)) {
  GJR_PRED <- GJR_PRED[is.finite(target) & is.finite(forecast)]
  GJR_PRED[, qlike := qlike_loss(target, forecast)]
  GJR_PRED[, mse := mse_loss(target, forecast)]
  fwrite(GJR_PRED, file.path(OUT_DIR, "gjr_garch_forecasts.csv.gz"),
         compress = "gzip", na = "NA")
}
fwrite(GJR_FIT_LOG, file.path(OUT_DIR, "gjr_garch_fit_log.csv"))

# Stock-level GJR diagnostics. These are essential in a thinly traded market:
# pooled QLIKE can otherwise be dominated by a small number of near-zero
# return-variance forecasts while the OHLC target remains positive.
if (nrow(GJR_PRED)) {
  GJR_STOCK_METRICS <- GJR_PRED[
    is.finite(target) & is.finite(forecast),
    .(
      n = .N,
      mean_target = mean(target),
      mean_forecast = mean(forecast),
      min_forecast = min(forecast),
      p01_forecast = as.numeric(quantile(forecast, 0.01, na.rm = TRUE)),
      median_forecast = median(forecast),
      QLIKE = mean(qlike_loss(target, forecast)),
      MSE = mean(mse_loss(target, forecast)),
      share_forecast_lt_1e_12 = mean(forecast < 1e-12),
      share_forecast_lt_target_1e_6 = mean(
        forecast / pmax(target, TARGET_EPS) < 1e-6
      )
    ),
    by = .(Stock, Category, stage, horizon)
  ]
  fwrite(
    GJR_STOCK_METRICS,
    file.path(OUT_DIR, "gjr_stock_level_diagnostics.csv")
  )
  
  # Every stock receives equal weight in this summary, regardless of its number
  # of forecast origins. Report both mean and median cross-stock losses.
  GJR_EQUAL_STOCK <- GJR_STOCK_METRICS[
    ,
    .(
      n_stocks = uniqueN(Stock),
      mean_stock_QLIKE = mean(QLIKE),
      median_stock_QLIKE = median(QLIKE),
      p90_stock_QLIKE = as.numeric(quantile(QLIKE, 0.90, na.rm = TRUE)),
      mean_stock_MSE = mean(MSE),
      median_stock_MSE = median(MSE)
    ),
    by = .(stage, horizon)
  ]
  fwrite(
    GJR_EQUAL_STOCK,
    file.path(OUT_DIR, "gjr_equal_stock_weight_metrics.csv")
  )
}

cat("================ GJR-GARCH FORECASTING COMPLETE ================\\n")

# =============================================================================
# PART IV. R-SIDE EVALUATION AND INFERENCE
# =============================================================================

metric_table <- function(dt) {
  dt[
    is.finite(target) & is.finite(forecast),
    .(
      n = .N,
      QLIKE = mean(qlike_loss(target, forecast)),
      MSE = mean(mse_loss(target, forecast))
    ),
    by = .(model, specification, stage, horizon)
  ]
}

R_PRED <- copy(HAR_PRED)
if (nrow(GJR_PRED)) R_PRED <- rbindlist(list(R_PRED, GJR_PRED), fill = TRUE)

R_METRICS <- metric_table(R_PRED)
fwrite(R_METRICS, file.path(OUT_DIR, "r_model_metrics.csv"))

# Stock-level loss summaries for all R models/specifications.
R_STOCK_METRICS <- R_PRED[
  is.finite(target) & is.finite(forecast),
  .(
    n = .N,
    QLIKE = mean(qlike_loss(target, forecast)),
    MSE = mean(mse_loss(target, forecast))
  ),
  by = .(Stock, model, specification, stage, horizon)
]
fwrite(
  R_STOCK_METRICS,
  file.path(OUT_DIR, "r_stock_level_metrics.csv")
)

R_EQUAL_STOCK_METRICS <- R_STOCK_METRICS[
  ,
  .(
    n_stocks = uniqueN(Stock),
    mean_stock_QLIKE = mean(QLIKE),
    median_stock_QLIKE = median(QLIKE),
    mean_stock_MSE = mean(MSE),
    median_stock_MSE = median(MSE)
  ),
  by = .(model, specification, stage, horizon)
]
fwrite(
  R_EQUAL_STOCK_METRICS,
  file.path(OUT_DIR, "r_equal_stock_weight_metrics.csv")
)

# ----------------------- HAR B vs C: COMMON SUPPORT --------------------------
har_bc <- merge(
  HAR_PRED[
    specification == "B_primitive",
    .(Stock, Date, Category, liq_state, horizon, stage,
      target_B = target, forecast_B = forecast)
  ],
  HAR_PRED[
    specification == "C_primitive_plus_TI",
    .(Stock, Date, horizon, stage,
      target_C = target, forecast_C = forecast)
  ],
  by = c("Stock", "Date", "horizon", "stage"),
  all = FALSE
)

har_bc <- har_bc[
  is.finite(target_B) & is.finite(target_C) &
    is.finite(forecast_B) & is.finite(forecast_C)
]

# Targets should be numerically identical on the merged support.
har_bc[, target := target_B]
har_bc[, qB := qlike_loss(target, forecast_B)]
har_bc[, qC := qlike_loss(target, forecast_C)]
har_bc[, mB := mse_loss(target, forecast_B)]
har_bc[, mC := mse_loss(target, forecast_C)]
har_bc[, d_qlike_B_minus_C := qB - qC]
har_bc[, d_mse_B_minus_C := mB - mC]

HAR_INCREMENTAL <- har_bc[, .(
  n = .N,
  QLIKE_B = mean(qB),
  QLIKE_C = mean(qC),
  Incremental_Gain_QLIKE_pct = 100 * (mean(qB) - mean(qC)) / mean(qB),
  MSE_B = mean(mB),
  MSE_C = mean(mC),
  Incremental_Gain_MSE_pct = 100 * (mean(mB) - mean(mC)) / mean(mB)
), by = .(stage, horizon)]

fwrite(HAR_INCREMENTAL, file.path(OUT_DIR, "log_har_incremental_B_vs_C.csv"))

# Liquidity-conditioned locked-test incremental value.
HAR_LIQ <- har_bc[
  stage == "locked_test" & !is.na(liq_state),
  .(
    n = .N,
    QLIKE_B = mean(qB),
    QLIKE_C = mean(qC),
    Incremental_Gain_QLIKE_pct = 100 * (mean(qB) - mean(qC)) / mean(qB),
    MSE_B = mean(mB),
    MSE_C = mean(mC),
    Incremental_Gain_MSE_pct = 100 * (mean(mB) - mean(mC)) / mean(mB)
  ),
  by = .(horizon, liq_state)
]
fwrite(HAR_LIQ, file.path(OUT_DIR, "log_har_liquidity_heterogeneity.csv"))

# Daily cross-sectional mean loss differential protects against treating all
# stock-date rows as independent.
HAR_DAILY_DIFF <- har_bc[
  ,
  .(
    n_stocks = .N,
    d_qlike = mean(d_qlike_B_minus_C),
    d_mse = mean(d_mse_B_minus_C)
  ),
  by = .(stage, horizon, Date)
]
fwrite(HAR_DAILY_DIFF, file.path(OUT_DIR, "log_har_daily_loss_differentials.csv"))

# HAC + moving-block bootstrap on the daily loss-difference series.
infer_list <- list()
jj <- 0L
for (stg in unique(HAR_DAILY_DIFF$stage)) {
  for (hh in HORIZONS) {
    x <- HAR_DAILY_DIFF[stage == stg & horizon == hh][order(Date), d_qlike]
    if (!length(x)) next
    
    # Overlapping h-day targets imply at least h-1 serial correlation.
    hac_lag <- max(hh - 1L, 1L)
    # Block length also respects the forecast horizon.
    block_len <- max(hh, 10L)
    
    hac <- nw_mean_test(x, lag = hac_lag)
    boot <- mbb_mean_test(
      x, block_length = block_len,
      B = BOOTSTRAP_REPS,
      seed = BOOTSTRAP_SEED + hh
    )
    
    jj <- jj + 1L
    infer_list[[jj]] <- cbind(
      data.table(stage = stg, horizon = hh),
      hac, boot
    )
  }
}
HAR_INFERENCE <- rbindlist(infer_list, fill = TRUE)

# Holm adjustment across the three horizons within each stage.
HAR_INFERENCE[, p_hac_holm := p.adjust(p_hac, method = "holm"), by = stage]
HAR_INFERENCE[, p_boot_holm := p.adjust(p_boot, method = "holm"), by = stage]
fwrite(HAR_INFERENCE, file.path(OUT_DIR, "log_har_B_vs_C_inference.csv"))

# -------------------------- FAMILY ABLATIONS ---------------------------------
if (RUN_FAMILY_ABLATIONS) {
  fam_specs <- c("B_plus_trend", "B_plus_momentum", "B_plus_range_TI")
  family_out <- list()
  kk <- 0L
  
  for (sp in fam_specs) {
    tmp <- merge(
      HAR_PRED[
        specification == "B_primitive",
        .(Stock, Date, horizon, stage, target, forecast_B = forecast)
      ],
      HAR_PRED[
        specification == sp,
        .(Stock, Date, horizon, stage, forecast_F = forecast)
      ],
      by = c("Stock", "Date", "horizon", "stage"),
      all = FALSE
    )
    tmp <- tmp[
      is.finite(target) & is.finite(forecast_B) & is.finite(forecast_F)
    ]
    if (!nrow(tmp)) next
    
    tmp[, qB := qlike_loss(target, forecast_B)]
    tmp[, qF := qlike_loss(target, forecast_F)]
    
    kk <- kk + 1L
    family_out[[kk]] <- tmp[, .(
      n = .N,
      QLIKE_B = mean(qB),
      QLIKE_Family = mean(qF),
      Incremental_Gain_QLIKE_pct = 100 * (mean(qB) - mean(qF)) / mean(qB)
    ), by = .(stage, horizon)][, family := sp]
  }
  
  if (length(family_out)) {
    HAR_FAMILY <- rbindlist(family_out, fill = TRUE)
    fwrite(HAR_FAMILY, file.path(OUT_DIR, "log_har_indicator_family_ablations.csv"))
  }
}

# -------------------------- BENCHMARK COMPARISON ------------------------------
# Comparison on common support between HAR-C and GJR-GARCH.
if (nrow(GJR_PRED)) {
  harc <- HAR_PRED[
    specification == "C_primitive_plus_TI",
    .(Stock, Date, horizon, stage, target, forecast_HAR_C = forecast)
  ]
  gjr0 <- GJR_PRED[
    ,
    .(Stock, Date, horizon, stage, forecast_GJR = forecast)
  ]
  cmp <- merge(
    harc, gjr0,
    by = c("Stock", "Date", "horizon", "stage"),
    all = FALSE
  )
  cmp <- cmp[
    is.finite(target) & is.finite(forecast_HAR_C) & is.finite(forecast_GJR)
  ]
  cmp[, qHAR := qlike_loss(target, forecast_HAR_C)]
  cmp[, qGJR := qlike_loss(target, forecast_GJR)]
  
  GJR_HAR_CMP <- cmp[, .(
    n = .N,
    QLIKE_HAR_C = mean(qHAR),
    QLIKE_GJR = mean(qGJR),
    HAR_C_vs_GJR_Gain_pct = 100 * (mean(qGJR) - mean(qHAR)) / mean(qGJR)
  ), by = .(stage, horizon)]
  
  fwrite(GJR_HAR_CMP, file.path(OUT_DIR, "log_har_C_vs_gjr_common_support.csv"))
}

# =============================================================================
# PART V. COLAB HANDOFF FILES
# =============================================================================

# Frozen panel for Python. Keep only model-facing columns and retain NAs;
# Colab must fit imputation/scaling on training data within each fold.
COLAB_COLS <- unique(c(
  "Stock", "Date", "Category", "observed_row", "ohlc_valid",
  "segment_id", "corp_event", "split_label", "liq_state", "obs_share20",
  paste0("target_yz_", HORIZONS),
  paste0("target_end_", HORIZONS),
  HAR_A_VARS,
  HAR_B_EXTRA,
  TECH_VARS
))
COLAB_COLS <- intersect(COLAB_COLS, names(GRID))

COLAB_PANEL <- GRID[
  observed_row == 1L,
  ..COLAB_COLS
]

fwrite(
  COLAB_PANEL,
  file.path(OUT_DIR, "COLAB_FROZEN_PANEL.csv.gz"),
  compress = "gzip",
  na = "NA"
)

# Save R forecasts in a single compact file so Python can later combine
# LightGBM/LSTM with the exact R benchmarks.
fwrite(
  R_PRED,
  file.path(OUT_DIR, "R_ALL_FORECASTS.csv.gz"),
  compress = "gzip",
  na = "NA"
)

# Human-readable handoff note.
handoff_lines <- c(
  "SIGNAL OR REDUNDANCY? -- COLAB HANDOFF",
  "",
  "Primary Python input: COLAB_FROZEN_PANEL.csv.gz",
  "R benchmark forecasts: R_ALL_FORECASTS.csv.gz",
  "Primary R econometric model: positivity-preserving Log-HAR-X with training-only Duan smearing",
  "Feature definitions: feature_manifest.csv",
  "Fold definitions: fold_registry.csv and log_har_refit_registry.csv",
  "",
  "Do NOT reconstruct targets or technical indicators in Python.",
  "Do NOT use random train/test splits.",
  "Fit every imputer/scaler only on the training portion of each chronological fold.",
  "Use 5 days as the primary horizon; 10 and 20 days are robustness horizons.",
  "Primary comparison: B_primitive vs C_primitive_plus_TI.",
  "Python models to fit next: LightGBM and compact pooled LSTM.",
  ""
)
writeLines(handoff_lines, file.path(OUT_DIR, "README_COLAB_HANDOFF.txt"))

# Session information and input-file checksum for reproducibility.
writeLines(capture.output(sessionInfo()), file.path(OUT_DIR, "R_sessionInfo.txt"))
checksum <- data.table(
  file = MASTER_FILENAME,
  md5 = unname(tools::md5sum(INPUT_FILE))
)
fwrite(checksum, file.path(OUT_DIR, "input_file_md5.csv"))

# Compact run-summary file.
run_summary <- rbindlist(list(
  audit_overall[, .(section = "preprocessing", metric, value = as.character(value))],
  data.table(
    section = "modeling",
    metric = c(
      "har_forecast_rows", "har_successful_fits",
      "gjr_forecast_rows", "gjr_successful_refits",
      "colab_panel_rows"
    ),
    value = as.character(c(
      nrow(HAR_PRED),
      HAR_FIT_LOG[status == "ok", .N],
      nrow(GJR_PRED),
      GJR_FIT_LOG[status == "ok", .N],
      nrow(COLAB_PANEL)
    ))
  )
), fill = TRUE)
fwrite(run_summary, file.path(OUT_DIR, "MASTER_RUN_SUMMARY.csv"))

# Save the exact master R script in the output folder when the script was
# invoked via source("...") or Rscript.
current_script <- tryCatch({
  of <- sys.frame(1)$ofile
  if (!is.null(of) && file.exists(of)) normalizePath(of) else NA_character_
}, error = function(e) NA_character_)

if (is.character(current_script) && length(current_script) == 1L &&
    !is.na(current_script) && file.exists(current_script)) {
  file.copy(
    current_script,
    file.path(OUT_DIR, "MASTER_SIGNAL_REDUNDANCY_R_FULL_PIPELINE_LOGHAR_FIXED.R"),
    overwrite = TRUE
  )
}

RUN_FINISHED_AT <- Sys.time()
runtime_manifest <- data.table(
  item = c("run_started_at", "run_finished_at", "elapsed_minutes"),
  value = c(
    format(RUN_STARTED_AT, "%Y-%m-%d %H:%M:%S %Z"),
    format(RUN_FINISHED_AT, "%Y-%m-%d %H:%M:%S %Z"),
    sprintf("%.3f", as.numeric(difftime(
      RUN_FINISHED_AT, RUN_STARTED_AT, units = "mins"
    )))
  )
)
fwrite(runtime_manifest, file.path(OUT_DIR, "runtime_manifest.csv"))


# =============================================================================
# PART VI. ZIP EVERYTHING INTO ONE ARCHIVE
# =============================================================================

if (file.exists(ZIP_FILE)) file.remove(ZIP_FILE)

old_wd <- getwd()
on.exit(setwd(old_wd), add = TRUE)
setwd(DATA_DIR)

zip::zipr(
  zipfile = ZIP_FILE,
  files = basename(OUT_DIR),
  recurse = TRUE,
  include_directories = TRUE
)

setwd(old_wd)

cat("\n============================================================\n")
cat("FULL R PIPELINE COMPLETE\n")
cat("============================================================\n")
cat("Output folder:\n  ", normalizePath(OUT_DIR, winslash = "/", mustWork = TRUE), "\n")
cat("ZIP archive:\n  ", normalizePath(ZIP_FILE, winslash = "/", mustWork = TRUE), "\n")
cat("\nKey files inside the ZIP:\n")
cat("  COLAB_FROZEN_PANEL.csv.gz\n")
cat("  R_ALL_FORECASTS.csv.gz\n")
cat("  log_har_incremental_B_vs_C.csv\n")
cat("  log_har_B_vs_C_inference.csv\n")
cat("  log_har_liquidity_heterogeneity.csv\n")
cat("  r_model_metrics.csv\n")
cat("  MASTER_RUN_SUMMARY.csv\n")
cat("============================================================\n")
