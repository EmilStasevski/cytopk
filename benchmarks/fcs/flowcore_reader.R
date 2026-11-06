# flowCore adapter for the FCS benchmarks.
#   Rscript flowcore_reader.R read  <out_dir> <file>...   per file: <out_dir>/<i>.bin (float64, row-major
#                                                         events x params), <i>.json {ok, names, n, p, error}
#   Rscript flowcore_reader.R time  <repeats> <file>       prints seconds of each read.FCS call, one per line
#   Rscript flowcore_reader.R import                       loads flowCore only (baseline for peak RSS)
# read.FCS options: transformation = FALSE (values as stored), truncate_max_range = FALSE (no clipping at
# $PnR); everything else is flowCore's default.
suppressPackageStartupMessages(library(flowCore))
args <- commandArgs(trailingOnly = TRUE)
rd <- function(f) read.FCS(f, transformation = FALSE, truncate_max_range = FALSE)
esc <- function(s) gsub("[[:cntrl:]]", " ", gsub("\"", "\\\\\"", gsub("\\\\", "\\\\\\\\", s)))
if (args[1] == "read") {
  out <- args[2]
  files <- args[-(1:2)]
  for (i in seq_along(files)) {
    js <- file.path(out, paste0(i - 1, ".json"))
    res <- tryCatch({
      ff <- suppressWarnings(rd(files[i]))
      m <- exprs(ff)
      con <- file(file.path(out, paste0(i - 1, ".bin")), "wb")
      writeBin(as.vector(t(m)), con, size = 8)
      close(con)
      nm <- paste0('"', esc(colnames(m)), '"', collapse = ",")
      sprintf('{"ok": true, "names": [%s], "n": %d, "p": %d}', nm, nrow(m), ncol(m))
    }, error = function(e) sprintf('{"ok": false, "error": "%s"}', esc(conditionMessage(e))))
    writeLines(res, js)
  }
} else if (args[1] == "time") {
  reps <- as.integer(args[2])
  for (r in seq_len(reps)) {
    t0 <- proc.time()[["elapsed"]]
    ff <- rd(args[3])
    m <- exprs(ff)
    storage.mode(m) <- "double"
    cat(proc.time()[["elapsed"]] - t0, "\n")
    rm(ff, m); invisible(gc())
  }
}
