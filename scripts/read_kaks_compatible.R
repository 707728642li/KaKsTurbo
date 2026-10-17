# Standalone reader: preserve Sequence IDs including the literal string "NA".
read_kaks_compatible <- function(path) {
    x <- read.table(path, header=TRUE, sep="\t", quote="", comment.char="",
                    colClasses="character", na.strings=character(),
                    check.names=FALSE, stringsAsFactors=FALSE)
    stopifnot(ncol(x)==22L,
              identical(names(x)[1:6], c("Sequence","Method","Ka","Ks","Ka/Ks","P-Value(Fisher)")))
    numeric_columns <- c("Ka","Ks","Ka/Ks","P-Value(Fisher)","Length",
                         "S-Sites","N-Sites","Substitutions","Syn-Subs","Nonsyn-Subs",
                         "Divergence-Distance","ML-Score","AICc","Akaike-Weight")
    for (n in numeric_columns) {
        missing <- x[[n]] == "NA"
        y <- suppressWarnings(as.numeric(x[[n]]))
        if (any(!missing & (!is.finite(y) | is.na(y)))) stop(paste("Invalid numeric field", n))
        y[missing] <- NA_real_
        x[[n]] <- y
    }
    x
}
