root <- normalizePath(dirname(commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))]), mustWork = FALSE)
root <- sub("^--file=", "", root)
root <- dirname(normalizePath(root))

load(file.path(root, "ensemblepp", "data", "temp.rda"))
write.csv(temp, file.path(root, "temp.csv"), row.names = FALSE)

load(file.path(root, "ensembleBMA", "data", "srft.rda"))
write.csv(srft, file.path(root, "srft.csv"), row.names = FALSE)

cat(R.version.string, "\n")
cat("Innsbruck:", nrow(temp), "rows,", ncol(temp) - 1, "members,", sum(is.na(temp)), "missing values\n")
cat("Pacific Northwest:", nrow(srft), "rows,", length(unique(srft$station)), "stations,", length(unique(srft$date)), "dates,", sum(is.na(srft)), "missing values\n")
