// The metrics tables a QC run writes: errors_membrane.csv, errors_sparse.csv,
// hole_summary.csv, refspace_summary.csv, and refspace_movie_summary.csv where
// the run wrote a movie. The set differs between runs, so callers discover it
// rather than naming the files.
//
// Cells stay as strings. The Python side guesses int, then float, then falls
// back to string per cell, which makes a column's type depend on its values —
// a frame column reads as int until one row is empty. Here the table reports
// text and the caller converts the column it knows the meaning of, so a
// malformed number throws at the point where someone wanted a number.
#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace wrdash {

struct CsvTable {
  std::string name;  // file stem, e.g. "errors_membrane"
  std::vector<std::string> columns;
  std::vector<std::vector<std::string>> rows;

  // Index of `column`, or a throw naming the column and the ones that do
  // exist. Callers index by name because column order has changed between
  // pipeline revisions.
  std::size_t column_index(const std::string& column) const;

  bool has_column(const std::string& column) const;

  // Throws if the cell is not a complete number — "" and "1.0abc" both throw
  // rather than yielding 0.0, which would plot as a real measurement.
  double number(std::size_t row, const std::string& column) const;
};

// Throws on a missing file, a file with no header line, or any data row whose
// cell count differs from the header's. A short row would otherwise shift
// every column after the gap.
CsvTable read_csv(const std::filesystem::path& path);

// Every *.csv directly under `dir`, sorted by name. A directory with no CSV
// files yields an empty vector; a missing directory throws.
std::vector<CsvTable> read_csv_dir(const std::filesystem::path& dir);

}  // namespace wrdash
