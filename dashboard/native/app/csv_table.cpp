#include "app/csv_table.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace wrdash {

namespace {

// Splits on commas only. The pipeline writes these files with csv.writer over
// numeric columns and bare identifiers, so no cell is quoted or embeds a
// comma; a quoted cell would arrive here with its quotes intact and fail the
// number() conversion loudly rather than being silently mis-split.
std::vector<std::string> split_row(const std::string& line) {
  std::vector<std::string> cells;
  std::string cell;
  std::istringstream stream(line);
  while (std::getline(stream, cell, ',')) cells.push_back(cell);
  // getline drops a trailing empty field: "a,b," yields two cells, not three.
  if (!line.empty() && line.back() == ',') cells.emplace_back();
  return cells;
}

std::string strip_cr(std::string s) {
  if (!s.empty() && s.back() == '\r') s.pop_back();
  return s;
}

}  // namespace

std::size_t CsvTable::column_index(const std::string& column) const {
  for (std::size_t i = 0; i < columns.size(); ++i) {
    if (columns[i] == column) return i;
  }
  std::string known;
  for (const auto& c : columns) {
    if (!known.empty()) known += ", ";
    known += c;
  }
  throw std::runtime_error("CSV table '" + name + "' has no column '" + column + "'; columns are: " +
                           known);
}

bool CsvTable::has_column(const std::string& column) const {
  return std::find(columns.begin(), columns.end(), column) != columns.end();
}

double CsvTable::number(std::size_t row, const std::string& column) const {
  if (row >= rows.size()) {
    throw std::runtime_error("CSV table '" + name + "' has " + std::to_string(rows.size()) +
                             " rows; asked for row " + std::to_string(row));
  }
  const std::string& cell = rows[row][column_index(column)];
  errno = 0;
  char* end = nullptr;
  const double value = std::strtod(cell.c_str(), &end);
  if (end == cell.c_str() || *end != '\0' || errno == ERANGE) {
    throw std::runtime_error("CSV table '" + name + "' row " + std::to_string(row) + " column '" +
                             column + "' is not a number: '" + cell + "'");
  }
  return value;
}

CsvTable read_csv(const std::filesystem::path& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open CSV " + path.string());

  CsvTable table;
  table.name = path.stem().string();

  std::string line;
  if (!std::getline(f, line)) {
    throw std::runtime_error("CSV " + path.string() + " is empty: no header line");
  }
  table.columns = split_row(strip_cr(line));

  std::size_t line_no = 1;
  while (std::getline(f, line)) {
    ++line_no;
    line = strip_cr(line);
    if (line.empty()) continue;  // trailing newline at end of file
    auto cells = split_row(line);
    if (cells.size() != table.columns.size()) {
      throw std::runtime_error("CSV " + path.string() + " line " + std::to_string(line_no) +
                               " has " + std::to_string(cells.size()) + " cells, header has " +
                               std::to_string(table.columns.size()));
    }
    table.rows.push_back(std::move(cells));
  }
  return table;
}

std::vector<CsvTable> read_csv_dir(const std::filesystem::path& dir) {
  if (!std::filesystem::is_directory(dir)) {
    throw std::runtime_error("not a directory: " + dir.string());
  }
  std::vector<std::filesystem::path> files;
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (entry.is_regular_file() && entry.path().extension() == ".csv") {
      files.push_back(entry.path());
    }
  }
  std::sort(files.begin(), files.end());

  std::vector<CsvTable> tables;
  tables.reserve(files.size());
  for (const auto& p : files) tables.push_back(read_csv(p));
  return tables;
}

}  // namespace wrdash
