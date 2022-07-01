#include "Standalone/Utils.h"
#include <fstream>
#include <limits>
#include <string>

namespace mlir {
using namespace mlir;
// Trim functions taken from https://stackoverflow.com/a/25385766
const char *ws = " \t\n\r\f\v";

// trim from end of string (right)
inline std::string &rtrim(std::string &s, const char *t = ws) {
  s.erase(s.find_last_not_of(t) + 1);
  return s;
}

// trim from beginning of string (left)
inline std::string &ltrim(std::string &s, const char *t = ws) {
  s.erase(0, s.find_first_not_of(t));
  return s;
}

// trim from both ends of string (right then left)
inline std::string &trim(std::string &s, const char *t = ws) {
  return ltrim(rtrim(s, t), t);
}

std::fstream &gotoLine(std::fstream &file, unsigned int num) {
  file.seekg(std::ios::beg);
  for (unsigned int i = 0; i < num - 1; ++i) {
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  return file;
}

void DEBUGpopulateFuncArgs(FuncOp funcOp, LAGradContext &ctx) {
  auto loc = funcOp.getLoc().cast<FileLineColLoc>();
  std::string funDeclSrc;
  std::string line;
  std::fstream sourceFile(loc.getFilename().str());
  llvm::errs() << "looking at function " << funcOp.getName() << "\n";
  if (sourceFile.is_open()) {
    gotoLine(sourceFile, loc.getLine());
    getline(sourceFile, funDeclSrc);
    while (funDeclSrc.find(')') == std::string::npos) {
      getline(sourceFile, line);
      funDeclSrc += line;
    }

    auto start = funDeclSrc.find('(');
    auto argend = funDeclSrc.find(')');
    SmallVector<std::string> tokens;
    assert(argend != std::string::npos && "Failed to find closing paren");
    start++;
    auto end = funDeclSrc.find(',');
    std::string subs;
    while (end != std::string::npos) {
      subs = funDeclSrc.substr(start, end - start);
      tokens.push_back(ltrim(subs));
      start = end + 1;
      end = funDeclSrc.find(',', start);
    }
    subs = funDeclSrc.substr(start, argend - start);
    tokens.push_back(ltrim(subs));

    for (size_t i = 0; i < tokens.size(); i++) {
      ctx.debug_names[funcOp.getArgument(i)] =
          tokens[i].substr(0, tokens[i].find(':'));
    }
    for (auto pair : ctx.debug_names) {
      llvm::errs() << pair.first << " -> " << pair.second << "\n";
    }
  } else {
    llvm::errs() << "Failed to open file\n";
  }
}

} // namespace mlir