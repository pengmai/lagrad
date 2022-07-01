#include "Standalone/Utils.h"
#include <fstream>
#include <limits>
#include <string>

#define ENABLE_NAME_DEBUG true

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

void parseCommaSepParens(std::string src, SmallVector<std::string> &tokens) {
  std::string subs;
  auto start = src.find('(');
  auto argend = src.find(')');
  assert(argend != std::string::npos && "Failed to find closing paren");
  src = src.substr(0, argend + 1);
  start++;
  auto end = src.find(',');
  while (end != std::string::npos) {
    subs = src.substr(start, end - start);
    tokens.push_back(trim(subs));
    start = end + 1;
    end = src.find(',', start);
  }
  subs = src.substr(start, argend - start);
  tokens.push_back(trim(subs));
}

void DEBUGpopulateRegion(Region *region, std::fstream &sourceFile,
                         LAGradContext &ctx) {
  assert(region && "Region cannot be null");
  std::string line, src, subs;
  SmallVector<std::string> tokens;

  for (auto &op : region->getOps()) {
    gotoLine(sourceFile, op.getLoc().cast<FileLineColLoc>().getLine());
    getline(sourceFile, src);
    if (op.getNumResults() == 1) {
      assert(src.find('=') != std::string::npos &&
             "Expect op line to contain equals sign");
      subs = src.substr(0, src.find('='));
      ctx.debug_names[op.getResult(0)] = trim(subs);
    }

    if (auto forOp = dyn_cast_or_null<scf::ForOp>(&op)) {
      if (forOp.getNumResults() > 1) {
        assert(src.find('=') != std::string::npos &&
               "Expect op line to contain equals sign");
        subs = src.substr(0, src.find(':'));
        for (size_t i = 0; i < forOp.getNumResults(); i++) {
          ctx.debug_names[op.getResult(i)] =
              trim(subs) + "#" + std::to_string(i);
        }
      }
      while (src.find('{') == std::string::npos) {
        getline(sourceFile, line);
        src += line;
      }

      tokens.clear();
      parseCommaSepParens(src, tokens);
      assert(tokens.size() == forOp.getNumIterOperands() &&
             "Mismatch of parsed names and for op iter args");
      for (auto pair : llvm::zip(forOp.getRegionIterArgs(), tokens)) {
        subs = std::get<1>(pair).substr(0, std::get<1>(pair).find('='));
        ctx.debug_names[std::get<0>(pair)] = trim(subs);
      }

      DEBUGpopulateRegion(&forOp.getRegion(), sourceFile, ctx);
    } else if (auto ifOp = dyn_cast_or_null<scf::IfOp>(&op)) {
      DEBUGpopulateRegion(&ifOp.thenRegion(), sourceFile, ctx);
      DEBUGpopulateRegion(&ifOp.elseRegion(), sourceFile, ctx);
    }
  }

  for (auto pair : ctx.debug_names) {
    llvm::errs() << "name: '" << pair.second << "'\n";
  }
}

void DEBUGpopulateFunc(LAGradContext &ctx, FuncOp funcOp) {
  if (!ENABLE_NAME_DEBUG) {
    return;
  }
  // Clearing the map makes printing everything less verbose
  ctx.debug_names.clear();
  auto loc = funcOp.getLoc().cast<FileLineColLoc>();
  std::string src, line, subs;
  std::fstream sourceFile(loc.getFilename().str());
  // llvm::errs() << "\nlooking at function " << funcOp.getName() << "\n";
  if (!sourceFile.is_open()) {
    // funcOp.emitWarning() << "failed to open file\n";
    return;
  }
  gotoLine(sourceFile, loc.getLine());
  getline(sourceFile, src);
  while (src.find(')') == std::string::npos) {
    getline(sourceFile, line);
    src += line;
  }

  SmallVector<std::string> tokens;
  parseCommaSepParens(src, tokens);
  for (size_t i = 0; i < tokens.size(); i++) {
    subs = tokens[i].substr(0, tokens[i].find(':'));
    ctx.debug_names[funcOp.getArgument(i)] = trim(subs);
  }
  DEBUGpopulateRegion(funcOp.getCallableRegion(), sourceFile, ctx);
}

} // namespace mlir