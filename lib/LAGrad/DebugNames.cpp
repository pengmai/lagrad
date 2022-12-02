#include "LAGrad/Logger.h"
#include "LAGrad/Utils.h"
// #include <algorithm>
#include <fstream>
#include <limits>
#include <string>

#define ENABLE_NAME_DEBUG true

namespace mlir {
using namespace mlir;
using llvm::errs;
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
  size_t start = src.find('(');
  size_t argend = src.find(')');
  assert(argend != std::string::npos && "Failed to find closing paren");
  if (start + 1 == argend) {
    return;
  }
  src = src.substr(0, argend + 1);
  start++;
  size_t end = src.find(',');
  while (end != std::string::npos) {
    subs = src.substr(start, end - start);
    // Need to handle tensor encodings being separated with a comma
    if (subs.find('<') != std::string::npos &&
        subs.find('>') == std::string::npos) {
      size_t saved_end = end;
      end = src.find(',', end + 1);
      if (end == std::string::npos) {
        end = src.find(')', saved_end + 1);
        subs = src.substr(start, end - start);
        tokens.push_back(trim(subs));
        return;
      }
      subs = src.substr(start, end - start);
    }
    tokens.push_back(trim(subs));
    start = end + 1;
    end = src.find(',', start);
  }
  subs = src.substr(start, argend - start);
  tokens.push_back(trim(subs));
}

void DEBUGpopulateRegion(Region *region, std::fstream &sourceFile,
                         NameMap &debug_names) {
  assert(region && "Region cannot be null");
  std::string line, src, subs;
  SmallVector<std::string> tokens;

  for (auto &op : region->getOps()) {
    auto loc = op.getLoc().dyn_cast<FileLineColLoc>();
    if (!loc) {
      return;
    }
    gotoLine(sourceFile, loc.getLine());
    getline(sourceFile, src);
    if (op.getNumResults() == 1) {
      assert(src.find('=') != std::string::npos &&
             "Expect op line to contain equals sign");
      subs = src.substr(0, src.find('='));
      debug_names[op.getResult(0)] = trim(subs);
    }
    if (auto genericOp = dyn_cast_or_null<linalg::GenericOp>(&op)) {
      size_t idx = 0;
      for (auto arg : genericOp.getBodyRegion().getArguments()) {
        debug_names[arg] =
            debug_names[op.getResult(0)] + "#arg" + std::to_string(idx);
        idx++;
      }
      DEBUGpopulateRegion(&genericOp.getBodyRegion(), sourceFile, debug_names);
    } else if (auto forOp = dyn_cast_or_null<scf::ForOp>(&op)) {
      if (forOp.getNumResults() > 1) {
        assert(src.find('=') != std::string::npos &&
               "Expect op line to contain equals sign");
        subs = src.substr(0, src.find(':'));
        for (size_t i = 0; i < forOp.getNumResults(); i++) {
          debug_names[op.getResult(i)] = trim(subs) + "#" + std::to_string(i);
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
        debug_names[std::get<0>(pair)] = trim(subs);
      }

      DEBUGpopulateRegion(&forOp.getRegion(), sourceFile, debug_names);
    } else if (auto ifOp = dyn_cast_or_null<scf::IfOp>(&op)) {
      DEBUGpopulateRegion(&ifOp.getThenRegion(), sourceFile, debug_names);
      DEBUGpopulateRegion(&ifOp.getElseRegion(), sourceFile, debug_names);
    } else if (auto callOp = dyn_cast_or_null<func::CallOp>(&op)) {
      if (callOp.getNumResults() > 1) {
        assert(src.find('=') != std::string::npos &&
               "Expect op line to contain equals sign");
        subs = src.substr(0, src.find(':'));
        for (size_t i = 0; i < callOp.getNumResults(); i++) {
          debug_names[op.getResult(i)] = trim(subs) + "#" + std::to_string(i);
        }
      }
      auto moduleOp = op.getParentOfType<ModuleOp>();
      assert(moduleOp && "module op was null");
      auto calledFunc =
          moduleOp.lookupSymbol<func::FuncOp>(callOp.getCalleeAttr());
      assert(calledFunc && "failed to find called function");
      DEBUGpopulateFunc(debug_names, calledFunc);
    }
  }

  // SmallVector<std::string> names;
  // names.reserve(ctx.debug_names.size());
  // for (auto pair : ctx.debug_names) {
  //   names.push_back(pair.second);
  // }
  // std::sort(names.begin(), names.end());
  // llvm::errs() << "[";
  // for (auto name : names) {
  //   llvm::errs() << name;
  //   if (name != names.back()) {
  //     llvm::errs() << ", ";
  //   }
  // }
  // llvm::errs() << "]\n";
}

void DEBUGpopulateFunc(NameMap &debug_names, func::FuncOp funcOp) {
  if (!ENABLE_NAME_DEBUG || funcOp.empty()) {
    return;
  }
  // Clearing the map makes printing everything less verbose
  // ctx.debug_names.clear();
  auto loc = funcOp.getLoc().dyn_cast<FileLineColLoc>();
  if (!loc) {
    return;
  }
  std::string src, line, subs;
  std::fstream sourceFile(loc.getFilename().str());
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
  if (tokens.size() != funcOp.getNumArguments()) {
    errs() << "Debug name parsing failed for function: " << funcOp.getName()
           << "\n";
    return;
  }
  for (size_t i = 0; i < tokens.size(); i++) {
    subs = tokens[i].substr(0, tokens[i].find(':'));
    debug_names[funcOp.getArgument(i)] = trim(subs);
  }
  DEBUGpopulateRegion(funcOp.getCallableRegion(), sourceFile, debug_names);
}

void Logger::red(MESSAGETYPE message) {
  errs() << RED << message << "\n" << RESET;
}

void Logger::green(MESSAGETYPE message) {
  errs() << GREEN << message << "\n" << RESET;
}

void Logger::yellow(MESSAGETYPE message) {
  errs() << YELLOW << message << "\n" << RESET;
}

void Logger::blue(MESSAGETYPE message) {
  errs() << BOLDBLUE << message << "\n" << RESET;
}

void Logger::magenta(MESSAGETYPE message) {
  errs() << BOLDMAGENTA << message << "\n" << RESET;
}

void Logger::cyan(MESSAGETYPE message) {
  errs() << CYAN << message << "\n" << RESET;
}

} // namespace mlir