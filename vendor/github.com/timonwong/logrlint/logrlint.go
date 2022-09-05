package logrlint

import (
	"flag"
	"fmt"
	"go/ast"
	"go/types"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/types/typeutil"
)

const Doc = `Checks key valur pairs for common logger libraries (logr,klog,zap).`

func NewAnalyzer() *analysis.Analyzer {
	l := &logrlint{
		enable: loggerCheckersFlag{
			newStringSet(defaultEnabledCheckers...),
		},
	}

	a := &analysis.Analyzer{
		Name:     "logrlint",
		Doc:      Doc,
		Run:      l.run,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	}

	checkerKeys := strings.Join(loggerCheckersByName.Keys(), ",")
	a.Flags.Init("logrlint", flag.ExitOnError)
	a.Flags.BoolVar(&l.disableAll, "disableall", false, "disable all logger checkers")
	a.Flags.Var(&l.disable, "disable", fmt.Sprintf("comma-separated list of disabled logger checker (%s)", checkerKeys))
	a.Flags.Var(&l.enable, "enable", fmt.Sprintf("comma-separated list of enabled logger checker (%s)", checkerKeys))
	return a
}

type logrlint struct {
	disableAll bool               // flag -disableall
	disable    loggerCheckersFlag // flag -disable
	enable     loggerCheckersFlag // flag -enable
}

func (l *logrlint) isCheckerDisabled(name string) bool {
	if l.disableAll {
		return !l.enable.Has(name)
	}
	if l.disable.Has(name) {
		return true
	}
	return !l.enable.Has(name)
}

func (l *logrlint) getLoggerFuncs(pkgPath string) stringSet {
	for name, entry := range loggerCheckersByName {
		if l.isCheckerDisabled(name) {
			// Skip ignored logger checker.
			continue
		}

		if entry.packageImport == pkgPath {
			return entry.funcs
		}

		if strings.HasSuffix(pkgPath, "/vendor/"+entry.packageImport) {
			return decorateVendoredFuncs(entry.funcs, pkgPath, entry.packageImport)
		}
	}

	return nil
}

func decorateVendoredFuncs(entryFuncs stringSet, currentPkgImport, packageImport string) stringSet {
	funcs := make(stringSet, len(entryFuncs))
	for fn := range entryFuncs {
		lastDot := strings.LastIndex(fn, ".")
		if lastDot == -1 {
			continue // invalid pattern
		}

		importOrReceiver := fn[:lastDot]
		fnName := fn[lastDot+1:]

		if strings.HasPrefix(importOrReceiver, "(") { // is receiver
			if !strings.HasSuffix(importOrReceiver, ")") {
				continue // invalid pattern
			}

			var pointerIndicator string
			if strings.HasPrefix(importOrReceiver[1:], "*") { // pointer type
				pointerIndicator = "*"
			}

			leftOver := strings.TrimPrefix(importOrReceiver, "("+pointerIndicator+packageImport+".")
			importOrReceiver = fmt.Sprintf("(%s%s.%s", pointerIndicator, currentPkgImport, leftOver)
		} else { // is import
			importOrReceiver = currentPkgImport
		}

		fn = fmt.Sprintf("%s.%s", importOrReceiver, fnName)
		funcs.Insert(fn)
	}
	return funcs
}

func (l *logrlint) isValidLoggerFunc(fn *types.Func) bool {
	pkg := fn.Pkg()
	if pkg == nil {
		return false
	}

	funcs := l.getLoggerFuncs(pkg.Path())
	return funcs.Has(fn.FullName())
}

func (l *logrlint) checkLoggerArguments(pass *analysis.Pass, call *ast.CallExpr) {
	fn, _ := typeutil.Callee(pass.TypesInfo, call).(*types.Func)
	if fn == nil {
		return // function pointer is not supported
	}

	sig, ok := fn.Type().(*types.Signature)
	if !ok || !sig.Variadic() {
		return // not variadic
	}

	if !l.isValidLoggerFunc(fn) {
		return
	}

	// ellipsis args is hard, just skip
	if call.Ellipsis.IsValid() {
		return
	}

	params := sig.Params()
	nparams := params.Len() // variadic => nonzero
	args := params.At(nparams - 1)
	iface, ok := args.Type().(*types.Slice).Elem().(*types.Interface)
	if !ok || !iface.Empty() {
		return // final (args) param is not ...interface{}
	}

	startIndex := nparams - 1
	nargs := len(call.Args)
	variadicLen := nargs - startIndex
	if variadicLen%2 != 0 {
		firstArg := call.Args[startIndex]
		lastArg := call.Args[nargs-1]
		pass.Report(analysis.Diagnostic{
			Pos:      firstArg.Pos(),
			End:      lastArg.End(),
			Category: "logging",
			Message:  "odd number of arguments passed as key-value pairs for logging",
		})
	}
}

func (l *logrlint) run(pass *analysis.Pass) (interface{}, error) {
	insp := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
	nodeFilter := []ast.Node{
		(*ast.CallExpr)(nil),
	}
	insp.Preorder(nodeFilter, func(node ast.Node) {
		call := node.(*ast.CallExpr)

		typ := pass.TypesInfo.Types[call.Fun].Type
		if typ == nil {
			// Skip checking functions with unknown type.
			return
		}

		l.checkLoggerArguments(pass, call)
	})

	return nil, nil
}
