// The tfproviderlintx command is a static checker for Terraform Providers that includes extra checks.
//
// Each analyzer flag name is preceded by the analyzer name: -NAME.flag.
// In addition, the -NAME flag itself controls whether the
// diagnostics of that analyzer are displayed. (A disabled analyzer may yet
// be run if it is required by some other analyzer that is enabled.)
package main

import (
	"github.com/bflad/tfproviderlint/helper/cmdflags"
	"github.com/bflad/tfproviderlint/passes"
	"github.com/bflad/tfproviderlint/xpasses"
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/multichecker"
)

func main() {
	cmdflags.AddVersionFlag()

	var analyzers []*analysis.Analyzer
	analyzers = append(analyzers, passes.AllChecks...)
	analyzers = append(analyzers, xpasses.AllChecks...)
	multichecker.Main(analyzers...)
}
