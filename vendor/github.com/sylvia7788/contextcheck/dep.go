package contextcheck

import (
	"go/types"
	"sync/atomic"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/buildssa"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/ssa"
)

type pkgInfo struct {
	pkgPkg *packages.Package // to find references later
	ssaPkg *ssa.Package      // to find func which has been built
	refCnt int32             // reference count
}

type collector struct {
	m map[string]*pkgInfo
}

func newCollector(pkgs []*packages.Package) (c *collector) {
	c = &collector{
		m: make(map[string]*pkgInfo),
	}

	// self-reference
	for _, pkg := range pkgs {
		c.m[pkg.PkgPath] = &pkgInfo{
			pkgPkg: pkg,
			refCnt: 1,
		}
	}

	// import reference
	for _, pkg := range pkgs {
		for _, imp := range pkg.Imports {
			if val, ok := c.m[imp.PkgPath]; ok {
				val.refCnt++
			}
		}
	}

	return
}

func (c *collector) DecUse(pass *analysis.Pass) {
	curPkg, ok := c.m[pass.Pkg.Path()]
	if !ok {
		return
	}

	if atomic.AddInt32(&curPkg.refCnt, -1) != 0 {
		curPkg.ssaPkg = pass.ResultOf[buildssa.Analyzer].(*buildssa.SSA).Pkg
		return
	}

	var release func(info *pkgInfo)
	release = func(info *pkgInfo) {
		for _, pkg := range info.pkgPkg.Imports {
			if val, ok := c.m[pkg.PkgPath]; ok {
				if atomic.AddInt32(&val.refCnt, -1) == 0 {
					release(val)
				}
			}
		}

		info.pkgPkg = nil
		info.ssaPkg = nil
	}
	release(curPkg)
}

func (c *collector) GetFunction(f *ssa.Function) (ff *ssa.Function) {
	info, ok := c.m[f.Pkg.Pkg.Path()]
	if !ok {
		return
	}

	// without recv => get by Func
	recv := f.Signature.Recv()
	if recv == nil {
		ff = info.ssaPkg.Func(f.Name())
		return
	}

	// with recv => find in prog according to type
	ntp, ptp := getNamedType(recv.Type())
	if ntp == nil {
		return
	}
	sel := info.ssaPkg.Prog.MethodSets.MethodSet(ntp).Lookup(ntp.Obj().Pkg(), f.Name())
	if sel == nil {
		sel = info.ssaPkg.Prog.MethodSets.MethodSet(ptp).Lookup(ntp.Obj().Pkg(), f.Name())
	}
	if sel == nil {
		return
	}
	ff = info.ssaPkg.Prog.MethodValue(sel)
	return
}

func getNamedType(tp types.Type) (ntp *types.Named, ptp *types.Pointer) {
	switch t := tp.(type) {
	case *types.Named:
		ntp = t
		ptp = types.NewPointer(tp)
	case *types.Pointer:
		if n, ok := t.Elem().(*types.Named); ok {
			ntp = n
			ptp = t
		}
	}
	return
}
