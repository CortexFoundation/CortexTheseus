package logrlint

import "sort"

var (
	defaultEnabledCheckers = []string{"logr", "klog", "zap"}
	loggerCheckersByName   = loggerCheckerMap{
		"logr": {
			packageImport: "github.com/go-logr/logr",
			funcs: newStringSet(
				"(github.com/go-logr/logr.Logger).Error",
				"(github.com/go-logr/logr.Logger).Info",
				"(github.com/go-logr/logr.Logger).WithValues"),
		},
		"klog": {
			packageImport: "k8s.io/klog/v2",
			funcs: newStringSet(
				"k8s.io/klog/v2.InfoS",
				"k8s.io/klog/v2.InfoSDepth",
				"k8s.io/klog/v2.ErrorS",
				"(k8s.io/klog/v2.Verbose).InfoS",
				"(k8s.io/klog/v2.Verbose).InfoSDepth",
				"(k8s.io/klog/v2.Verbose).ErrorS",
			),
		},
		"zap": {
			packageImport: "go.uber.org/zap",
			funcs: newStringSet(
				"(*go.uber.org/zap.SugaredLogger).With",
				"(*go.uber.org/zap.SugaredLogger).Debugw",
				"(*go.uber.org/zap.SugaredLogger).Infow",
				"(*go.uber.org/zap.SugaredLogger).Warnw",
				"(*go.uber.org/zap.SugaredLogger).Errorw",
				"(*go.uber.org/zap.SugaredLogger).DPanicw",
				"(*go.uber.org/zap.SugaredLogger).Panicw",
				"(*go.uber.org/zap.SugaredLogger).Fatalw",
			),
		},
	}
)

type loggerChecker struct {
	packageImport string
	funcs         stringSet
}

type loggerCheckerMap map[string]loggerChecker

func (m loggerCheckerMap) HasKey(key string) bool {
	_, ok := m[key]
	return ok
}

func (m loggerCheckerMap) Keys() []string {
	names := make([]string, 0, len(m))
	for name := range m {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}
