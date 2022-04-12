// Copyright 2017 Felix Lange <fjl@twurst.com>.
// Use of this source code is governed by the MIT license,
// which can be found in the LICENSE file.

package main

import (
	"fmt"
	"go/ast"
	"go/printer"
	"go/token"
	"go/types"
	"io"
	"strconv"
	"strings"

	. "github.com/garslo/gogen"
)

var (
	NIL     = Name("nil")
	intType = types.Universe.Lookup("int").Type()
)

type marshalMethod struct {
	mtyp        *marshalerType
	scope       *funcScope
	isUnmarshal bool
	// cached identifiers for map, slice conversions
	iterKey, iterVal Var
}

func newMarshalMethod(mtyp *marshalerType, isUnmarshal bool) *marshalMethod {
	s := newFuncScope(mtyp.scope)
	return &marshalMethod{
		mtyp:        mtyp,
		scope:       newFuncScope(mtyp.scope),
		isUnmarshal: isUnmarshal,
		iterKey:     Name(s.newIdent("k")),
		iterVal:     Name(s.newIdent("v")),
	}
}

func writeFunction(w io.Writer, fs *token.FileSet, fn Function) {
	printer.Fprint(w, fs, fn.Declaration())
	fmt.Fprintln(w)
}

// genUnmarshalJSON generates the UnmarshalJSON method.
func genUnmarshalJSON(mtyp *marshalerType) Function {
	var (
		m        = newMarshalMethod(mtyp, true)
		recv     = m.receiver()
		input    = Name(m.scope.newIdent("input"))
		intertyp = m.intermediateType(m.scope.newIdent(m.mtyp.orig.Obj().Name()))
		dec      = Name(m.scope.newIdent("dec"))
		json     = Name(m.scope.parent.packageName("encoding/json"))
	)
	fn := Function{
		Receiver:    recv,
		Name:        "UnmarshalJSON",
		ReturnTypes: Types{{TypeName: "error"}},
		Parameters:  Types{{Name: input.Name, TypeName: "[]byte"}},
		Body: []Statement{
			declStmt{intertyp},
			Declare{Name: dec.Name, TypeName: intertyp.Name},
			errCheck(CallFunction{
				Func:   Dotted{Receiver: json, Name: "Unmarshal"},
				Params: []Expression{input, AddressOf{Value: dec}},
			}),
		},
	}
	fn.Body = append(fn.Body, m.unmarshalConversions(dec, Name(recv.Name), "json")...)
	fn.Body = append(fn.Body, Return{Values: []Expression{NIL}})
	return fn
}

// genMarshalJSON generates the MarshalJSON method.
func genMarshalJSON(mtyp *marshalerType) Function {
	var (
		m        = newMarshalMethod(mtyp, false)
		recv     = m.receiver()
		intertyp = m.intermediateType(m.scope.newIdent(m.mtyp.orig.Obj().Name()))
		enc      = Name(m.scope.newIdent("enc"))
		json     = Name(m.scope.parent.packageName("encoding/json"))
	)
	fn := Function{
		Receiver:    recv,
		Name:        "MarshalJSON",
		ReturnTypes: Types{{TypeName: "[]byte"}, {TypeName: "error"}},
		Body: []Statement{
			declStmt{intertyp},
			Declare{Name: enc.Name, TypeName: intertyp.Name},
		},
	}
	fn.Body = append(fn.Body, m.marshalConversions(Name(recv.Name), enc, "json")...)
	fn.Body = append(fn.Body, Return{Values: []Expression{
		CallFunction{
			Func:   Dotted{Receiver: json, Name: "Marshal"},
			Params: []Expression{AddressOf{Value: enc}},
		},
	}})
	return fn
}

// genUnmarshalYAML generates the UnmarshalYAML method.
func genUnmarshalYAML(mtyp *marshalerType) Function {
	return genUnmarshalLikeYAML(mtyp, "YAML")
}

// genUnmarshalTOML generates the UnmarshalTOML method.
func genUnmarshalTOML(mtyp *marshalerType) Function {
	return genUnmarshalLikeYAML(mtyp, "TOML")
}

func genUnmarshalLikeYAML(mtyp *marshalerType, name string) Function {
	var (
		m         = newMarshalMethod(mtyp, true)
		recv      = m.receiver()
		unmarshal = Name(m.scope.newIdent("unmarshal"))
		intertyp  = m.intermediateType(m.scope.newIdent(m.mtyp.orig.Obj().Name()))
		dec       = Name(m.scope.newIdent("dec"))
		tag       = strings.ToLower(name)
	)
	fn := Function{
		Receiver:    recv,
		Name:        "Unmarshal" + name,
		ReturnTypes: Types{{TypeName: "error"}},
		Parameters:  Types{{Name: unmarshal.Name, TypeName: "func (interface{}) error"}},
		Body: []Statement{
			declStmt{intertyp},
			Declare{Name: dec.Name, TypeName: intertyp.Name},
			errCheck(CallFunction{Func: unmarshal, Params: []Expression{AddressOf{Value: dec}}}),
		},
	}
	fn.Body = append(fn.Body, m.unmarshalConversions(dec, Name(recv.Name), tag)...)
	fn.Body = append(fn.Body, Return{Values: []Expression{NIL}})
	return fn
}

// genMarshalYAML generates the MarshalYAML method.
func genMarshalYAML(mtyp *marshalerType) Function {
	return genMarshalLikeYAML(mtyp, "YAML")
}

// genMarshalTOML generates the MarshalTOML method.
func genMarshalTOML(mtyp *marshalerType) Function {
	return genMarshalLikeYAML(mtyp, "TOML")
}

func genMarshalLikeYAML(mtyp *marshalerType, name string) Function {
	var (
		m        = newMarshalMethod(mtyp, false)
		recv     = m.receiver()
		intertyp = m.intermediateType(m.scope.newIdent(m.mtyp.orig.Obj().Name()))
		enc      = Name(m.scope.newIdent("enc"))
		tag      = strings.ToLower(name)
	)
	fn := Function{
		Receiver:    recv,
		Name:        "Marshal" + name,
		ReturnTypes: Types{{TypeName: "interface{}"}, {TypeName: "error"}},
		Body: []Statement{
			declStmt{intertyp},
			Declare{Name: enc.Name, TypeName: intertyp.Name},
		},
	}
	fn.Body = append(fn.Body, m.marshalConversions(Name(recv.Name), enc, tag)...)
	fn.Body = append(fn.Body, Return{Values: []Expression{AddressOf{Value: enc}, NIL}})
	return fn
}

func (m *marshalMethod) receiver() Receiver {
	letter := strings.ToLower(m.mtyp.name[:1])
	r := Receiver{Name: m.scope.newIdent(letter), Type: Name(m.mtyp.name)}
	if m.isUnmarshal {
		r.Type = Star{Value: r.Type}
	}
	return r
}

func (m *marshalMethod) intermediateType(name string) Struct {
	s := Struct{Name: name}
	for _, f := range m.mtyp.Fields {
		if m.isUnmarshal && f.function != nil {
			continue // fields generated from functions cannot be assigned on unmarshal
		}
		typ := f.typ
		if m.isUnmarshal {
			typ = ensureNilCheckable(typ)
		}
		s.Fields = append(s.Fields, Field{
			Name:     f.name,
			TypeName: types.TypeString(typ, m.mtyp.scope.qualify),
			Tag:      f.tag,
		})
	}
	return s
}

func (m *marshalMethod) unmarshalConversions(from, to Var, format string) (s []Statement) {
	for _, f := range m.mtyp.Fields {
		if f.function != nil {
			continue // fields generated from functions cannot be assigned
		}

		accessFrom := Dotted{Receiver: from, Name: f.name}
		accessTo := Dotted{Receiver: to, Name: f.name}
		typ := ensureNilCheckable(f.typ)
		if !f.isRequired(format) {
			s = append(s, If{
				Condition: NotEqual{Lhs: accessFrom, Rhs: NIL},
				Body:      m.convert(accessFrom, accessTo, typ, f.origTyp),
			})
		} else {
			err := fmt.Sprintf("missing required field '%s' for %s", f.encodedName(format), m.mtyp.name)
			errors := m.scope.parent.packageName("errors")
			s = append(s, If{
				Condition: Equals{Lhs: accessFrom, Rhs: NIL},
				Body: []Statement{
					Return{
						Values: []Expression{
							CallFunction{
								Func:   Dotted{Receiver: Name(errors), Name: "New"},
								Params: []Expression{stringLit{err}},
							},
						},
					},
				},
			})
			s = append(s, m.convert(accessFrom, accessTo, typ, f.origTyp)...)
		}
	}
	return s
}

func (m *marshalMethod) marshalConversions(from, to Var, format string) (s []Statement) {
	for _, f := range m.mtyp.Fields {
		accessFrom := Dotted{Receiver: from, Name: f.name}
		accessTo := Dotted{Receiver: to, Name: f.name}
		if f.function != nil {
			s = append(s, m.convert(CallFunction{Func: accessFrom}, accessTo, f.origTyp, f.typ)...)
		} else {
			s = append(s, m.convert(accessFrom, accessTo, f.origTyp, f.typ)...)
		}
	}
	return s
}

func (m *marshalMethod) convert(from, to Expression, fromtyp, totyp types.Type) (s []Statement) {
	// Remove pointer introduced by ensureNilCheckable during field building.
	if isPointer(fromtyp) && !isPointer(totyp) {
		from = Star{Value: from}
		fromtyp = fromtyp.(*types.Pointer).Elem()
	} else if !isPointer(fromtyp) && isPointer(totyp) {
		from = AddressOf{Value: from}
		fromtyp = types.NewPointer(fromtyp)
	}
	// Generate the conversion.
	qf := m.mtyp.scope.qualify
	switch {
	case types.ConvertibleTo(fromtyp, totyp):
		s = append(s, Assign{Lhs: to, Rhs: simpleConv(from, fromtyp, totyp, qf)})
	case underlyingSlice(fromtyp) != nil:
		s = append(s, m.loopConv(from, to, sliceKV(fromtyp), sliceKV(totyp))...)
	case underlyingMap(fromtyp) != nil:
		s = append(s, m.loopConv(from, to, mapKV(fromtyp), mapKV(totyp))...)
	default:
		invalidConv(fromtyp, totyp, qf)
	}
	return s
}

type kvType struct {
	Type      types.Type
	Key, Elem types.Type
}

func mapKV(typ types.Type) kvType {
	maptyp := underlyingMap(typ)
	return kvType{typ, maptyp.Key(), maptyp.Elem()}
}

func sliceKV(typ types.Type) kvType {
	slicetyp := underlyingSlice(typ)
	return kvType{typ, intType, slicetyp.Elem()}
}

func (m *marshalMethod) loopConv(from, to Expression, fromTyp, toTyp kvType) (conv []Statement) {
	if hasSideEffects(from) {
		orig := from
		from = Name(m.scope.newIdent("tmp"))
		conv = []Statement{DeclareAndAssign{Lhs: from, Rhs: orig}}
	}
	// The actual conversion is a loop that assigns each element.
	inner := []Statement{
		Assign{Lhs: to, Rhs: makeExpr(toTyp.Type, from, m.scope.parent.qualify)},
		Range{
			Key:        m.iterKey,
			Value:      m.iterVal,
			RangeValue: from,
			Body: []Statement{Assign{
				Lhs: Index{Value: to, Index: simpleConv(m.iterKey, fromTyp.Key, toTyp.Key, m.scope.parent.qualify)},
				Rhs: simpleConv(m.iterVal, fromTyp.Elem, toTyp.Elem, m.scope.parent.qualify),
			}},
		},
	}
	// Preserve nil maps and slices when marshaling. This is not required for unmarshaling
	// methods because the field is already nil-checked earlier.
	if !m.isUnmarshal {
		inner = []Statement{If{
			Condition: NotEqual{Lhs: from, Rhs: NIL},
			Body:      inner,
		}}
	}
	return append(conv, inner...)
}

func simpleConv(from Expression, fromtyp, totyp types.Type, qf types.Qualifier) Expression {
	if types.AssignableTo(fromtyp, totyp) {
		return from
	}
	if !types.ConvertibleTo(fromtyp, totyp) {
		invalidConv(fromtyp, totyp, qf)
	}
	toname := types.TypeString(totyp, qf)
	if isPointer(totyp) {
		toname = "(" + toname + ")" // hack alert!
	}
	return CallFunction{Func: Name(toname), Params: []Expression{from}}
}

func invalidConv(from, to types.Type, qf types.Qualifier) {
	panic(fmt.Errorf("BUG: invalid conversion %s -> %s", types.TypeString(from, qf), types.TypeString(to, qf)))
}

func makeExpr(typ types.Type, lenfrom Expression, qf types.Qualifier) Expression {
	return CallFunction{Func: Name("make"), Params: []Expression{
		Name(types.TypeString(typ, qf)),
		CallFunction{Func: Name("len"), Params: []Expression{lenfrom}},
	}}
}

func errCheck(expr Expression) If {
	err := Name("err")
	return If{
		Init:      DeclareAndAssign{Lhs: err, Rhs: expr},
		Condition: NotEqual{Lhs: err, Rhs: NIL},
		Body:      []Statement{Return{Values: []Expression{err}}},
	}
}

// hasSideEffects returns whether an expression may have side effects.
func hasSideEffects(expr Expression) bool {
	switch expr := expr.(type) {
	case Var:
		return false
	case Dotted:
		return hasSideEffects(expr.Receiver)
	case Star:
		return hasSideEffects(expr.Value)
	case Index:
		return hasSideEffects(expr.Index) && hasSideEffects(expr.Value)
	default:
		return true
	}
}

type stringLit struct {
	V string
}

func (l stringLit) Expression() ast.Expr {
	return &ast.BasicLit{Kind: token.STRING, Value: strconv.Quote(l.V)}
}

type declStmt struct {
	d Declaration
}

func (ds declStmt) Statement() ast.Stmt {
	return &ast.DeclStmt{Decl: ds.d.Declaration()}
}
