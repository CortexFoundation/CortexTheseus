package tagflag

type parseOpt func(p *Parser)

// Don't perform default behaviour if -h or -help are passed.
func NoDefaultHelp() parseOpt {
	return func(p *Parser) {
		p.noDefaultHelp = true
	}
}

// Provides a description for the program to be shown in the usage message.
func Description(desc string) parseOpt {
	return func(p *Parser) {
		p.description = desc
	}
}

func Program(name string) parseOpt {
	return func(p *Parser) {
		p.program = name
	}
}

func ParseIntermixed(enabled bool) parseOpt {
	return func(p *Parser) {
		p.parseIntermixed = enabled
	}
}

func Parent(parent *Parser) parseOpt {
	return func(p *Parser) {
		p.parent = parent
	}
}
