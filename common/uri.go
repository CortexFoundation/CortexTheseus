package common

type Scheme int

const (
	LocalStorage Scheme = 1
	ServerInfer  Scheme = 2
)

type URI struct {
	scheme Scheme
	path   string
}

func ParseURI(uri string) (URI, error) {
	if uri[0:7] == "locstr:" {
		return URI{
			scheme: LocalStorage,
			path:   uri[7:],
		}, nil
	}

	return URI{
		scheme: ServerInfer,
		path:   uri,
	}, nil
}
